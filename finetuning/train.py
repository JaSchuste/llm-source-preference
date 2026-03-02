#!/usr/bin/env python3
"""
Fine-tune a causal LM with LoRA to mitigate source preference bias.

Training objective (symmetric KL divergence):

    L = λ1 · symKL(f(2t), f'(2t))
      + λ2 · symKL(f(2t), f'(3t))

where
  f  = frozen reference model (base model weights, no LoRA)
  f' = adapted model (trained with LoRA)
  2t = two-table prompt
  3t = three-table prompt (minority once, majority repeated twice)

Term 1 keeps f'(2t) close to the reference distribution.
Term 2 makes f'(3t) match the reference 2-table distribution, removing
        the repetition/majority bias.

After main training a short refinement phase runs using only Term 2
(λ1=0, λ2=1) for `--refinement-steps` optimizer steps.

Fixed hyperparameters (best found in hparam search):
  λ1=0.6208, λ2=1-λ1
  lr=2e-4, lora_r=16, lora_alpha=16
  num_epochs=2, grad_accum=8, refinement_steps=300
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.config import DEFAULT_SYSTEM_MESSAGE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------

LAMBDA1 = 0.6208126425372757
LAMBDA2 = 1.0 - LAMBDA1
LEARNING_RATE = 2e-4
LORA_R = 16
LORA_ALPHA = 16
NUM_EPOCHS = 2
GRAD_ACCUM = 8
REFINEMENT_STEPS = 300
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
BATCH_SIZE = 1
MAX_LENGTH = 2048
EVAL_STEPS = 512

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ConsistencyDataset(Dataset):
    """Loads (2-table, 3-table) prompt pairs from a JSONL file."""

    def __init__(self, data_file: str, tokenizer, max_length: int = MAX_LENGTH) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data: List[Dict] = []
        with open(data_file) as f:
            for line in f:
                self.data.append(json.loads(line))
        logger.info("Loaded %d pairs from %s", len(self.data), data_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        p2 = self._apply_chat_template(item["prompt_2_table"])
        p3 = self._apply_chat_template(item["prompt_3_table"])

        enc2 = self.tokenizer(p2, truncation=True, max_length=self.max_length, return_tensors="pt", add_special_tokens=False)
        enc3 = self.tokenizer(p3, truncation=True, max_length=self.max_length, return_tensors="pt", add_special_tokens=False)

        return {
            "input_ids_2": enc2["input_ids"].squeeze(0),
            "attention_mask_2": enc2["attention_mask"].squeeze(0),
            "input_ids_3": enc3["input_ids"].squeeze(0),
            "attention_mask_3": enc3["attention_mask"].squeeze(0),
        }

    def _apply_chat_template(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _collate_fn(batch: List[Dict]) -> Dict:
    """Pad variable-length sequences in a batch."""
    max2 = max(x["input_ids_2"].size(0) for x in batch)
    max3 = max(x["input_ids_3"].size(0) for x in batch)

    ids2, mask2, ids3, mask3 = [], [], [], []
    for x in batch:
        p2 = max2 - x["input_ids_2"].size(0)
        ids2.append(F.pad(x["input_ids_2"], (0, p2), value=0))
        mask2.append(F.pad(x["attention_mask_2"], (0, p2), value=0))
        p3 = max3 - x["input_ids_3"].size(0)
        ids3.append(F.pad(x["input_ids_3"], (0, p3), value=0))
        mask3.append(F.pad(x["attention_mask_3"], (0, p3), value=0))

    return {
        "input_ids_2": torch.stack(ids2),
        "attention_mask_2": torch.stack(mask2),
        "input_ids_3": torch.stack(ids3),
        "attention_mask_3": torch.stack(mask3),
    }


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _get_answer_log_probs(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return log P(A) and log P(B) for the last token position.

    Returns shape [batch, 2].
    """
    tok_a = tokenizer.encode("A", add_special_tokens=False)[0]
    tok_b = tokenizer.encode("B", add_special_tokens=False)[0]
    answer_tokens = torch.tensor([tok_a, tok_b], device=input_ids.device)

    device_type = "cuda" if input_ids.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    logits = out.logits[:, -1, :]  # [batch, vocab]
    answer_logits = logits[:, answer_tokens]  # [batch, 2]
    return F.log_softmax(answer_logits, dim=-1)


def _sym_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence: 0.5*(KL(p||q) + KL(q||p))."""
    return 0.5 * (
        F.kl_div(log_q, log_p, reduction="batchmean", log_target=True)
        + F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)
    )


def _compute_loss(
    adapted_model,
    reference_model,
    tokenizer,
    batch: Dict,
    device: torch.device,
    lambda1: float,
    lambda2: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute weighted loss and individual terms."""
    ids2 = batch["input_ids_2"].to(device)
    msk2 = batch["attention_mask_2"].to(device)
    ids3 = batch["input_ids_3"].to(device)
    msk3 = batch["attention_mask_3"].to(device)

    with torch.no_grad():
        log_f_2 = _get_answer_log_probs(reference_model, tokenizer, ids2, msk2)

    log_fp_2 = _get_answer_log_probs(adapted_model, tokenizer, ids2, msk2)
    log_fp_3 = _get_answer_log_probs(adapted_model, tokenizer, ids3, msk3)

    t1 = _sym_kl(log_f_2, log_fp_2)
    t2 = _sym_kl(log_f_2, log_fp_3)

    loss = lambda1 * t1 + lambda2 * t2
    return loss, t1, t2


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_steps(
    adapted_model,
    reference_model,
    tokenizer,
    dataloader,
    optimizer,
    scheduler,
    device: torch.device,
    lambda1: float,
    lambda2: float,
    grad_accum: int,
    max_steps: Optional[int],
    eval_fn,
    global_step: int,
    desc: str = "Training",
) -> Tuple[Dict, int]:
    """Run training for one pass over dataloader (or until max_steps)."""
    adapted_model.train()
    reference_model.eval()

    total_loss = total_t1 = total_t2 = 0.0
    n_updates = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=desc)
    for step, batch in enumerate(pbar):
        loss, t1, t2 = _compute_loss(
            adapted_model, reference_model, tokenizer, batch, device,
            lambda1, lambda2,
        )

        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            n_updates += 1
            total_loss += loss.item()
            total_t1 += t1.item() * lambda1
            total_t2 += t2.item() * lambda2

            avg = total_loss / n_updates
            pbar.set_postfix({
                "loss": f"{avg:.4f}",
                "t1": f"{total_t1/n_updates:.4f}",
                "t2": f"{total_t2/n_updates:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if eval_fn is not None and EVAL_STEPS > 0 and global_step % EVAL_STEPS == 0:
                eval_fn(global_step)
                adapted_model.train()

            if max_steps is not None and global_step >= max_steps:
                break

    metrics = {
        "loss": total_loss / max(n_updates, 1),
        "loss_t1": total_t1 / max(n_updates, 1),
        "loss_t2": total_t2 / max(n_updates, 1),
    }
    return metrics, global_step


@torch.no_grad()
def _evaluate(
    adapted_model,
    reference_model,
    tokenizer,
    dataloader,
    device: torch.device,
    lambda1: float,
    lambda2: float,
) -> Dict:
    adapted_model.eval()
    reference_model.eval()
    total_loss = total_t1 = total_t2 = 0.0
    n = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        loss, t1, t2 = _compute_loss(
            adapted_model, reference_model, tokenizer, batch, device,
            lambda1, lambda2,
        )
        total_loss += loss.item()
        total_t1 += t1.item() * lambda1
        total_t2 += t2.item() * lambda2
        n += 1
    return {
        "loss": total_loss / max(n, 1),
        "loss_t1": total_t1 / max(n, 1),
        "loss_t2": total_t2 / max(n, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune model with KL-divergence loss.")
    p.add_argument("--model-name", default="google/gemma-3-4b-it")
    p.add_argument("--train-data", default="finetuning_data/train.jsonl")
    p.add_argument("--val-data", default="finetuning_data/val.jsonl")
    p.add_argument("--output-dir", default="finetuned_model")
    p.add_argument("--seed", type=int, default=42)
    # All hyperparams are fixed; expose them for transparency / override
    p.add_argument("--lambda1", type=float, default=LAMBDA1)
    p.add_argument("--lambda2", type=float, default=LAMBDA2)
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p.add_argument("--lora-r", type=int, default=LORA_R)
    p.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--refinement-steps", type=int, default=REFINEMENT_STEPS,
                   help="Extra optimizer steps after main training using only Term 2 (λ2=1).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, get_linear_schedule_with_warmup
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        logger.error("Missing dependency: %s. Run: pip install transformers peft", e)
        sys.exit(1)

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Save config
    config_dict = vars(args)
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Tokenizer
    logger.info("Loading tokenizer from %s ...", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Reference model (frozen)
    logger.info("Loading reference model (frozen) ...")
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    # Adapted model (to be fine-tuned)
    logger.info("Loading adapted model ...")
    adapted_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    ).to(device)
    adapted_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    adapted_model = get_peft_model(adapted_model, lora_config)
    adapted_model.print_trainable_parameters()

    # Datasets & dataloaders
    train_dataset = ConsistencyDataset(args.train_data, tokenizer)
    val_dataset = ConsistencyDataset(args.val_data, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=_collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=_collate_fn, num_workers=0,
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        adapted_model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_loss = float("inf")

    def save_if_best(val_loss: float) -> None:
        nonlocal best_val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best_model"
            adapted_model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            logger.info("  → New best val loss: %.4f  (saved to %s)", val_loss, best_path)

    def eval_fn(step: int) -> None:
        val_metrics = _evaluate(
            adapted_model, reference_model, tokenizer, val_loader, device,
            args.lambda1, args.lambda2,
        )
        logger.info(
            "[step %d] val_loss=%.4f  t1=%.4f  t2=%.4f",
            step, val_metrics["loss"], val_metrics["loss_t1"], val_metrics["loss_t2"],
        )
        save_if_best(val_metrics["loss"])

    # ---- Main training ----
    global_step = 0
    for epoch in range(args.num_epochs):
        logger.info("Epoch %d / %d", epoch + 1, args.num_epochs)
        metrics, global_step = _train_steps(
            adapted_model, reference_model, tokenizer, train_loader,
            optimizer, scheduler, device,
            args.lambda1, args.lambda2,
            args.grad_accum, None, eval_fn, global_step,
            desc=f"Epoch {epoch + 1}",
        )
        logger.info(
            "Epoch %d done: loss=%.4f  t1=%.4f  t2=%.4f",
            epoch + 1, metrics["loss"], metrics["loss_t1"], metrics["loss_t2"],
        )
        # End-of-epoch eval
        eval_fn(global_step)

    # ---- Refinement phase (Term 2 only: make f'(3t) match f(2t)) ----
    if args.refinement_steps > 0:
        logger.info(
            "Starting refinement phase: %d steps with λ1=0, λ2=1 ...",
            args.refinement_steps,
        )
        # Short dataset pass for refinement (shuffle a subset)
        import torch.utils.data as tud
        n_refine = min(args.refinement_steps * args.grad_accum, len(train_dataset))
        refine_idx = torch.randperm(len(train_dataset))[:n_refine].tolist()
        refine_subset = tud.Subset(train_dataset, refine_idx)
        refine_loader = DataLoader(
            refine_subset, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=_collate_fn, num_workers=0,
        )
        # New scheduler for refinement (no warmup)
        refine_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=args.refinement_steps,
        )
        _, global_step = _train_steps(
            adapted_model, reference_model, tokenizer, refine_loader,
            optimizer, refine_scheduler, device,
            lambda1=0.0, lambda2=1.0,
            grad_accum=args.grad_accum,
            max_steps=args.refinement_steps,
            eval_fn=None,
            global_step=global_step,
            desc="Refinement",
        )
        # Final eval after refinement
        eval_fn(global_step)

    # Save final model
    final_path = output_dir / "final_model"
    adapted_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("Final model saved to %s", final_path)
    logger.info("Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    main()
