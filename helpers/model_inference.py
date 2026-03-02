"""
Model inference module for the source preference evaluation framework.

Provides straightforward HuggingFace model loading and single-token inference
for extracting answer-choice probabilities.  No cluster-specific logic, custom
device maps, or Accelerate -- just basic ``transformers`` with ``device_map="auto"``.
"""

import gc
import logging
import warnings
from typing import List, Tuple, Union

import torch
import numpy as np

from helpers.config import (
    TARGET_TOKENS,
    TARGET_TOKENS_WITH_NEUTRAL,
    ALT_TARGET_TOKENS,
    ALT_TARGET_TOKENS_WITH_NEUTRAL,
    BATCH_SIZE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Model loading
# =============================================================================

def load_model(model_name: str) -> Tuple:
    """Load a HuggingFace causal-LM and its tokenizer.

    The model is loaded with ``device_map="auto"`` and ``torch_dtype="auto"``
    (uses the dtype specified in the model config) and immediately put into
    eval mode.  The batch size is taken from ``BATCH_SIZE`` in
    ``helpers/config.py``.

    Args:
        model_name: HuggingFace model identifier (e.g.
            ``"Qwen/Qwen2-7B-Instruct"``) or a local path.

    Returns:
        A three-tuple ``(tokenizer, model, batch_size)``.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    logger.info("Loading tokenizer for %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token to eos_token (%s)", tokenizer.eos_token)

    logger.info("Loading model %s with device_map='auto' ...", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    model.eval()
    logger.info("Model loaded and set to eval mode")

    return tokenizer, model, BATCH_SIZE


# =============================================================================
# Inference
# =============================================================================

def get_answer_probabilities(
    model,
    tokenizer,
    prompts: List[str],
    system_message: str,
    neutral: bool = False,
    alttok: bool = False,
) -> Union[List[Tuple[float, float]], List[Tuple[float, float, float]]]:
    """Compute next-token probabilities over the answer tokens for a batch.

    Each prompt is formatted as a two-turn chat (system + user), tokenised with
    left-padding, and fed through a single ``model.generate`` call with
    ``max_new_tokens=1``.  The resulting logits over the target token vocabulary
    (``A``/``B`` or ``1``/``2``, optionally ``C``/``3``) are softmax-normalised
    to produce per-answer probabilities.

    Args:
        model: A HuggingFace ``AutoModelForCausalLM`` instance.
        tokenizer: The matching tokenizer.
        prompts: List of user-facing prompt strings (without chat template).
        system_message: System instruction prepended to every prompt.
        neutral: If ``True``, also compute the probability for the third
            (neutral) answer token.
        alttok: If ``True``, use ``1``/``2``/``3`` tokens instead of
            ``A``/``B``/``C``.

    Returns:
        A list of tuples -- ``(prob_a, prob_b)`` per prompt, or
        ``(prob_a, prob_b, prob_c)`` when *neutral* is ``True``.
    """
    return _run_inference(model, tokenizer, prompts, system_message, neutral, alttok)


def _run_inference(
    model,
    tokenizer,
    prompts: List[str],
    system_message: str,
    neutral: bool,
    alttok: bool,
) -> Union[List[Tuple[float, float]], List[Tuple[float, float, float]]]:
    """Run a single forward pass and extract answer probabilities."""

    # ----- Select target tokens -----
    if alttok:
        token_strs = ALT_TARGET_TOKENS_WITH_NEUTRAL if neutral else ALT_TARGET_TOKENS
    else:
        token_strs = TARGET_TOKENS_WITH_NEUTRAL if neutral else TARGET_TOKENS

    target_ids = [
        tokenizer.encode(tok, add_special_tokens=False)[-1] for tok in token_strs
    ]

    # ----- Apply chat template -----
    chat_inputs = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    # ----- Tokenise (left-padded for generation) -----
    # add_special_tokens=False: the chat template already inserted all special
    # tokens; re-adding them would produce duplicates (e.g. double BOS).
    tokenizer.padding_side = "left"
    tokenized = tokenizer(
        chat_inputs,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )

    device = next(model.parameters()).device
    tokenized = {
        k: v.to(device) for k, v in tokenized.items()
    }

    # ----- Generate a single token -----
    # Suppress the harmless "Setting `pad_token_id` to `eos_token_id`" warning
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*pad_token_id.*eos_token_id.*", category=UserWarning
        )
        outputs = model.generate(
            **tokenized,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # ----- Extract probabilities via softmax over target tokens -----
    # outputs.scores is a tuple with one tensor per generated token;
    # we only generated 1 token so index [0] gives shape (batch, vocab).
    logits = outputs.scores[0]  # (batch_size, vocab_size)

    # Gather logits for the target token ids
    target_ids_tensor = torch.tensor(target_ids, device=logits.device)
    target_logits = logits[:, target_ids_tensor]  # (batch_size, num_targets)

    # Softmax over just the target tokens
    probs = torch.nn.functional.softmax(target_logits, dim=-1)  # (batch_size, num_targets)
    probs_np = probs.cpu().numpy()

    # Build result tuples
    results: list = []
    if neutral:
        for row in probs_np:
            results.append((float(row[0]), float(row[1]), float(row[2])))
    else:
        for row in probs_np:
            results.append((float(row[0]), float(row[1])))

    # Cleanup tensors
    del outputs, logits, target_logits, probs, tokenized

    return results

