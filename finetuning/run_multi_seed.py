#!/usr/bin/env python3
"""
Multi-seed finetuning pipeline.

Runs the full training pipeline for multiple random seeds sequentially
(no cluster parallelization):

  For each seed:
    1. Generate training data  (finetuning/generate_training_data.py)
    2. Train the model          (finetuning/train.py)
    3. Merge the LoRA adapter   (finetuning/merge_adapter.py)

Fixed hyperparameters (from hparam search) are defaults in the respective scripts.
Seeds: [42, 123, 456, 789, 1011, 1213, 1415]

Usage:
    python finetuning/run_multi_seed.py \\
        --model-name google/gemma-3-4b-it \\
        --output-dir multi_seed_runs \\
        --data-dir data

Results layout:
    <output-dir>/
      run_0_seed_42/
        data/          train.jsonl, val.jsonl, training_blacklist.json
        model/         best_model/, final_model/, training_config.json
        merged/        (standalone merged model)
      run_1_seed_123/
        ...
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415]


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def _run(cmd: list, step_name: str) -> bool:
    """Run a subprocess command; return True on success."""
    logger.info("[%s] Running: %s", step_name, " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("[%s] FAILED with exit code %d", step_name, result.returncode)
        return False
    logger.info("[%s] Done.", step_name)
    return True


def _generate_data(
    seed: int,
    data_dir: Path,
    output_dir: Path,
    python: str,
    script_dir: Path,
) -> bool:
    if (output_dir / "train.jsonl").exists():
        logger.info("Training data already exists at %s, skipping.", output_dir)
        return True
    output_dir.mkdir(parents=True, exist_ok=True)
    return _run(
        [
            python,
            str(script_dir / "generate_training_data.py"),
            "--data-dir", str(data_dir),
            "--output-dir", str(output_dir),
            "--seed", str(seed),
        ],
        step_name=f"generate_data(seed={seed})",
    )


def _train(
    seed: int,
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    python: str,
    script_dir: Path,
) -> bool:
    if (output_dir / "best_model").exists() or (output_dir / "final_model").exists():
        logger.info("Model already trained at %s, skipping.", output_dir)
        return True
    output_dir.mkdir(parents=True, exist_ok=True)
    return _run(
        [
            python,
            str(script_dir / "train.py"),
            "--model-name", model_name,
            "--train-data", str(data_dir / "train.jsonl"),
            "--val-data", str(data_dir / "val.jsonl"),
            "--output-dir", str(output_dir),
            "--seed", str(seed),
        ],
        step_name=f"train(seed={seed})",
    )


def _merge(
    model_name: str,
    model_dir: Path,
    merged_dir: Path,
    python: str,
    script_dir: Path,
) -> bool:
    if merged_dir.exists():
        logger.info("Merged model already exists at %s, skipping.", merged_dir)
        return True

    # Prefer best_model; fall back to final_model
    adapter = model_dir / "best_model"
    if not adapter.exists():
        adapter = model_dir / "final_model"
    if not adapter.exists():
        logger.error("No adapter found in %s.", model_dir)
        return False

    return _run(
        [
            python,
            str(script_dir / "merge_adapter.py"),
            "--base-model", model_name,
            "--adapter-path", str(adapter),
            "--output-path", str(merged_dir),
        ],
        step_name=f"merge({adapter.name}→{merged_dir.name})",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run multi-seed finetuning pipeline (generate → train → merge)."
    )
    p.add_argument("--model-name", default="google/gemma-3-4b-it",
                   help="HuggingFace model name or local path")
    p.add_argument("--output-dir", default="multi_seed_runs",
                   help="Root directory for all run outputs")
    p.add_argument("--data-dir", default="data",
                   help="Directory containing the NeoQA data files")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                   help="Random seeds (one training run per seed)")
    p.add_argument("--skip-merge", action="store_true",
                   help="Skip LoRA adapter merging step")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    python = sys.executable
    script_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_dir)
    data_dir = Path(args.data_dir)

    logger.info("=" * 60)
    logger.info("Multi-seed finetuning pipeline")
    logger.info("  Model   : %s", args.model_name)
    logger.info("  Seeds   : %s", args.seeds)
    logger.info("  Output  : %s", output_root)
    logger.info("  Data    : %s", data_dir)
    logger.info("=" * 60)

    results = {}

    for i, seed in enumerate(args.seeds):
        run_dir = output_root / f"run_{i}_seed_{seed}"
        data_output = run_dir / "data"
        model_output = run_dir / "model"
        merged_output = run_dir / "merged"

        logger.info("")
        logger.info("─" * 60)
        logger.info("Run %d / %d  (seed=%d)", i + 1, len(args.seeds), seed)
        logger.info("─" * 60)

        ok_data = _generate_data(seed, data_dir, data_output, python, script_dir)
        if not ok_data:
            logger.error("Data generation failed for seed=%d. Skipping run.", seed)
            results[seed] = "data_failed"
            continue

        ok_train = _train(seed, args.model_name, data_output, model_output, python, script_dir)
        if not ok_train:
            logger.error("Training failed for seed=%d. Skipping merge.", seed)
            results[seed] = "train_failed"
            continue

        if not args.skip_merge:
            ok_merge = _merge(args.model_name, model_output, merged_output, python, script_dir)
            results[seed] = "success" if ok_merge else "merge_failed"
        else:
            results[seed] = "success (merge skipped)"

    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    for seed, status in results.items():
        logger.info("  seed=%-6d  %s", seed, status)
    logger.info("=" * 60)

    if any("failed" in v for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
