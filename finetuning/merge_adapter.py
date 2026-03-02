#!/usr/bin/env python3
"""
Merge a LoRA adapter into the base model to produce a standalone model.

Usage:
    python finetuning/merge_adapter.py \\
        --base-model google/gemma-3-4b-it \\
        --adapter-path path/to/best_model \\
        --output-path path/to/merged_model
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def merge_adapter(base_model: str, adapter_path: str, output_path: str) -> None:
    """Load base model + LoRA adapter, merge, and save a full standalone model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        logger.error("Missing dependency: %s. Run: pip install transformers peft", e)
        sys.exit(1)

    logger.info("Base model : %s", base_model)
    logger.info("Adapter    : %s", adapter_path)
    logger.info("Output     : %s", output_path)

    logger.info("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    logger.info("Loading base model (CPU) ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    logger.info("Loading LoRA adapter ...")
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging adapter weights ...")
    model = model.merge_and_unload()

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Saving merged model to %s ...", out)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    logger.info("Done. Merged model saved to %s", out)


def main() -> None:
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    p.add_argument("--base-model", default="google/gemma-3-4b-it",
                   help="Base model name or local path")
    p.add_argument("--adapter-path", required=True,
                   help="Path to the LoRA adapter directory (e.g. finetuned_model/best_model)")
    p.add_argument("--output-path", required=True,
                   help="Directory to save the merged standalone model")
    args = p.parse_args()
    merge_adapter(args.base_model, args.adapter_path, args.output_path)


if __name__ == "__main__":
    main()
