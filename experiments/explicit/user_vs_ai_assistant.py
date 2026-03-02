"""
Stated preference experiment: User vs AI assistant.

Source A is labelled "user"; Source B is labelled "AI assistant".
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helpers.models import Source
from helpers.runner import parse_args, run_explicit_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "explicit_user_vs_ai_assistant"


def generate_source_pair(source_gen, args):
    user_src = Source(name="user")
    user_src._group = "user"
    ai_src = Source(name="AI assistant")
    ai_src._group = "ai_assistant"
    return user_src, ai_src


def get_group_label(source):
    return getattr(source, "_group", "unknown")


def main():
    args = parse_args()
    run_explicit_experiment(
        args=args,
        experiment_type=EXPERIMENT_TYPE,
        generate_source_pair=generate_source_pair,
        get_group_label=get_group_label,
    )


if __name__ == "__main__":
    main()
