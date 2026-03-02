"""
No source vs newspaper experiment.

Source A has no attribution ("No source available"); Source B is an artificial
newspaper. Tests whether the presence of any named source influences model
responses compared to an unsourced claim.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.models import Source
from helpers.runner import NO_SOURCE, parse_args, run_standard_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "no_source_vs_newspaper"
BOTH_NO_SOURCE = False
SHOW_CIRCULATION = False


def generate_source_pair(entity, source_gen, args):
    no_src = Source(name="No source available")
    no_src._group = "no_source"
    newspaper = source_gen.generate_newspaper_source(
        exclude_timeline_id=entity.timeline_id
    )
    newspaper._group = "newspaper"
    return no_src, newspaper


def get_group_label(source):
    return getattr(source, "_group", "unknown")


def main():
    args = parse_args()
    run_standard_experiment(
        args=args,
        experiment_type=EXPERIMENT_TYPE,
        generate_source_pair=generate_source_pair,
        get_group_label=get_group_label,
        both_no_source=BOTH_NO_SOURCE,
        show_circulation=SHOW_CIRCULATION,
    )


if __name__ == "__main__":
    main()
