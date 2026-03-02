"""
Stated preference experiment: High vs low circulation newspaper.

Source A is a newspaper with circulation in the top 25th percentile of US newspapers;
Source B is a newspaper with circulation in the bottom 25th percentile of US newspapers.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helpers.runner import parse_args, run_explicit_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "explicit_high_low_circulation_newspaper"


def generate_source_pair(source_gen, args):
    high = source_gen.generate_newspaper_source(
        exclude_timeline_id=None,
        circulation=source_gen._get_circulation_by_tier("high"),
    )
    low = source_gen.generate_newspaper_source(
        exclude_timeline_id=None,
        circulation=source_gen._get_circulation_by_tier("low"),
    )
    high._group = "high_circulation"
    low._group = "low_circulation"
    return high, low


def get_group_label(source):
    return getattr(source, "_group", "newspaper")


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
