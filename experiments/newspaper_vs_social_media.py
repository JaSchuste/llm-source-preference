"""
Newspaper vs social media user experiment.

Source A is an artificial newspaper; 
Source B is a social media user with no follower count shown.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.runner import parse_args, run_standard_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "newspaper_vs_social_media"
BOTH_NO_SOURCE = False
SHOW_CIRCULATION = False


def generate_source_pair(entity, source_gen, args):
    newspaper = source_gen.generate_newspaper_source(
        exclude_timeline_id=entity.timeline_id
    )
    social = source_gen.generate_social_media_user(popularity_tier=None)
    newspaper._group = "newspaper"
    social._group = "social_media"
    return newspaper, social


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
