"""
Stated preference experiment: High vs low follower social media user.

Source A is a social media user with a high follower count; Source B is a social media
user with a low follower count.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helpers.runner import parse_args, run_explicit_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "explicit_high_low_followers_social_media"


def generate_source_pair(source_gen, args):
    high = source_gen.generate_social_media_user(popularity_tier="high")
    low = source_gen.generate_social_media_user(popularity_tier="low")
    high._group = "high_followers"
    low._group = "low_followers"
    return high, low


def get_group_label(source):
    return getattr(source, "_group", "social_media")


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
