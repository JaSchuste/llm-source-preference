"""
Stated preference experiment: Government source vs social media user.

Source A is a government/institutional source; Source B is a social media user
with no follower count shown.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helpers.runner import parse_args, run_explicit_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "explicit_government_vs_social_media"


def generate_source_pair(source_gen, args):
    government = source_gen.generate_government_source(
        entity_class="person",
        entity_timeline_id=None,
    )
    government._group = "government"
    social = source_gen.generate_social_media_user(popularity_tier=None)
    social._group = "social_media"
    return government, social


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
