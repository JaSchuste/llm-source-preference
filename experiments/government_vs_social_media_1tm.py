"""
Government vs social media majority experiment (1 table majority format).

Two tables are shown. 
Source A is one government source
Source B is a list of two social media sources.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.models import SourceGroup
from helpers.runner import parse_args, run_1tm_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "government_vs_social_media_1tm"


def generate_sources(entity, source_gen, args):
    government = source_gen.generate_government_source(
        entity_class=entity.entity_class.value,
        entity_timeline_id=entity.timeline_id,
    )
    government._group = "government"
    social_1 = source_gen.generate_social_media_user(popularity_tier=None)
    social_1._group = "social_media"
    social_2 = source_gen.generate_social_media_user(popularity_tier=None)
    social_2._group = "social_media"
    low_group = SourceGroup(sources=[social_1, social_2])
    return government, low_group


def get_group_label(source):
    if isinstance(source, SourceGroup):
        first = source.sources[0] if source.sources else None
        return getattr(first, "_group", "social_media")
    return getattr(source, "_group", "unknown")


def main():
    args = parse_args()
    run_1tm_experiment(
        args=args,
        experiment_type=EXPERIMENT_TYPE,
        generate_sources=generate_sources,
        get_group_label=get_group_label,
    )


if __name__ == "__main__":
    main()
