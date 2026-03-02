"""
Government vs social media repeated experiment (2 table majority format, repeated source).

Three tables are shown. 
One table contains the government source; 
two separate tables each show the same social media user in the source attribution.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.models import SourceGroup
from helpers.runner import parse_args, run_2tm_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "government_vs_social_media_repeated"


def generate_sources(entity, source_gen, args):
    government = source_gen.generate_government_source(
        entity_class=entity.entity_class.value,
        entity_timeline_id=entity.timeline_id,
    )
    government._group = "government"
    # Same social media user used in both majority tables (fake majority)
    social = source_gen.generate_social_media_user(popularity_tier=None)
    social._group = "social_media"
    low_group = SourceGroup(sources=[social, social])
    return government, low_group


def get_group_label(source):
    if isinstance(source, SourceGroup):
        first = source.sources[0] if source.sources else None
        return getattr(first, "_group", "social_media")
    return getattr(source, "_group", "unknown")


def main():
    args = parse_args()
    run_2tm_experiment(
        args=args,
        experiment_type=EXPERIMENT_TYPE,
        generate_sources=generate_sources,
        get_group_label=get_group_label,
    )


if __name__ == "__main__":
    main()
