"""
No source vs newspaper majority experiment (verbose format).

Three tables are shown.
Source A is a newspaper source; 
Source B is "No source available" and repeated
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.models import Source, SourceGroup
from helpers.runner import parse_args, run_2tm_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "no_source_vs_newspaper_repeated"


def generate_sources(entity, source_gen, args):
    high_src = source_gen.generate_newspaper_source(
        exclude_timeline_id=entity.timeline_id
    )
    high_src._group = "newspaper"
    low_1 = Source(name="No source available")
    low_1._group = "no_source"
    low_2 = Source(name="No source available")
    low_2._group = "no_source"
    low_group = SourceGroup(sources=[low_1, low_2])
    return high_src, low_group


def get_group_label(source):
    if isinstance(source, SourceGroup):
        first = source.sources[0] if source.sources else None
        return getattr(first, "_group", "no_source")
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
