"""
Control repeated (verbose majority baseline) experiment.

All three tables have no source attribution. One value is repeated
and treated as "no source majority".
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.models import Source, SourceGroup
from helpers.runner import parse_args, run_2tm_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "control_repeated"


def generate_sources(entity, source_gen, args):
    # All tables show "No source available", but minority and majority are
    # tracked separately so the evaluation can distinguish them.
    high_src = Source(name="No source available")
    high_src._group = "no_source"
    low_1 = Source(name="No source available")
    low_1._group = "no_source_repeated"
    low_2 = Source(name="No source available")
    low_2._group = "no_source_repeated"
    low_group = SourceGroup(sources=[low_1, low_2])
    return high_src, low_group


def get_group_label(source):
    if isinstance(source, SourceGroup):
        first = source.sources[0] if source.sources else None
        return getattr(first, "_group", "no_source_repeated")
    return getattr(source, "_group", "no_source")


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
