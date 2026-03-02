"""
No source vs government/institutional source experiment.

Source A has no attribution; Source B is a government or institutional source.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.models import Source
from helpers.runner import parse_args, run_standard_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "no_source_vs_government"
BOTH_NO_SOURCE = False
SHOW_CIRCULATION = False


def generate_source_pair(entity, source_gen, args):
    no_src = Source(name="No source available")
    no_src._group = "no_source"
    government = source_gen.generate_government_source(
        entity_class=entity.entity_class.value,
        entity_timeline_id=entity.timeline_id,
    )
    government._group = "government"
    return no_src, government


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
