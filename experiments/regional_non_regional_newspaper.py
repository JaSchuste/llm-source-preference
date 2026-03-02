"""
Regional vs non-regional newspaper experiment.

Source A is a newspaper whose location is drawn from the entity's own timeline
(regional); 
Source B is a newspaper from a different timeline (non-regional).
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.runner import parse_args, run_standard_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "regional_non_regional_newspaper"
BOTH_NO_SOURCE = False
SHOW_CIRCULATION = False


def generate_source_pair(entity, source_gen, args):
    regional, non_regional = source_gen.generate_regional_non_regional_pair(
        entity_timeline_id=entity.timeline_id,
        entity_name=entity.name,
        entity_class=entity.entity_class.value,
    )
    regional._group = "regional"
    non_regional._group = "non_regional"
    return regional, non_regional


def get_group_label(source):
    return getattr(source, "_group", "newspaper")


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
