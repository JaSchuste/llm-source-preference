"""
Stated preference experiment: Generic person vs government source.

Source A is a generic person without a title; Source B is a government/institutional source.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helpers.models import Source
from helpers.runner import parse_args, run_explicit_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "explicit_person_vs_government"


def generate_source_pair(source_gen, args):
    person = source_gen.generate_generic_person()
    person_src = Source(name=person.simple_name)
    person_src._group = "person"
    government = source_gen.generate_government_source(
        entity_class="person",
        entity_timeline_id=None,
    )
    government._group = "government"
    return person_src, government


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
