"""
Stated preference experiment: Generic person vs newspaper.

Source A is a generic person without a title; Source B is an artificial newspaper.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helpers.models import Source
from helpers.runner import parse_args, run_explicit_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "explicit_person_vs_newspaper"


def generate_source_pair(source_gen, args):
    person = source_gen.generate_generic_person()
    person_src = Source(name=person.simple_name)
    person_src._group = "person"
    newspaper = source_gen.generate_newspaper_source(exclude_timeline_id=None)
    newspaper._group = "newspaper"
    return person_src, newspaper


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
