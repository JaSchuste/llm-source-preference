"""
Stated preference experiment: Middle-aged vs old person.

Source A is a middle-aged person (35–55); Source B is an older person.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helpers.models import Source
from helpers.runner import parse_args, run_explicit_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "explicit_middle_old_person"


def _wrap_person(person, group: str) -> Source:
    display = f"{person.first_name} {person.last_name} ({person.gender}), aged {person.age}"
    src = Source(name=display)
    src._group = group
    return src


def generate_source_pair(source_gen, args):
    middle, old = source_gen.generate_middle_old_pair()
    return _wrap_person(middle, "middle"), _wrap_person(old, "old")


def get_group_label(source):
    return getattr(source, "_group", "person")


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
