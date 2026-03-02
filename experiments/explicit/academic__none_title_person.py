"""
Stated preference experiment: Academic title vs no title (person).

Source A is a person with an academic title; Source B is a person without any title.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helpers.models import Source
from helpers.runner import parse_args, run_explicit_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "explicit_academic__none_title_person"


def _wrap_academic(person, group: str) -> Source:
    src = Source(name=person.name)
    src._group = group
    return src


def generate_source_pair(source_gen, args):
    high, low = source_gen.generate_high_low_academia_pair()
    return _wrap_academic(high, "high_academia"), _wrap_academic(low, "low_academia")


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
