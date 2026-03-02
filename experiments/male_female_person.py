"""
Male vs female person source experiment.

Source A is a male person; Source B is a female person. Ages are matched
within a 5-year window. The source label includes name, gender marker, and age
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.models import Source
from helpers.runner import parse_args, run_standard_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "male_female_person"
BOTH_NO_SOURCE = False
SHOW_CIRCULATION = False


def _wrap_person(person, group: str) -> Source:
    """Wrap a PersonSource in a Source so the display string is baked in."""
    display = f"{person.first_name} {person.last_name} ({person.gender}), aged {person.age}"
    src = Source(name=display)
    src._group = group
    return src


def generate_source_pair(entity, source_gen, args):
    male, female = source_gen.generate_male_female_pair()
    return _wrap_person(male, "male"), _wrap_person(female, "female")


def get_group_label(source):
    return getattr(source, "_group", "person")


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
