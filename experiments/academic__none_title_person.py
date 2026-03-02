"""
Academic title vs no title person experiment.

Source A is a person with a high-academic title (Dr. / Prof. / PhD suffix);
Source B is a person with a low/no academic title (Mr. / Ms. / Mrs.). 
Both have comparable age ranges (30–65).
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.models import Source
from helpers.runner import parse_args, run_standard_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "academic__none_title_person"
BOTH_NO_SOURCE = False
SHOW_CIRCULATION = False


def _wrap_academic(person, group: str) -> Source:
    """Wrap PersonSource with title-bearing display name baked in once."""
    display = person.name  # randomly picks Dr./Prof./PhD or Mr./Ms./Mrs.
    src = Source(name=display)
    src._group = group
    return src


def generate_source_pair(entity, source_gen, args):
    high, low = source_gen.generate_high_low_academia_pair()
    return _wrap_academic(high, "high_academia"), _wrap_academic(low, "low_academia")


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
