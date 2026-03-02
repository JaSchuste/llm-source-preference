"""
Control experiment: no source vs no source.

Both tables present the same entity with conflicting values but without any
source attribution. This serves as the baseline C for computing Source Preference
(SP) in all other experiments.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.runner import NO_SOURCE, parse_args, run_standard_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "control"
BOTH_NO_SOURCE = True
SHOW_CIRCULATION = False


def generate_source_pair(entity, source_gen, args):
    return NO_SOURCE, NO_SOURCE


def get_group_label(source):
    return "no_source"


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
