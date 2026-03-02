"""
Traditional (known) vs internet-style (anonymous) social media user experiment.

Source A is an anonymous social media user with a CamelCase username
(e.g. @BoldEagle4821); 
Source B is a traditional social media user whose username reveals
their real name (e.g. @Jane_Doe).
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.runner import parse_args, run_standard_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "traditional_internet_style_social_media"
BOTH_NO_SOURCE = False
SHOW_CIRCULATION = False


def generate_source_pair(entity, source_gen, args):
    anon, known = source_gen.generate_anon_known_social_media_pair()
    anon._group = "anonymous_user"
    known._group = "known_user"
    return anon, known


def get_group_label(source):
    return getattr(source, "_group", "social_media")


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
