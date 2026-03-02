"""
Stated preference experiment: Internet-style (anonymous) vs traditional (known) social media user.

Source A is a social media user with an anonymous, internet-style username;
Source B is a social media user with a traditional, real-name-style username.
"""

import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helpers.runner import parse_args, run_explicit_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_TYPE = "explicit_traditional_internet_style_social_media"


def generate_source_pair(source_gen, args):
    anon, known = source_gen.generate_anon_known_social_media_pair()
    anon._group = "anonymous_user"
    known._group = "known_user"
    return anon, known


def get_group_label(source):
    return getattr(source, "_group", "social_media")


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
