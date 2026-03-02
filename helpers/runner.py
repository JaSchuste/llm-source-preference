"""
Shared experiment runner for source preference evaluation scripts.

All experiment scripts import from this module to avoid duplicating the
main loop, argparse setup, data loading, and result saving logic.

Entry points:
  - run_standard_experiment  : standard two-table prompts
  - run_2tm_experiment       : three-table majority prompts (2 majority tables)
  - run_1tm_experiment       : two-table implicit-majority prompts (1 minority table)
  - run_explicit_experiment  : explicit source-trust prompts (no entity tables)
"""

import argparse
import json
import logging
import re
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from helpers.config import (
    ALTINST_SYSTEM_MESSAGE,
    ALTTOK_SYSTEM_MESSAGE,
    DEFAULT_SYSTEM_MESSAGE,
    EXPLICIT_MAX_MATCHUPS,
    EXPLICIT_SYSTEM_MESSAGE,
    REPETITION_AWARE_SYSTEM_MESSAGE,
    SOURCE_AWARE_SYSTEM_MESSAGE,
    LOG_SAMPLE_RESULTS,
    SNM_SYSTEM_MESSAGE,
)
from helpers.conflict_analyzer import extract_conflicts
from helpers.data_loader import (
    load_circulation_values,
    load_entities,
    load_government_templates,
    load_names,
    load_question_cache,
    load_source_templates,
    load_timeline_locations,
    save_question_cache,
)
from helpers.evaluation import evaluate_results
from helpers.model_inference import get_answer_probabilities, load_model
from helpers.models import Source, SourceGroup
from helpers.prompt_builder import (
    build_1tm_prompt,
    build_2tm_prompt,
    build_explicit_prompt,
    build_two_table_prompt,
    generate_question,
)
from helpers.source_generator import SourceGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NO_SOURCE sentinel
# ---------------------------------------------------------------------------

NO_SOURCE = Source(name="No source available")


def _is_no_source(source: Any) -> bool:
    return isinstance(source, Source) and source.name == "No source available"


# ---------------------------------------------------------------------------
# Control experiment detection
# ---------------------------------------------------------------------------

def _is_control_experiment(experiment_type: str) -> bool:
    """Return True when *experiment_type* is a control (no-source baseline) experiment.

    Matches ``control`` and ``control_NEUTRAL``, ``control_ALTINST``, etc.
    """
    return bool(re.fullmatch(r"control(_NEUTRAL|_ALTINST|_SNM|_ALTTOK)?", experiment_type))


_NUM_SAMPLE_ENTRIES = 6


def _log_sample_results(
    all_results: List[Dict[str, Any]],
    tokenizer: Any,
    system_message: str,
) -> None:
    """Log the first *_NUM_SAMPLE_ENTRIES* result entries for inspection.

    Each logged entry includes the full chat-template-applied prompt (i.e.
    exactly the string that is tokenized and fed to the model) alongside all
    other result fields.
    """
    if not LOG_SAMPLE_RESULTS:
        return
    n = min(_NUM_SAMPLE_ENTRIES, len(all_results))
    if n == 0:
        return
    logger.info("=" * 80)
    logger.info("SAMPLE RESULTS: first %d entries (with tokenized prompts)", n)
    logger.info("=" * 80)
    for idx, entry in enumerate(all_results[:n]):
        # Apply chat template to reconstruct the exact tokenizer input
        raw_prompt = entry.get("prompt", "")
        tokenized_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": raw_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        logger.info("-" * 80)
        logger.info("Entry %d / %d", idx + 1, n)
        logger.info("-" * 80)
        for key, value in entry.items():
            if key == "prompt":
                continue  # replaced by the tokenized version below
            logger.info("  %-20s : %s", key, value)
        logger.info("  %-20s :\n%s", "tokenized_prompt", tokenized_prompt)
    logger.info("=" * 80)


def _check_control_baseline(
    args: argparse.Namespace, model_name: str, experiment_type: str
) -> None:
    """Raise :exc:`FileNotFoundError` if the control baseline for *model_name* is missing.

    Control experiments and explicit experiments are exempt from this check.

    Parameters
    ----------
    args:
        Parsed CLI arguments (provides ``results_dir`` and ``model``).
    model_name:
        Short model name derived from ``args.model`` (used in file naming).
    experiment_type:
        Base experiment type string (without mode suffix).
    """
    suffix = get_experiment_suffix(args)
    exp_key = experiment_type + suffix

    if _is_control_experiment(exp_key):
        return  # This IS the control — no check needed

    if experiment_type.startswith("explicit_"):
        return  # Explicit experiments do not use the control reference

    ctrl_name = "control" + suffix
    ctrl_path = Path(args.results_dir) / "raw" / ctrl_name / f"neoqa_{model_name}.json"
    if not ctrl_path.exists():
        cmd = f"python experiments/control.py --model {args.model}"
        if getattr(args, "data_dir", "data") != "data":
            cmd += f" --data-dir {args.data_dir}"
        if getattr(args, "results_dir", "results") != "results":
            cmd += f" --results-dir {args.results_dir}"
        if getattr(args, "neutral", False):
            cmd += " --neutral"
        if getattr(args, "altinst", False):
            cmd += " --altinst"
        if getattr(args, "snm", False):
            cmd += " --snm"
        if getattr(args, "alttok", False):
            cmd += " --alttok"
        raise FileNotFoundError(
            f"Control baseline not found: {ctrl_path}\n"
            f"Run the control experiment first:\n  {cmd}"
        )


# ---------------------------------------------------------------------------
# Common CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse common command-line arguments shared by all experiment scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model identifier or local path",
    )
    parser.add_argument("--data-dir", default="data", dest="data_dir",
                        help="Directory containing data files (default: data)")
    parser.add_argument("--results-dir", default="results", dest="results_dir",
                        help="Root directory for saving results (default: results)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for source generation (default: 42)")
    parser.add_argument("--neutral", action="store_true",
                        help="Add a 'Can't be answered' neutral option")
    parser.add_argument("--altinst", action="store_true",
                        help="Use alternative instruction system message")
    parser.add_argument("--snm", action="store_true",
                        help="Source-not-mentioned mode (hides no-source labels)")
    parser.add_argument("--alttok", action="store_true",
                        help="Use numeric answer tokens (1/2/3) instead of (A/B/C)")
    parser.add_argument("--source-aware-prompt", action="store_true",
                        help="Instruct the model to assess source credibility")
    parser.add_argument("--repetition-aware-prompt", action="store_true",
                        help="Instruct the model to ignore simple repetition of information")
    parser.add_argument("--finetuned", action="store_true",
                        help="Apply training blacklist to avoid seen sources")
    return parser.parse_args()


def get_system_message(args: argparse.Namespace) -> str:
    """Select the appropriate system message based on experiment flags."""
    if args.altinst:
        return ALTINST_SYSTEM_MESSAGE
    if args.snm:
        return SNM_SYSTEM_MESSAGE
    if args.alttok:
        return ALTTOK_SYSTEM_MESSAGE
    if args.source_aware_prompt:
        return SOURCE_AWARE_SYSTEM_MESSAGE
    if args.repetition_aware_prompt:
        return REPETITION_AWARE_SYSTEM_MESSAGE
    return DEFAULT_SYSTEM_MESSAGE


def get_experiment_suffix(args: argparse.Namespace) -> str:
    """Build the file-name suffix encoding active experiment flags."""
    parts = []
    if getattr(args, "neutral", False):
        parts.append("NEUTRAL")
    if getattr(args, "altinst", False):
        parts.append("ALTINST")
    if getattr(args, "snm", False):
        parts.append("SNM")
    if getattr(args, "alttok", False):
        parts.append("ALTTOK")
    return ("_" + "_".join(parts)) if parts else ""


# ---------------------------------------------------------------------------
# Data and model loading
# ---------------------------------------------------------------------------

def load_data_and_model(
    args: argparse.Namespace,
) -> Tuple[List, SourceGenerator, Dict, Any, Any, int, str]:
    """Load all data files and the language model.

    Returns
    -------
    tuple
        ``(entities, source_gen, question_cache, model, tokenizer,
           batch_size, model_name)``
    """
    entities = load_entities(args.data_dir)
    question_cache = load_question_cache(args.data_dir)
    source_templates = load_source_templates(args.data_dir)
    timeline_locations = load_timeline_locations(args.data_dir)
    government_templates = load_government_templates(args.data_dir)
    circulation_values = load_circulation_values(args.data_dir)
    names_data = load_names(args.data_dir)

    training_blacklist = None
    if args.finetuned:
        bl_path = Path(args.data_dir) / "training_blacklist.json"
        if bl_path.exists():
            with open(bl_path, encoding="utf-8") as fh:
                training_blacklist = json.load(fh)

    source_gen = SourceGenerator(
        source_templates=source_templates,
        timeline_locations=timeline_locations,
        government_templates=government_templates,
        names_data=names_data,
        circulation_values=circulation_values,
        use_training_blacklist=args.finetuned,
        training_blacklist=training_blacklist,
    )

    tokenizer, model, batch_size = load_model(args.model)
    # Extract a short model name for use in file names
    model_name = args.model.rstrip("/").split("/")[-1]
    return entities, source_gen, question_cache, model, tokenizer, batch_size, model_name


# ---------------------------------------------------------------------------
# Result saving and evaluation
# ---------------------------------------------------------------------------

def _save_and_evaluate(
    all_results: List[Dict[str, Any]],
    experiment_type: str,
    args: argparse.Namespace,
    model_name: str,
) -> None:
    """Persist results and run evaluation.

    For control experiments: saves raw JSON to disk, then evaluates.
    For all other experiments: evaluates entirely in-memory (no raw file saved).
    """
    suffix = get_experiment_suffix(args)
    exp_key = experiment_type + suffix

    if _is_control_experiment(exp_key):
        # Control is the baseline — save raw results so other experiments can reference them
        raw_path = (
            Path(args.results_dir) / "raw" / exp_key / f"neoqa_{model_name}.json"
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2)
        logger.info("Saved %d raw results → %s", len(all_results), raw_path)
        evaluate_results(raw_path, results_dir=args.results_dir)
    else:
        # Non-control: evaluate in-memory, save only aggregated results
        evaluate_results(
            data=all_results,
            experiment_type_override=exp_key,
            results_dir=args.results_dir,
        )


# ---------------------------------------------------------------------------
# Standard two-table experiment runner
# ---------------------------------------------------------------------------

def run_standard_experiment(
    args: argparse.Namespace,
    experiment_type: str,
    generate_source_pair: Callable,
    get_group_label: Callable,
    both_no_source: bool,
    show_circulation: bool,
    get_regional_location: Optional[Callable] = None,
) -> None:
    """Run a standard two-table source authority experiment.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    experiment_type:
        Internal name used for file naming and result records (e.g. ``"control"``).
    generate_source_pair:
        ``(entity, source_gen, args) -> (src_a, src_b)``
        Called once per (entity, conflict, pair) to produce fresh sources.
    get_group_label:
        ``(source) -> str``
        Returns the group label (e.g. ``"newspaper"``) for a source object.
    both_no_source:
        When ``True`` both tables have no source attribution (control).
    show_circulation:
        Whether to include circulation / follower numbers in source labels.
    get_regional_location:
        Optional ``(src_a, src_b) -> Optional[str]`` that returns the location
        string for regionality experiments.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_name = args.model.rstrip("/").split("/")[-1]
    _check_control_baseline(args, model_name, experiment_type)

    (
        entities, source_gen, question_cache,
        model, tokenizer, batch_size, model_name,
    ) = load_data_and_model(args)
    system_message = get_system_message(args)

    all_prompts: List[str] = []
    all_meta: List[Dict[str, Any]] = []
    question_cache_dirty = False

    for entity in entities:
        conflicts = extract_conflicts(entity)
        if not conflicts:
            continue

        for conflict in conflicts:
            for pair_idx, (first_value, other_value) in enumerate(
                conflict.get_pairs()
            ):
                pair_id = f"{entity.id}_{conflict.field}_{pair_idx}"

                # Generate / cache question
                cache_key = f"{entity.entity_class.value}_{conflict.field}"
                if cache_key not in question_cache:
                    question_cache_dirty = True
                question = generate_question(
                    entity, conflict.field, question_cache, model, tokenizer
                )

                # Generate sources (fresh per pair)
                src_a, src_b = generate_source_pair(entity, source_gen, args)

                # Build variation list
                rs_fixed = random.choice([False, True])
                if args.neutral:
                    var_configs = [
                        (False, rs_fixed, 0), (False, rs_fixed, 1),
                        (False, rs_fixed, 2),
                        (True, rs_fixed, 0), (True, rs_fixed, 1),
                        (True, rs_fixed, 2),
                    ]
                else:
                    var_configs = [
                        (False, rs_fixed, 2), (True, rs_fixed, 2),
                    ]

                for var_idx, (reverse_tables, reverse_sources, neutral_pos) in enumerate(
                    var_configs
                ):
                    # Determine which source appears at each table position
                    base_a = src_b if reverse_sources else src_a
                    base_b = src_a if reverse_sources else src_b
                    actual_a = base_b if reverse_tables else base_a
                    actual_b = base_a if reverse_tables else base_b

                    # Value assignment: reverse_tables swaps A/B values
                    val_a = other_value if reverse_tables else first_value
                    val_b = first_value if reverse_tables else other_value

                    group_a = get_group_label(actual_a)
                    group_b = get_group_label(actual_b)

                    # is_no_source flags for SNM mode
                    src_a_ns = _is_no_source(actual_a) and not both_no_source
                    src_b_ns = _is_no_source(actual_b) and not both_no_source

                    regional_loc = (
                        get_regional_location(actual_a, actual_b)
                        if get_regional_location else None
                    )

                    prompt_str = build_two_table_prompt(
                        entity=entity,
                        conflict_field=conflict.field,
                        value_a=val_a,
                        value_b=val_b,
                        source_a_info=actual_a,
                        source_b_info=actual_b,
                        question=question,
                        neutral=args.neutral,
                        alttok=args.alttok,
                        neutral_position=neutral_pos,
                        snm=args.snm,
                        both_no_source=both_no_source,
                        source_a_is_no_source=src_a_ns,
                        source_b_is_no_source=src_b_ns,
                        show_circulation=show_circulation,
                        regional_location=regional_loc,
                    )

                    all_prompts.append(prompt_str)
                    all_meta.append({
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "entity_class": entity.entity_class.value,
                        "field": conflict.field,
                        "pair_id": pair_id,
                        "value_indices": [0, pair_idx + 1],
                        "first_value": first_value,
                        "other_value": other_value,
                        "variation": var_idx,
                        "reverse_tables": reverse_tables,
                        "reverse_sources": reverse_sources,
                        "value_a": val_a,
                        "value_b": val_b,
                        "source_a": actual_a.to_dict(),
                        "source_b": actual_b.to_dict(),
                        "experiment_type": experiment_type,
                        "prompt": prompt_str,
                        "group_a": group_a,
                        "group_b": group_b,
                    })

    if question_cache_dirty:
        save_question_cache(question_cache, args.data_dir)

    logger.info(
        "Generated %d prompts for experiment '%s'", len(all_prompts), experiment_type
    )

    # Run inference in explicit batches for progress tracking
    all_prob_tuples: list = []
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(all_prompts), batch_size),
                  total=num_batches, desc=experiment_type):
        batch = all_prompts[i : i + batch_size]
        probs = get_answer_probabilities(
            model, tokenizer, batch, system_message, args.neutral, args.alttok
        )
        all_prob_tuples.extend(probs)

    all_results: List[Dict[str, Any]] = []
    for meta, prob_tuple in zip(all_meta, all_prob_tuples):
        entry = {**meta, "model_name": model_name}
        if args.neutral:
            entry["prob_a"], entry["prob_b"], entry["prob_c"] = prob_tuple
        else:
            entry["prob_a"], entry["prob_b"] = prob_tuple
        all_results.append(entry)

    _log_sample_results(all_results, tokenizer, system_message)


    _save_and_evaluate(all_results, experiment_type, args, model_name)


# ---------------------------------------------------------------------------
# Verbose majority (3-table) experiment runner
# ---------------------------------------------------------------------------

def run_2tm_experiment(
    args: argparse.Namespace,
    experiment_type: str,
    generate_sources: Callable,
    get_group_label: Callable,
) -> None:
    """Run a verbose majority experiment with three separate tables.

    Parameters
    ----------
    generate_sources:
        ``(entity, source_gen, args) -> (high_src, low_src_group)``
        where *low_src_group* is a :class:`SourceGroup` of 2 sources.
    get_group_label:
        ``(source) -> str`` -- handles both single sources and SourceGroup.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_name = args.model.rstrip("/").split("/")[-1]
    _check_control_baseline(args, model_name, experiment_type)

    (
        entities, source_gen, question_cache,
        model, tokenizer, batch_size, model_name,
    ) = load_data_and_model(args)
    system_message = get_system_message(args)

    # Fixed 6 variation configs: 3 table orders × 2 answer orders
    variation_configs = [
        (["H", "L", "L"], "AH_BL"),
        (["L", "H", "L"], "AH_BL"),
        (["L", "L", "H"], "AH_BL"),
        (["H", "L", "L"], "AL_BH"),
        (["L", "H", "L"], "AL_BH"),
        (["L", "L", "H"], "AL_BH"),
    ]

    all_prompts: List[str] = []
    all_meta: List[Dict[str, Any]] = []
    question_cache_dirty = False

    for entity in entities:
        conflicts = extract_conflicts(entity)
        if not conflicts:
            continue

        for conflict in conflicts:
            for pair_idx, (first_value, other_value) in enumerate(
                conflict.get_pairs()
            ):
                pair_id = f"{entity.id}_{conflict.field}_{pair_idx}"

                cache_key = f"{entity.entity_class.value}_{conflict.field}"
                if cache_key not in question_cache:
                    question_cache_dirty = True
                question = generate_question(
                    entity, conflict.field, question_cache, model, tokenizer
                )

                high_src, low_src_group = generate_sources(entity, source_gen, args)

                # first_value → high-authority (minority) claim
                # other_value → majority claim
                value_high = first_value
                value_low = other_value

                for var_idx, (table_order, answer_order) in enumerate(
                    variation_configs
                ):
                    neutral_pos = var_idx % 3 if args.neutral else 2

                    prompt_str = build_2tm_prompt(
                        entity=entity,
                        conflict_field=conflict.field,
                        value_high=value_high,
                        value_low=value_low,
                        high_auth_source_info=high_src,
                        low_auth_source_infos=low_src_group,
                        table_order=table_order,
                        answer_order=answer_order,
                        question=question,
                        neutral=args.neutral,
                        alttok=args.alttok,
                        neutral_position=neutral_pos,
                    )

                    # Source at answer position A and B
                    if answer_order == "AH_BL":
                        val_a, val_b = value_high, value_low
                        src_a, src_b = high_src, low_src_group
                    else:
                        val_a, val_b = value_low, value_high
                        src_a, src_b = low_src_group, high_src

                    group_a = get_group_label(src_a)
                    group_b = get_group_label(src_b)

                    all_prompts.append(prompt_str)
                    all_meta.append({
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "entity_class": entity.entity_class.value,
                        "field": conflict.field,
                        "pair_id": pair_id,
                        "value_indices": [0, pair_idx + 1],
                        "first_value": first_value,
                        "other_value": other_value,
                        "variation": var_idx,
                        "reverse_tables": False,
                        "reverse_sources": False,
                        "value_a": val_a,
                        "value_b": val_b,
                        "source_a": src_a.to_dict() if hasattr(src_a, "to_dict") else {"source": str(src_a)},
                        "source_b": src_b.to_dict() if hasattr(src_b, "to_dict") else {"source": str(src_b)},
                        "experiment_type": experiment_type,
                        "prompt": prompt_str,
                        "group_a": group_a,
                        "group_b": group_b,
                    })

    if question_cache_dirty:
        save_question_cache(question_cache, args.data_dir)

    logger.info(
        "Generated %d prompts for experiment '%s'", len(all_prompts), experiment_type
    )

    all_prob_tuples: list = []
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(all_prompts), batch_size),
                  total=num_batches, desc=experiment_type):
        batch = all_prompts[i : i + batch_size]
        probs = get_answer_probabilities(
            model, tokenizer, batch, system_message, args.neutral, args.alttok
        )
        all_prob_tuples.extend(probs)

    all_results: List[Dict[str, Any]] = []
    for meta, prob_tuple in zip(all_meta, all_prob_tuples):
        entry = {**meta, "model_name": model_name}
        if args.neutral:
            entry["prob_a"], entry["prob_b"], entry["prob_c"] = prob_tuple
        else:
            entry["prob_a"], entry["prob_b"] = prob_tuple
        all_results.append(entry)

    _log_sample_results(all_results, tokenizer, system_message)


    _save_and_evaluate(all_results, experiment_type, args, model_name)


# ---------------------------------------------------------------------------
# Implicit majority (2-table with source lists) experiment runner
# ---------------------------------------------------------------------------

def run_1tm_experiment(
    args: argparse.Namespace,
    experiment_type: str,
    generate_sources: Callable,
    get_group_label: Callable,
) -> None:
    """Run an implicit majority experiment with two tables and source lists.

    The majority side is presented as a comma-separated list of source names
    in a single table caption.

    Parameters
    ----------
    generate_sources:
        ``(entity, source_gen, args) -> (high_src, low_src_group)``
    get_group_label:
        ``(source) -> str``
    """
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_name = args.model.rstrip("/").split("/")[-1]
    _check_control_baseline(args, model_name, experiment_type)

    (
        entities, source_gen, question_cache,
        model, tokenizer, batch_size, model_name,
    ) = load_data_and_model(args)
    system_message = get_system_message(args)

    # 2 variations: swap_tables=False, swap_tables=True
    swap_configs = [False, True]

    all_prompts: List[str] = []
    all_meta: List[Dict[str, Any]] = []
    question_cache_dirty = False

    for entity in entities:
        conflicts = extract_conflicts(entity)
        if not conflicts:
            continue

        for conflict in conflicts:
            for pair_idx, (first_value, other_value) in enumerate(
                conflict.get_pairs()
            ):
                pair_id = f"{entity.id}_{conflict.field}_{pair_idx}"

                cache_key = f"{entity.entity_class.value}_{conflict.field}"
                if cache_key not in question_cache:
                    question_cache_dirty = True
                question = generate_question(
                    entity, conflict.field, question_cache, model, tokenizer
                )

                high_src, low_src_group = generate_sources(entity, source_gen, args)
                value_high = first_value
                value_low = other_value

                for var_idx, swap_tables in enumerate(swap_configs):
                    neutral_pos = var_idx % 3 if args.neutral else 2

                    prompt_str = build_1tm_prompt(
                        entity=entity,
                        conflict_field=conflict.field,
                        value_high=value_high,
                        value_low=value_low,
                        high_source_info=high_src,
                        low_source_infos=low_src_group,
                        swap_tables=swap_tables,
                        question=question,
                        neutral=args.neutral,
                        alttok=args.alttok,
                        neutral_position=neutral_pos,
                    )

                    if not swap_tables:
                        val_a, val_b = value_high, value_low
                        src_a, src_b = high_src, low_src_group
                    else:
                        val_a, val_b = value_low, value_high
                        src_a, src_b = low_src_group, high_src

                    group_a = get_group_label(src_a)
                    group_b = get_group_label(src_b)

                    all_prompts.append(prompt_str)
                    all_meta.append({
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "entity_class": entity.entity_class.value,
                        "field": conflict.field,
                        "pair_id": pair_id,
                        "value_indices": [0, pair_idx + 1],
                        "first_value": first_value,
                        "other_value": other_value,
                        "variation": var_idx,
                        "reverse_tables": False,
                        "reverse_sources": False,
                        "value_a": val_a,
                        "value_b": val_b,
                        "source_a": src_a.to_dict() if hasattr(src_a, "to_dict") else {"source": str(src_a)},
                        "source_b": src_b.to_dict() if hasattr(src_b, "to_dict") else {"source": str(src_b)},
                        "experiment_type": experiment_type,
                        "prompt": prompt_str,
                        "group_a": group_a,
                        "group_b": group_b,
                    })

    if question_cache_dirty:
        save_question_cache(question_cache, args.data_dir)

    logger.info(
        "Generated %d prompts for experiment '%s'", len(all_prompts), experiment_type
    )

    all_prob_tuples: list = []
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(all_prompts), batch_size),
                  total=num_batches, desc=experiment_type):
        batch = all_prompts[i : i + batch_size]
        probs = get_answer_probabilities(
            model, tokenizer, batch, system_message, args.neutral, args.alttok
        )
        all_prob_tuples.extend(probs)

    all_results: List[Dict[str, Any]] = []
    for meta, prob_tuple in zip(all_meta, all_prob_tuples):
        entry = {**meta, "model_name": model_name}
        if args.neutral:
            entry["prob_a"], entry["prob_b"], entry["prob_c"] = prob_tuple
        else:
            entry["prob_a"], entry["prob_b"] = prob_tuple
        all_results.append(entry)

    _log_sample_results(all_results, tokenizer, system_message)


    _save_and_evaluate(all_results, experiment_type, args, model_name)


# ---------------------------------------------------------------------------
# Explicit trust experiment runner
# ---------------------------------------------------------------------------

def run_explicit_experiment(
    args: argparse.Namespace,
    experiment_type: str,
    generate_source_pair: Callable,
    get_group_label: Callable,
) -> None:
    """Run an explicit source-trust experiment (no entity tables).

    For each matchup a question is drawn at random from ``explicit_questions.txt``
    and two source objects are generated via *generate_source_pair*.  Each
    matchup produces two variations (source positions normal + swapped) so the
    model's positional bias can be distinguished from genuine preference.

    Up to :data:`~helpers.config.EXPLICIT_MAX_MATCHUPS` matchups are generated.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    experiment_type:
        Internal name used for file naming and result records.
    generate_source_pair:
        ``(source_gen, args) -> (src_a, src_b)``
    get_group_label:
        ``(source) -> str``
    """
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load explicit questions
    questions_path = Path(args.data_dir) / "explicit_questions.txt"
    if not questions_path.exists():
        raise FileNotFoundError(
            f"explicit_questions.txt not found in {args.data_dir}. "
            "Run download_dataset() first or place the file manually."
        )
    with open(questions_path, encoding="utf-8") as fh:
        questions = [line.strip() for line in fh if line.strip()]
    if not questions:
        raise ValueError("explicit_questions.txt is empty")

    # Load sources + model (no entities / question-cache needed)
    source_templates = load_source_templates(args.data_dir)
    timeline_locations = load_timeline_locations(args.data_dir)
    government_templates = load_government_templates(args.data_dir)
    circulation_values = load_circulation_values(args.data_dir)
    names_data = load_names(args.data_dir)

    training_blacklist = None
    if args.finetuned:
        bl_path = Path(args.data_dir) / "training_blacklist.json"
        if bl_path.exists():
            with open(bl_path, encoding="utf-8") as fh:
                training_blacklist = json.load(fh)

    source_gen = SourceGenerator(
        source_templates=source_templates,
        timeline_locations=timeline_locations,
        government_templates=government_templates,
        names_data=names_data,
        circulation_values=circulation_values,
        use_training_blacklist=args.finetuned,
        training_blacklist=training_blacklist,
    )

    tokenizer, model, batch_size = load_model(args.model)
    model_name = args.model.rstrip("/").split("/")[-1]

    all_prompts: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    for matchup_idx in range(EXPLICIT_MAX_MATCHUPS):
        question = random.choice(questions)
        src_a, src_b = generate_source_pair(source_gen, args)

        group_a = get_group_label(src_a)
        group_b = get_group_label(src_b)

        # Two variations: normal order, then swapped
        for var_idx, swap in enumerate([False, True]):
            first_src = src_b if swap else src_a
            second_src = src_a if swap else src_b
            first_group = group_b if swap else group_a
            second_group = group_a if swap else group_b

            prompt_str = build_explicit_prompt(
                question_text=question,
                source_a_info=first_src,
                source_b_info=second_src,
                neutral=False,
            )
            all_prompts.append(prompt_str)
            all_meta.append({
                "matchup_idx": matchup_idx,
                "variation": var_idx,
                "question": question,
                "source_a": first_src.to_dict() if hasattr(first_src, "to_dict") else {"name": str(first_src)},
                "source_b": second_src.to_dict() if hasattr(second_src, "to_dict") else {"name": str(second_src)},
                "experiment_type": experiment_type,
                "prompt": prompt_str,
                "group_a": first_group,
                "group_b": second_group,
            })

    logger.info(
        "Generated %d prompts for explicit experiment '%s'", len(all_prompts), experiment_type
    )

    all_prob_tuples_exp: list = []
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(all_prompts), batch_size),
                  total=num_batches, desc=experiment_type):
        batch = all_prompts[i : i + batch_size]
        probs = get_answer_probabilities(
            model, tokenizer, batch, EXPLICIT_SYSTEM_MESSAGE, neutral=False, alttok=False
        )
        all_prob_tuples_exp.extend(probs)

    all_results_exp: List[Dict[str, Any]] = []
    for meta, prob_tuple in zip(all_meta, all_prob_tuples_exp):
        entry = {**meta, "model_name": model_name}
        entry["prob_a"], entry["prob_b"] = prob_tuple
        all_results_exp.append(entry)

    _log_sample_results(all_results_exp, tokenizer, EXPLICIT_SYSTEM_MESSAGE)


    _save_and_evaluate(all_results_exp, experiment_type, args, model_name)
