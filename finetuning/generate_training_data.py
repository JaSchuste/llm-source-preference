#!/usr/bin/env python3
"""
Generate fine-tuning dataset for source preference mitigation.

For each sample, generates a pair of prompts:
  - prompt_2_table : standard two-table prompt (one minority, one majority source)
  - prompt_3_table : three-table prompt (minority once, majority repeated twice)

The training objective is to make the model's answer distribution on the
3-table prompt match its distribution on the 2-table prompt, removing the
repetition / majority bias.

Comparison types included:
  circulation, male_vs_female, person_vs_newspaper
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from tqdm import tqdm

# Allow running from the repo root or from this file's directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.conflict_analyzer import extract_conflicts
from helpers.data_loader import (
    load_circulation_values,
    load_entities,
    load_names,
    load_question_cache,
    load_source_templates,
    load_timeline_locations,
    save_question_cache,
)
from helpers.models import (
    Entity,
    EntityClass,
    Source,
    SourceGroup,
)
from helpers.prompt_builder import (
    build_two_table_prompt,
    build_2tm_prompt,
    create_entity_table,
    format_answer_choices,
    format_source_info,
    generate_question,
)
from helpers.source_generator import SourceGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Comparison types to generate training data for
COMPARISON_TYPES = [
    "circulation",
    "male_vs_female",
    "person_vs_newspaper",
]

_NO_SOURCE = Source(name="No source available")


def _is_no_source(src: Any) -> bool:
    return isinstance(src, Source) and src.name == "No source available"


# ---------------------------------------------------------------------------
# Source pair generation
# ---------------------------------------------------------------------------

def _make_source_pair(
    comparison_type: str,
    entity: Entity,
    source_gen: SourceGenerator,
) -> Optional[Tuple[Any, Any]]:
    """Return (source_minority, source_majority) for the given comparison type.

    source_minority appears ONCE in the 3-table prompt (high authority /
    reference side).  source_majority appears TWICE (the repeated side).

    Returns None if generation fails.
    """
    try:
        if comparison_type == "circulation":
            high = source_gen.generate_newspaper_source(
                exclude_timeline_id=entity.timeline_id,
                circulation=source_gen._get_circulation_by_tier("high"),
            )
            low = source_gen.generate_newspaper_source(
                exclude_timeline_id=entity.timeline_id,
                circulation=source_gen._get_circulation_by_tier("low"),
            )
            return high, low

        elif comparison_type == "male_vs_female":
            male, female = source_gen.generate_male_female_pair()
            male_src = Source(
                name=f"{male.first_name} {male.last_name} ({male.gender}), aged {male.age}"
            )
            female_src = Source(
                name=f"{female.first_name} {female.last_name} ({female.gender}), aged {female.age}"
            )
            return male_src, female_src

        elif comparison_type == "person_vs_newspaper":
            person = source_gen.generate_generic_person()
            newspaper = source_gen.generate_newspaper_source(
                exclude_timeline_id=entity.timeline_id
            )
            person_src = Source(name=person.simple_name)
            return person_src, newspaper

    except Exception as exc:
        logger.debug("Source generation failed for %s: %s", comparison_type, exc)
        return None

    return None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_prompts(
    entity: Entity,
    conflict_field: str,
    value_minority: str,
    value_majority: str,
    source_minority: Any,
    source_majority: Any,
    question_cache: Dict[str, str],
    answer_order: str,
    table_order_str: str,
) -> Tuple[str, str]:
    """Build the (prompt_2_table, prompt_3_table) pair.

    answer_order is 'AH_BL' (A=minority value, B=majority value) or
    'AL_BH' (A=majority value, B=minority value).

    table_order_str is one of 'MRR', 'RMR', 'RRM' describing the position of
    the minority ('M') and repeated majority ('R') tables in the 3-table prompt.
    """
    question = generate_question(entity, conflict_field, question_cache)

    # Determine (val_a, val_b) and (src_a, src_b) for the 2-table prompt
    if answer_order == "AH_BL":
        val_a, val_b = value_minority, value_majority
        src_a, src_b = source_minority, source_majority
    else:  # AL_BH
        val_a, val_b = value_majority, value_minority
        src_a, src_b = source_majority, source_minority

    # 2-table prompt
    is_no_src_a = _is_no_source(src_a)
    is_no_src_b = _is_no_source(src_b)
    prompt_2 = build_two_table_prompt(
        entity=entity,
        conflict_field=conflict_field,
        value_a=val_a,
        value_b=val_b,
        source_a_info=src_a,
        source_b_info=src_b,
        question=question,
        source_a_is_no_source=is_no_src_a,
        source_b_is_no_source=is_no_src_b,
    )

    # 3-table prompt — minority (M) appears once, majority (R) repeated twice
    table_order_list = [
        "H" if ch == "M" else "L"
        for ch in table_order_str
    ]
    low_sources = SourceGroup([source_majority, source_majority])

    prompt_3 = build_2tm_prompt(
        entity=entity,
        conflict_field=conflict_field,
        value_high=value_minority,
        value_low=value_majority,
        high_auth_source_info=source_minority,
        low_auth_source_infos=low_sources,
        table_order=table_order_list,
        answer_order=answer_order,
        question=question,
    )

    return prompt_2, prompt_3


# ---------------------------------------------------------------------------
# Blacklist tracking
# ---------------------------------------------------------------------------

class BlacklistTracker:
    """Tracks source components used during training for later exclusion."""

    def __init__(self) -> None:
        pass

    def track(self, source: Any) -> None:
        pass

    def to_dict(self) -> Dict:
        return {
            "description": (
                "Source components used in finetuning training data. "
                "These should NOT be used when running evaluation experiments."
            ),
        }


# ---------------------------------------------------------------------------
# Entity selection
# ---------------------------------------------------------------------------

def _select_entity_pool(
    entities: List[Entity],
    max_entities: Optional[int],
    auto_balance: bool,
) -> List[Entity]:
    """Select a (possibly balanced) subset of entities that have conflicts."""
    eligible = [
        e for e in entities
        if e.entity_class != EntityClass.MISCELLANEOUS
        and extract_conflicts(e)
    ]
    if not eligible:
        return []

    if auto_balance and max_entities is not None:
        class_map: Dict[EntityClass, List[Entity]] = defaultdict(list)
        for e in eligible:
            class_map[e.entity_class].append(e)
        for lst in class_map.values():
            random.shuffle(lst)
        selected: List[Entity] = []
        while len(selected) < max_entities and class_map:
            for ec in list(class_map.keys()):
                cands = class_map[ec]
                if not cands:
                    del class_map[ec]
                    continue
                selected.append(cands.pop())
                if len(selected) >= max_entities:
                    break
        return selected

    random.shuffle(eligible)
    if max_entities is not None:
        eligible = eligible[:max_entities]
    return eligible


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n_pairs: int,
    entity_pool: List[Entity],
    source_gen: SourceGenerator,
    question_cache: Dict[str, str],
    blacklist: BlacklistTracker,
    desc: str = "Generating pairs",
) -> List[Dict]:
    """Generate *n_pairs* training samples."""
    dataset: List[Dict] = []
    max_attempts = n_pairs * 15

    with tqdm(total=n_pairs, desc=desc) as pbar:
        attempts = 0
        while len(dataset) < n_pairs and attempts < max_attempts:
            attempts += 1
            try:
                entity = random.choice(entity_pool)
                comparison_type = random.choice(COMPARISON_TYPES)

                conflicts = extract_conflicts(entity)
                if not conflicts:
                    continue
                conflict = random.choice(conflicts)
                if len(conflict.values) < 2:
                    continue

                sampled = random.sample(conflict.values, 2)
                value_minority, value_majority = sampled[0], sampled[1]

                pair = _make_source_pair(comparison_type, entity, source_gen)
                if pair is None:
                    continue
                source_minority, source_majority = pair

                # Deterministic answer-order per entity+field pair to ensure
                # the same (A/B) mapping in both 2-table and 3-table prompts.
                seed_val = hash(f"{entity.id}_{conflict.field}") % 2
                answer_order = "AH_BL" if seed_val == 0 else "AL_BH"

                table_order_str = random.choice(["MRR", "RMR", "RRM"])

                prompt_2, prompt_3 = _build_prompts(
                    entity=entity,
                    conflict_field=conflict.field,
                    value_minority=value_minority,
                    value_majority=value_majority,
                    source_minority=source_minority,
                    source_majority=source_majority,
                    question_cache=question_cache,
                    answer_order=answer_order,
                    table_order_str=table_order_str,
                )

                blacklist.track(source_minority)
                blacklist.track(source_majority)

                dataset.append({
                    "prompt_2_table": prompt_2,
                    "prompt_3_table": prompt_3,
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "field": conflict.field,
                    "comparison_type": comparison_type,
                    "value_minority": value_minority,
                    "value_majority": value_majority,
                    "answer_order": answer_order,
                    "table_order": table_order_str,
                })
                pbar.update(1)

            except Exception as exc:
                logger.debug("Pair generation error: %s", exc)

    if len(dataset) < n_pairs:
        logger.warning(
            "Only generated %d / %d requested pairs.", len(dataset), n_pairs
        )
    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate (2-table, 3-table) training pairs for finetuning."
    )
    p.add_argument("--data-dir", default="data", help="Directory with data files")
    p.add_argument("--output-dir", default="finetuning_data", help="Output directory")
    p.add_argument("--train-size", type=int, default=1460, help="Number of training pairs")
    p.add_argument("--val-size", type=int, default=40, help="Number of validation pairs")
    p.add_argument("--max-entities", type=int, default=12, help="Max unique entities to draw from")
    p.add_argument("--auto-balance", action="store_true", default=True,
                   help="Balance entity selection across entity classes")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    logger.info("Seed: %d", args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir

    # Load data
    logger.info("Loading entities from %s ...", data_dir)
    entities = load_entities(data_dir)
    logger.info("Loaded %d entities.", len(entities))

    source_templates = load_source_templates(data_dir)
    timeline_locations = load_timeline_locations(data_dir)
    names = load_names(data_dir)
    circulation_values = load_circulation_values(data_dir)
    question_cache = load_question_cache(data_dir)

    source_gen = SourceGenerator(
        source_templates=source_templates,
        timeline_locations=timeline_locations,
        names_data=names,
        circulation_values=circulation_values,
    )

    # Select entity pool
    entity_pool = _select_entity_pool(
        entities, args.max_entities, args.auto_balance
    )
    logger.info("Entity pool: %d entities.", len(entity_pool))

    # Save entity pool metadata
    pool_meta = [
        {"entity_id": e.id, "entity_name": e.name, "entity_class": e.entity_class.value}
        for e in entity_pool
    ]
    with open(output_dir / "selected_entities.json", "w") as f:
        json.dump(pool_meta, f, indent=2)

    blacklist = BlacklistTracker()

    # Training set
    logger.info("Generating %d training pairs ...", args.train_size)
    train_data = generate_dataset(
        args.train_size, entity_pool, source_gen, question_cache, blacklist,
        desc="Training pairs",
    )
    with open(output_dir / "train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    logger.info("Saved %d training pairs to %s.", len(train_data), output_dir / "train.jsonl")

    # Validation set
    logger.info("Generating %d validation pairs ...", args.val_size)
    val_data = generate_dataset(
        args.val_size, entity_pool, source_gen, question_cache, blacklist,
        desc="Validation pairs",
    )
    with open(output_dir / "val.jsonl", "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")
    logger.info("Saved %d validation pairs to %s.", len(val_data), output_dir / "val.jsonl")

    # Save question cache
    save_question_cache(question_cache, data_dir)

    # Blacklist
    blacklist_path = output_dir / "training_blacklist.json"
    with open(blacklist_path, "w") as f:
        json.dump(blacklist.to_dict(), f, indent=2)
    logger.info("Saved training blacklist to %s.", blacklist_path)

    # Summary
    type_counts: Dict[str, int] = defaultdict(int)
    for item in train_data:
        type_counts[item["comparison_type"]] += 1
    logger.info("Training comparison type distribution:")
    for ct, cnt in sorted(type_counts.items()):
        logger.info("  %-35s %4d  (%.1f%%)", ct, cnt, 100 * cnt / max(len(train_data), 1))


if __name__ == "__main__":
    main()
