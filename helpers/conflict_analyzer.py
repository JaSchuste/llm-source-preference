"""
Conflict analysis utilities for the source preference evaluation framework.

This module extracts conflicting attribute values from NeoQA entities. A
"conflict" is an attribute field that has two or more distinct values,
which can then be used to construct evaluation prompts where different
sources claim different values.

Typical usage:
    from helpers.conflict_analyzer import extract_conflicts, count_total_pairs

    conflicts = extract_conflicts(entity)
    for conflict in conflicts:
        for value_a, value_b in conflict.get_pairs():
            # Build a prompt pitting value_a against value_b
            ...
"""

from typing import List

from helpers.models import Entity, Conflict
from helpers.config import EXCLUDED_FIELDS


# =============================================================================
# Conflict extraction
# =============================================================================

def extract_conflicts(entity: Entity) -> List[Conflict]:
    """Extract all conflicting attribute fields from an entity.

    An attribute field is considered a conflict if it has 2 or more distinct
    values. Certain metadata fields (id, entity_class, description,
    timeline_id) are always excluded.

    Special handling for the "spouse" field: it is only included if the
    entity has a marital_status attribute indicating a married or
    civil-partnership status. This avoids generating spouse-related
    questions for entities that are single/divorced/widowed.

    Args:
        entity: The Entity to analyze.

    Returns:
        List of Conflict objects, one per qualifying multi-valued field.
    """
    conflicts = []

    for field_name, values in entity.attributes.items():
        # Skip metadata fields that are not factual attributes
        if field_name in EXCLUDED_FIELDS:
            continue

        # Special case: only include spouse if the entity is currently married.
        # Use only the first marital_status value (the primary/current status).
        if field_name == "spouse":
            marital_values = entity.get_attribute("marital_status")
            if not marital_values:
                continue
            if marital_values[0] not in {"married", "in a civil partnership"}:
                continue

        # Only include fields with 2 or more values (i.e., actual conflicts)
        if len(values) >= 2:
            conflicts.append(Conflict(field=field_name, values=values))

    return conflicts


# =============================================================================
# Conflict checking
# =============================================================================

def has_conflicts(entity: Entity) -> bool:
    """Check whether an entity has any conflicting attribute fields.

    This is a convenience function that avoids building the full list
    of Conflict objects when you only need a boolean check.

    Args:
        entity: The Entity to check.

    Returns:
        True if the entity has at least one field with 2+ values
        (respecting the same exclusion rules as extract_conflicts).
    """
    return len(extract_conflicts(entity)) > 0


# =============================================================================
# Pair counting
# =============================================================================

def count_total_pairs(entities: List[Entity]) -> int:
    """Count the total number of conflict pairs across all entities.

    Each conflict generates (len(values) - 1) pairs by pairing the first
    value with each subsequent value (see Conflict.get_pairs). Each pair
    produces 2 prompt variations (2 table orderings, source assignment fixed
    randomly), so the total number of prompts is 2 * count_total_pairs(entities).

    This is useful for progress bars and resource estimation.

    Args:
        entities: List of Entity objects to analyze.

    Returns:
        Total number of conflict pairs (before variation expansion).

    Example:
        If an entity has a field with values ["a", "b", "c"], that field
        produces 2 pairs: (a, b) and (a, c). Multiply by 2 for the
        total prompt count.
    """
    total = 0
    for entity in entities:
        conflicts = extract_conflicts(entity)
        for conflict in conflicts:
            # Each conflict produces (n - 1) pairs where n = len(values)
            total += len(conflict.values) - 1
    return total
