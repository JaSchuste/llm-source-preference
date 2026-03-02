"""
Data models for the source preference evaluation framework.

Defines typed data classes for entities, sources, conflicts, prompts, and results.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class EntityClass(Enum):
    """Entity classes in the NeoQA dataset."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    BUILDING = "building"
    EVENT = "event"
    PRODUCT = "product"
    ART = "art"
    MISCELLANEOUS = "miscellaneous"
    UNKNOWN = "unknown"


# =============================================================================
# Source data classes
# =============================================================================

@dataclass
class Source:
    """A news source (newspaper) with optional circulation data."""
    name: str
    circulation: Optional[int] = None
    country: Optional[str] = None
    regional_marker: Optional[str] = None  # 'regional' or 'non_regional'

    @property
    def has_circulation(self) -> bool:
        return self.circulation is not None and self.circulation > 0

    def to_dict(self) -> Dict[str, Any]:
        return {"Source": self.name, "circulation": self.circulation}


@dataclass
class SocialMediaUser:
    """A social media user with username and follower count."""
    username: str
    follower_count: int  # -1 means not displayed

    @property
    def show_follower_count(self) -> bool:
        return self.follower_count >= 0

    @property
    def name(self) -> str:
        return f"User @{self.username}"

    @property
    def popularity_tier(self) -> str:
        return "high" if self.follower_count >= 100 else "low"

    def to_dict(self) -> Dict[str, Any]:
        return {"Source": f"User @{self.username}", "follower_count": self.follower_count}


@dataclass
class GovernmentSource:
    """A government/institutional source generated from templates."""
    name: str
    location: str
    entity_class: str

    def to_dict(self) -> Dict[str, Any]:
        return {"Source": self.name, "location": self.location, "entity_class": self.entity_class}


@dataclass
class PersonSource:
    """A person source with name, age, and demographic attributes."""
    first_name: str
    last_name: str
    age: int
    gender: str  # 'M' or 'F'
    age_group: Optional[str] = None  # 'young', 'old', 'middle'
    academic_title: Optional[str] = None  # 'high' or 'low'

    @property
    def name(self) -> str:
        """Display name including relevant demographic info."""
        if self.academic_title:
            if self.academic_title == "high":
                choice = random.choice(["prefix_dr", "prefix_prof", "suffix_phd"])
                if choice == "prefix_dr":
                    return f"Dr. {self.first_name} {self.last_name}"
                elif choice == "prefix_prof":
                    return f"Prof. {self.first_name} {self.last_name}"
                else:
                    return f"{self.first_name} {self.last_name}, PhD"
            elif self.academic_title == "low":
                if self.gender == "M":
                    return f"Mr. {self.first_name} {self.last_name}"
                else:
                    title = random.choice(["Ms.", "Mrs."])
                    return f"{title} {self.first_name} {self.last_name}"

        # Default: show age and gender for demographic experiments
        return f"{self.first_name} {self.last_name} ({self.gender}), aged {self.age}"

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @property
    def simple_name(self) -> str:
        """Name without age/gender/title (for generic person experiments)."""
        return f"{self.first_name} {self.last_name}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Source": self.name,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "age": self.age,
            "gender": self.gender,
            "age_group": self.age_group,
            "academic_title": self.academic_title,
        }


@dataclass
class SourceGroup:
    """A group of sources supporting the same value (for majority experiments)."""
    sources: List[Union[Source, SocialMediaUser, GovernmentSource, PersonSource]]

    @property
    def name(self) -> str:
        if self.sources:
            return self.sources[0].name
        return "Source Group"

    @property
    def names(self) -> List[str]:
        return [s.name for s in self.sources]

    @property
    def primary_source(self):
        return self.sources[0]

    def to_dict(self) -> Dict[str, Any]:
        return {"sources": [s.to_dict() for s in self.sources], "count": len(self.sources)}


# =============================================================================
# Entity and conflict data classes
# =============================================================================

@dataclass
class Entity:
    """A NeoQA entity with attributes that may have conflicting values."""
    id: str
    name: str
    entity_class: EntityClass
    timeline_id: str
    attributes: Dict[str, List[str]] = field(default_factory=dict)
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create an Entity from a JSON dictionary."""
        entity_id = data.get("id", "")
        name = data.get("name", "")
        entity_class_str = data.get("entity_class", "unknown")
        timeline_id = data.get("timeline_id", "")
        description = data.get("description")

        try:
            entity_class = EntityClass(entity_class_str)
        except ValueError:
            entity_class = EntityClass.UNKNOWN

        metadata_fields = {"id", "name", "entity_class", "timeline_id", "description"}
        attributes = {}
        for key, value in data.items():
            if key not in metadata_fields:
                if isinstance(value, list):
                    attributes[key] = [str(v) for v in value]
                else:
                    attributes[key] = [str(value)]

        return cls(
            id=entity_id,
            name=name,
            entity_class=entity_class,
            timeline_id=timeline_id,
            attributes=attributes,
            description=description,
        )

    def get_attribute(self, field_name: str) -> List[str]:
        return self.attributes.get(field_name, [])

    def has_multiple_values(self, field_name: str) -> bool:
        return len(self.get_attribute(field_name)) > 1


@dataclass
class Conflict:
    """A field with multiple conflicting values."""
    field: str
    values: List[str]

    def get_pairs(self) -> List[Tuple[str, str]]:
        """Return all (first_value, other_value) pairs."""
        pairs = []
        first_value = self.values[0]
        for other_value in self.values[1:]:
            pairs.append((first_value, other_value))
        return pairs


# =============================================================================
# Prompt and result data classes
# =============================================================================

@dataclass
class PromptVariation:
    """A single prompt variation with metadata."""
    prompt: str
    value_a: str
    value_b: str
    source_a: Any
    source_b: Any
    reverse_tables: bool
    reverse_sources: bool
    variation_id: int


@dataclass
class EvaluationResult:
    """The result of evaluating one prompt variation."""
    entity_id: str
    entity_name: str
    entity_class: str
    field: str
    pair_id: str
    value_indices: List[int]
    first_value: str
    other_value: str
    variation: int
    reverse_tables: bool
    reverse_sources: bool
    value_a: str
    value_b: str
    source_a: Dict[str, Any]
    source_b: Dict[str, Any]
    comparison_type: str
    prompt: str
    prob_a: float
    prob_b: float
    model_name: str
    group_a: str
    group_b: str
    prob_c: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "entity_class": self.entity_class,
            "field": self.field,
            "pair_id": self.pair_id,
            "value_indices": self.value_indices,
            "first_value": self.first_value,
            "other_value": self.other_value,
            "variation": self.variation,
            "reverse_tables": self.reverse_tables,
            "reverse_sources": self.reverse_sources,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "comparison_type": self.comparison_type,
            "prompt": self.prompt,
            "model_name": self.model_name,
            "group_a": self.group_a,
            "group_b": self.group_b,
            "prob_a": self.prob_a,
            "prob_b": self.prob_b,
        }
        if self.prob_c is not None:
            result["prob_c"] = self.prob_c
        return result
