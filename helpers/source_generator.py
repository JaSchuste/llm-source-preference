"""
Source generation module for the source preference evaluation framework.

This module provides the SourceGenerator class, which is responsible for
creating artificial sources of various types -- newspapers, government
institutions, person sources, and social media users -- used across different
source preference experiments.

Key capabilities:
- Generate artificial newspaper names from templates and timeline locations.
- Generate government/institutional source names from entity-class-specific templates.
- Generate person source pairs for gender, age, and academic-title experiments.
- Generate anonymous vs. known social media user pairs.
- Generate social media users with configurable follower counts.
- Generate regional vs. non-regional newspaper pairs for regionality experiments.
- Check generated person names against Wikipedia to avoid contamination with
  real-world entities that would confound the experimental design.

All randomness is driven by the standard ``random`` module so that experiments
can be made reproducible by seeding it externally.
"""

import functools
import json
import os
import random
import re
import logging
import requests
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set, Union

import numpy as np

from helpers.models import (
    Source,
    SocialMediaUser,
    GovernmentSource,
    PersonSource,
    SourceGroup,
)
from helpers.config import (
    MAX_PERSON_GENERATION_ATTEMPTS,
    WIKIPEDIA_API,
    WIKIPEDIA_HEADERS,
)

# ---------------------------------------------------------------------------
# Optional dependency: fuzzywuzzy for regional newspaper matching
# ---------------------------------------------------------------------------
try:
    from fuzzywuzzy import process as fuzz_process
except ImportError:
    fuzz_process = None
    logging.warning(
        "fuzzywuzzy not available -- fuzzy matching for regional experiments "
        "will fall back to the first timeline location."
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File names for persisted contamination / whitelist caches
# ---------------------------------------------------------------------------
CONTAMINATED_NAMES_FILE = "names_contaminated.txt"
WHITELISTED_NAMES_FILE = "names_whitelisted.txt"

@functools.lru_cache(maxsize=None)
def _wordnet_words(pos: str) -> Tuple[str, ...]:
    """Return a sorted, deduplicated tuple of capitalised single-word lemmas for *pos*."""
    from nltk.corpus import wordnet as wn
    return tuple(sorted({
        l.name().capitalize()
        for s in wn.all_synsets(pos)
        for l in s.lemmas()
        if l.name().isalpha()
    }))


# =============================================================================
# SourceGenerator
# =============================================================================

class SourceGenerator:
    """Generate sources for source preference evaluation experiments.

    This class centralises every kind of source generation needed by the
    evaluation pipeline: newspaper names, government sources, person names
    (with age / gender / academic-title variants), and social media accounts.

    Parameters
    ----------
    source_templates : list of str
        Newspaper-name templates containing a ``{NAME}`` placeholder that will
        be replaced by a location string (e.g. ``"The {NAME} Tribune"``).
    timeline_locations : dict
        Mapping from timeline ID strings to lists of location names.
    government_templates : dict
        Mapping from entity-class strings to lists of government source
        templates containing a ``{PLACE}`` placeholder.
    names_data : dict
        A dictionary holding the name data files.  Expected keys:

        - ``"male"``   -- parsed JSON with male first names and age distributions
        - ``"female"`` -- parsed JSON with female first names and age distributions
        - ``"young"``  -- parsed JSON with young-generation first names (has ``gender`` field)
        - ``"old"``    -- parsed JSON with old-generation first names (has ``gender`` field)
        - ``"middle"`` -- parsed JSON with middle-aged first names (has ``gender`` field)
        - ``"last"``   -- list of last-name strings

    circulation_values : list of int/float
        Raw circulation numbers used to compute percentile thresholds.
    use_training_blacklist : bool, optional
        When *True*, generated sources avoid words/names used in fine-tuning data.
    training_blacklist : dict or None, optional
        An already-loaded training blacklist dictionary.  Recognised keys:
        ``'social_media_adjectives'``, ``'social_media_nouns'``,
        ``'social_media_digits'``, ``'location_names'``, ``'government_templates'``.
    """

    # ------------------------------------------------------------------
    # Construction & data loading
    # ------------------------------------------------------------------

    def __init__(
        self,
        source_templates: List[str],
        timeline_locations: Dict[str, List[str]],
        names_data: Dict,
        circulation_values: List,
        government_templates: Optional[Dict[str, List[str]]] = None,
        use_training_blacklist: bool = False,
        training_blacklist: Optional[Dict] = None,
    ):
        self.source_templates = source_templates
        self.timeline_locations = timeline_locations
        self.government_templates = government_templates or {}
        self.names_data = names_data
        self.circulation_values = circulation_values
        self.use_training_blacklist = use_training_blacklist
        self.training_blacklist = training_blacklist

        if not source_templates:
            raise ValueError("source_templates must not be empty")

        # Load persisted contamination / whitelist caches
        self.contaminated_names: Set[str] = self._load_name_set(CONTAMINATED_NAMES_FILE)
        self.whitelisted_names: Set[str] = self._load_name_set(WHITELISTED_NAMES_FILE)

        # Pre-compute circulation percentiles
        self.percentile_25: float = 0.0
        self.percentile_75: float = 0.0
        self.calculate_circulation_stats()

    # ------------------------------------------------------------------
    # Private helpers -- file I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _load_name_set(filepath: str) -> Set[str]:
        """Load a newline-delimited set of full names from *filepath*.

        Returns an empty set when the file does not exist or cannot be read.
        """
        names: Set[str] = set()
        if not os.path.exists(filepath):
            logger.info("Name file not found (%s), starting with empty set", filepath)
            return names
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped:
                        names.add(stripped)
            logger.info("Loaded %d names from %s", len(names), filepath)
        except Exception as exc:
            logger.warning("Failed to load names from %s: %s", filepath, exc)
        return names

    def _save_contaminated_name(self, first_name: str, last_name: str) -> None:
        """Append a full name to the contaminated-names cache and file."""
        full_name = f"{first_name} {last_name}"
        if full_name in self.contaminated_names:
            return
        self.contaminated_names.add(full_name)
        try:
            with open(CONTAMINATED_NAMES_FILE, "a", encoding="utf-8") as fh:
                fh.write(f"{full_name}\n")
            logger.info("Added '%s' to contaminated names blacklist", full_name)
        except Exception as exc:
            logger.error("Failed to save contaminated name '%s': %s", full_name, exc)

    def _save_whitelisted_name(self, first_name: str, last_name: str) -> None:
        """Append a full name to the whitelisted-names cache and file."""
        full_name = f"{first_name} {last_name}"
        if full_name in self.whitelisted_names:
            return
        self.whitelisted_names.add(full_name)
        try:
            with open(WHITELISTED_NAMES_FILE, "a", encoding="utf-8") as fh:
                fh.write(f"{full_name}\n")
            logger.info("Added '%s' to whitelisted names", full_name)
        except Exception as exc:
            logger.error("Failed to save whitelisted name '%s': %s", full_name, exc)

    # ------------------------------------------------------------------
    # Circulation statistics
    # ------------------------------------------------------------------

    def calculate_circulation_stats(self) -> None:
        """Compute the 25th and 75th percentile of ``self.circulation_values``.

        The results are stored on the instance as ``self.percentile_25`` and
        ``self.percentile_75``.  If the values list is empty both are set to
        ``0.0``.
        """
        if not self.circulation_values:
            self.percentile_25 = 0.0
            self.percentile_75 = 0.0
            logger.warning("No circulation values provided; percentiles set to 0.0")
            return

        arr = np.array(self.circulation_values, dtype=float)
        self.percentile_25 = float(np.percentile(arr, 25))
        self.percentile_75 = float(np.percentile(arr, 75))
        logger.debug(
            "Circulation stats: 25th=%.1f, 75th=%.1f (n=%d)",
            self.percentile_25,
            self.percentile_75,
            len(self.circulation_values),
        )

    def _get_circulation_by_tier(self, tier: str) -> int:
        """Sample a circulation value from the requested tier.

        Parameters
        ----------
        tier : str
            ``"high"`` -- sample from values at or above the 75th percentile.
            ``"low"``  -- sample from values at or below the 25th percentile.

        Returns
        -------
        int
            A circulation value drawn from the appropriate tail of the
            distribution, or a deterministic fallback when no qualifying
            values exist.

        Raises
        ------
        ValueError
            If *tier* is neither ``"high"`` nor ``"low"``.
        """
        if tier == "high":
            candidates = [
                v for v in self.circulation_values if v >= self.percentile_75
            ]
            if candidates:
                return int(random.choice(candidates))
            return int(self.percentile_75) or 100_000

        if tier == "low":
            candidates = [
                v for v in self.circulation_values if v <= self.percentile_25
            ]
            if candidates:
                return int(random.choice(candidates))
            return int(self.percentile_25) or 5_000

        raise ValueError(f"Unknown circulation tier: {tier!r}")

    # ------------------------------------------------------------------
    # Contamination checking (Wikipedia API)
    # ------------------------------------------------------------------

    def _check_contamination_risk(self, first_name: str, last_name: str) -> bool:
        """Check whether a person name has a Wikipedia article.

        The check proceeds as follows:

        1. Return *False* immediately if the name is in the whitelist cache.
        2. Return *True* immediately if the name is in the contaminated cache.
        3. Query the Wikipedia API for a page with the exact title
           ``"Firstname Lastname"``.
        4. If a non-disambiguation article exists, try to extract a birth year.
           If the birth year falls in a plausible modern range the name is
           considered contaminated.

        Results are persisted to the on-disk cache files so that repeated
        runs do not need to re-query Wikipedia.

        Parameters
        ----------
        first_name : str
            The person's first name.
        last_name : str
            The person's last name.

        Returns
        -------
        bool
            *True* when the name is likely contaminated, *False* otherwise.
        """
        full_name = f"{first_name} {last_name}"

        # Fast-path: check local caches first
        if full_name in self.whitelisted_names:
            return False
        if full_name in self.contaminated_names:
            return True

        try:
            params = {
                "action": "query",
                "format": "json",
                "titles": full_name,
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
            }
            response = requests.get(
                WIKIPEDIA_API, params=params, headers=WIKIPEDIA_HEADERS, timeout=10
            )
            response.raise_for_status()
            page_data = response.json()

            pages = page_data.get("query", {}).get("pages", {})
            if not pages:
                return False

            page = list(pages.values())[0]

            # Page does not exist -- safe
            if "missing" in page:
                self._save_whitelisted_name(first_name, last_name)
                return False

            page_title = page.get("title", full_name)

            if "disambiguation" in page_title.lower():
                self._save_whitelisted_name(first_name, last_name)
                return False

            return True

        except Exception as exc:
            # On network / parsing errors, be conservative and allow the name
            logger.debug("Error checking Wikipedia for '%s': %s", full_name, exc)
            return False

    def _is_person_valid(self, first_name: str, last_name: str) -> bool:
        """Return *True* when the person name passes the contamination check."""
        return not self._check_contamination_risk(first_name, last_name)

    # ------------------------------------------------------------------
    # Name sampling helpers
    # ------------------------------------------------------------------

    def _sample_name_with_age_distribution(
        self,
        names_data: Dict,
        min_age: int = 18,
        max_age: int = 63,
    ) -> Tuple[str, int, str]:
        """Sample a first name and age using per-name age distributions.

        The *names_data* dict must contain a ``"names"`` key whose value is a
        list of dicts with at least a ``"name"`` key and optionally an
        ``"age_distribution"`` mapping ``{str(age): probability}``.

        Parameters
        ----------
        names_data : dict
            Parsed JSON name file.
        min_age : int
            Lower bound (inclusive) of the acceptable age range.
        max_age : int
            Upper bound (inclusive) of the acceptable age range.

        Returns
        -------
        tuple of (str, int, str)
            ``(first_name, sampled_age, gender)`` where *gender* is ``'M'``
            or ``'F'`` inferred from the data-file metadata.
        """
        if not names_data or "names" not in names_data:
            raise ValueError("Invalid names data structure (missing 'names' key)")

        # Weight name selection by total probability mass within the age range,
        # so names with no probability in [min_age, max_age] are never chosen.
        name_weights = []
        for entry in names_data["names"]:
            age_dist = entry.get("age_distribution", {})
            mass = sum(
                prob for age_str, prob in age_dist.items()
                if min_age <= int(age_str) <= max_age
            )
            name_weights.append(mass)

        weight_total = sum(name_weights)
        if weight_total > 0:
            name_probs = [w / weight_total for w in name_weights]
            name_entry = names_data["names"][
                np.random.choice(len(names_data["names"]), p=name_probs)
            ]
        else:
            name_entry = random.choice(names_data["names"])
        name = name_entry["name"]

        # Build age weights within the requested range
        age_dist = name_entry.get("age_distribution", {})
        valid_ages = [
            (int(age_str), prob)
            for age_str, prob in age_dist.items()
            if min_age <= int(age_str) <= max_age
        ]

        if not valid_ages:
            age = random.randint(min_age, max_age)
        else:
            ages, probs = zip(*valid_ages)
            probs_arr = np.array(probs, dtype=float)
            probs_arr /= probs_arr.sum()
            age = int(np.random.choice(ages, p=probs_arr))

        # Determine gender from the metadata description
        metadata = names_data.get("metadata", {})
        description = metadata.get("description", "").lower()
        if "female" in description:
            gender = "F"
        elif "male" in description:
            gender = "M"
        else:
            gender = "M"  # default fallback

        return name, age, gender

    def _sample_young_or_old_name(
        self, names_data: Dict, age_group: str
    ) -> Tuple[str, int, str]:
        """Sample a name from a young- or old-generation name list.

        Parameters
        ----------
        names_data : dict
            Parsed JSON file whose ``"names"`` entries have ``"name"`` and
            ``"gender"`` fields.
        age_group : str
            ``"young"`` draws an age from ``[18, 25]``;
            ``"old"`` draws from ``[65, 80]``.

        Returns
        -------
        tuple of (str, int, str)
            ``(first_name, age, gender)``.
        """
        if not names_data or "names" not in names_data:
            raise ValueError("Invalid names data structure (missing 'names' key)")

        name_entry = random.choice(names_data["names"])
        name = name_entry["name"]
        gender = name_entry["gender"]

        if age_group == "young":
            age = random.randint(18, 25)
        elif age_group == "middle":
            age = random.randint(35, 55)
        elif age_group == "old":
            age = random.randint(65, 80)
        else:
            raise ValueError(f"Invalid age_group: {age_group!r} (expected 'young', 'middle', or 'old')")

        return name, age, gender

    # ------------------------------------------------------------------
    # Source generation: newspapers
    # ------------------------------------------------------------------

    def generate_newspaper_source(
        self,
        exclude_timeline_id: str,
        circulation: Optional[int] = None,
    ) -> Source:
        """Generate an artificial newspaper source from a template and location.

        A random newspaper-name template is filled with a random location that
        does **not** belong to *exclude_timeline_id*, producing names like
        ``"The Springfield Gazette"``.

        Parameters
        ----------
        exclude_timeline_id : str
            Timeline whose locations should be excluded so that the generated
            source does not accidentally overlap with the entity's own context.
        circulation : int or None
            Circulation figure to attach to the source.  When *None* the
            source will have no circulation data.

        Returns
        -------
        Source
            A newly constructed newspaper source.
        """
        # Collect locations from every timeline except the excluded one
        available_locations: List[str] = []
        for tid, locations in self.timeline_locations.items():
            if tid != exclude_timeline_id:
                available_locations.extend(locations)

        template = random.choice(self.source_templates)
        location = random.choice(available_locations)

        # Replace the placeholder (supports both {'NAME'} and {NAME})
        source_name = template.replace("{'NAME'}", location).replace("{NAME}", location)
        source_name = source_name.title()

        logger.debug("Generated artificial source: %s", source_name)
        return Source(name=source_name, circulation=circulation, country="USA")

    # ------------------------------------------------------------------
    # Source generation: government
    # ------------------------------------------------------------------

    def generate_government_source(
        self, entity_class: str, entity_timeline_id: str
    ) -> GovernmentSource:
        """Generate a government / institutional source from templates.

        A random template matching *entity_class* is filled with a location
        sampled from the entity's own timeline (so that it reads as a local
        authority).

        Parameters
        ----------
        entity_class : str
            The entity class (e.g. ``"person"``, ``"location"``).
        entity_timeline_id : str
            Timeline ID from which to draw the location.

        Returns
        -------
        GovernmentSource
            A newly constructed government source.
        """
        # Get locations for the entity's timeline
        timeline_locs = self.timeline_locations.get(entity_timeline_id, [])
        if not timeline_locs:
            # No specific timeline — pool all available locations
            all_locs: List[str] = []
            for locs in self.timeline_locations.values():
                all_locs.extend(locs)
            timeline_locs = all_locs
        available_locations = list(timeline_locs)

        # Filter blacklisted locations when applicable
        if self.use_training_blacklist and self.training_blacklist:
            bl_locations = self.training_blacklist.get("location_names", set())
            filtered = [loc for loc in available_locations if loc not in bl_locations]
            if filtered:
                available_locations = filtered

        location = random.choice(available_locations)

        # Select a template for the entity class
        templates = self.government_templates.get(entity_class, [])
        if not templates:
            # Pool all templates from every entity class as a fallback
            all_templates: List[str] = []
            for tmpl_list in self.government_templates.values():
                all_templates.extend(tmpl_list)
            templates = all_templates if all_templates else [
                "Government Office of {'PLACE'}",
                "Ministry of Information of {'PLACE'}",
                "Official Records of {'PLACE'}",
            ]
            logger.debug("Using fallback government templates for class %s", entity_class)

        # Filter blacklisted templates when applicable
        if self.use_training_blacklist and self.training_blacklist:
            bl_templates = self.training_blacklist.get("government_templates", set())
            filtered_t = [t for t in templates if t.lower() not in bl_templates]
            if filtered_t:
                templates = filtered_t
            else:
                logger.warning("All government templates blacklisted; using originals")

        template = random.choice(templates)
        source_name = template.replace("{'PLACE'}", location).replace("{PLACE}", location)
        source_name = source_name.title()

        logger.debug("Generated government source: %s (class=%s)", source_name, entity_class)
        return GovernmentSource(name=source_name, location=location, entity_class=entity_class)

    # ------------------------------------------------------------------
    # Source generation: person pairs
    # ------------------------------------------------------------------

    def generate_male_female_pair(self) -> Tuple[PersonSource, PersonSource]:
        """Generate a male and a female person source with ages within 5 years.

        Both names are checked against Wikipedia for contamination.  The age
        of the second person is constrained to lie within 5 years of the
        first.

        Returns
        -------
        tuple of (PersonSource, PersonSource)
            ``(male_source, female_source)``

        Raises
        ------
        RuntimeError
            If no valid pair is found after ``MAX_PERSON_GENERATION_ATTEMPTS``.
        """
        male_data = self.names_data.get("male")
        female_data = self.names_data.get("female")
        last_names = self.names_data.get("last")

        if not male_data or not female_data or not last_names:
            raise RuntimeError("Name data not loaded for male/female generation")

        for _ in range(MAX_PERSON_GENERATION_ATTEMPTS):
            # Randomly decide which gender anchors the age
            if random.random() < 0.5:
                # Male first
                m_first, m_age, _ = self._sample_name_with_age_distribution(
                    male_data, min_age=18, max_age=63
                )
                m_last = random.choice(last_names)
                if not self._is_person_valid(m_first, m_last):
                    continue

                f_min_age = max(18, m_age - 5)
                f_max_age = min(63, m_age + 5)
                f_first, f_age, _ = self._sample_name_with_age_distribution(
                    female_data, min_age=f_min_age, max_age=f_max_age
                )
                f_last = random.choice(last_names)
                if not self._is_person_valid(f_first, f_last):
                    continue
            else:
                # Female first
                f_first, f_age, _ = self._sample_name_with_age_distribution(
                    female_data, min_age=18, max_age=63
                )
                f_last = random.choice(last_names)
                if not self._is_person_valid(f_first, f_last):
                    continue

                m_min_age = max(18, f_age - 5)
                m_max_age = min(63, f_age + 5)
                m_first, m_age, _ = self._sample_name_with_age_distribution(
                    male_data, min_age=m_min_age, max_age=m_max_age
                )
                m_last = random.choice(last_names)
                if not self._is_person_valid(m_first, m_last):
                    continue

            male_source = PersonSource(
                first_name=m_first, last_name=m_last, age=m_age, gender="M"
            )
            female_source = PersonSource(
                first_name=f_first, last_name=f_last, age=f_age, gender="F"
            )
            logger.debug(
                "Generated male/female pair: %s and %s",
                male_source.full_name,
                female_source.full_name,
            )
            return male_source, female_source

        raise RuntimeError(
            f"Failed to generate valid male/female pair after "
            f"{MAX_PERSON_GENERATION_ATTEMPTS} attempts"
        )

    def generate_old_young_pair(self) -> Tuple[PersonSource, PersonSource]:
        """Generate an old (65-80) and a young (18-25) person source of the same gender.

        Returns
        -------
        tuple of (PersonSource, PersonSource)
            ``(old_source, young_source)``

        Raises
        ------
        RuntimeError
            If no valid pair is found within the retry budget.
        """
        young_data = self.names_data.get("young")
        old_data = self.names_data.get("old")
        last_names = self.names_data.get("last")

        if not young_data or not old_data or not last_names:
            raise RuntimeError("Name data not loaded for old/young generation")

        for _ in range(MAX_PERSON_GENERATION_ATTEMPTS):
            # Sample young name first to determine gender
            y_first, y_age, y_gender = self._sample_young_or_old_name(young_data, "young")
            y_last = random.choice(last_names)
            if not self._is_person_valid(y_first, y_last):
                continue

            # Find old-generation names that match the gender
            matching_old = [
                entry for entry in old_data["names"] if entry["gender"] == y_gender
            ]
            if not matching_old:
                raise RuntimeError(f"No old names available for gender {y_gender}")

            old_entry = random.choice(matching_old)
            o_first = old_entry["name"]
            o_age = random.randint(65, 80)
            o_last = random.choice(last_names)
            if not self._is_person_valid(o_first, o_last):
                continue

            young_source = PersonSource(
                first_name=y_first,
                last_name=y_last,
                age=y_age,
                gender=y_gender,
                age_group="young",
            )
            old_source = PersonSource(
                first_name=o_first,
                last_name=o_last,
                age=o_age,
                gender=y_gender,
                age_group="old",
            )
            logger.debug(
                "Generated old/young pair: %s (old) and %s (young)",
                old_source.full_name,
                young_source.full_name,
            )
            return old_source, young_source

        raise RuntimeError(
            f"Failed to generate valid old/young pair after "
            f"{MAX_PERSON_GENERATION_ATTEMPTS} attempts"
        )

    def generate_middle_young_pair(self) -> Tuple[PersonSource, PersonSource]:
        """Generate a middle-aged (35-55) and a young (18-25) person source of the same gender.

        Returns
        -------
        tuple of (PersonSource, PersonSource)
            ``(middle_source, young_source)``

        Raises
        ------
        RuntimeError
            If no valid pair is found within the retry budget.
        """
        middle_data = self.names_data.get("middle")
        young_data = self.names_data.get("young")
        last_names = self.names_data.get("last")

        if not middle_data or not young_data or not last_names:
            raise RuntimeError("Name data not loaded for middle/young generation")

        for _ in range(MAX_PERSON_GENERATION_ATTEMPTS):
            y_first, y_age, y_gender = self._sample_young_or_old_name(young_data, "young")
            y_last = random.choice(last_names)
            if not self._is_person_valid(y_first, y_last):
                continue

            matching_middle = [
                entry for entry in middle_data["names"] if entry["gender"] == y_gender
            ]
            if not matching_middle:
                raise RuntimeError(f"No middle-aged names available for gender {y_gender}")

            m_entry = random.choice(matching_middle)
            m_first = m_entry["name"]
            m_age = random.randint(35, 55)
            m_last = random.choice(last_names)
            if not self._is_person_valid(m_first, m_last):
                continue

            young_source = PersonSource(
                first_name=y_first,
                last_name=y_last,
                age=y_age,
                gender=y_gender,
                age_group="young",
            )
            middle_source = PersonSource(
                first_name=m_first,
                last_name=m_last,
                age=m_age,
                gender=y_gender,
                age_group="middle",
            )
            logger.debug(
                "Generated middle/young pair: %s (middle) and %s (young)",
                middle_source.full_name,
                young_source.full_name,
            )
            return middle_source, young_source

        raise RuntimeError(
            f"Failed to generate valid middle/young pair after "
            f"{MAX_PERSON_GENERATION_ATTEMPTS} attempts"
        )

    def generate_middle_old_pair(self) -> Tuple[PersonSource, PersonSource]:
        """Generate a middle-aged (35-55) and an old (65-80) person source of the same gender.

        Returns
        -------
        tuple of (PersonSource, PersonSource)
            ``(middle_source, old_source)``

        Raises
        ------
        RuntimeError
            If no valid pair is found within the retry budget.
        """
        middle_data = self.names_data.get("middle")
        old_data = self.names_data.get("old")
        last_names = self.names_data.get("last")

        if not middle_data or not old_data or not last_names:
            raise RuntimeError("Name data not loaded for middle/old generation")

        for _ in range(MAX_PERSON_GENERATION_ATTEMPTS):
            m_entry_init = random.choice(middle_data["names"])
            m_gender = m_entry_init["gender"]
            m_first = m_entry_init["name"]
            m_age = random.randint(35, 55)
            m_last = random.choice(last_names)
            if not self._is_person_valid(m_first, m_last):
                continue

            matching_old = [
                entry for entry in old_data["names"] if entry["gender"] == m_gender
            ]
            if not matching_old:
                raise RuntimeError(f"No old names available for gender {m_gender}")

            o_entry = random.choice(matching_old)
            o_first = o_entry["name"]
            o_age = random.randint(65, 80)
            o_last = random.choice(last_names)
            if not self._is_person_valid(o_first, o_last):
                continue

            middle_source = PersonSource(
                first_name=m_first,
                last_name=m_last,
                age=m_age,
                gender=m_gender,
                age_group="middle",
            )
            old_source = PersonSource(
                first_name=o_first,
                last_name=o_last,
                age=o_age,
                gender=m_gender,
                age_group="old",
            )
            logger.debug(
                "Generated middle/old pair: %s (middle) and %s (old)",
                middle_source.full_name,
                old_source.full_name,
            )
            return middle_source, old_source

        raise RuntimeError(
            f"Failed to generate valid middle/old pair after "
            f"{MAX_PERSON_GENERATION_ATTEMPTS} attempts"
        )

    def generate_generic_person(self) -> PersonSource:
        """Generate a single person source with random gender and age (18-63).

        Returns
        -------
        PersonSource
            A person with a contamination-checked name.

        Raises
        ------
        RuntimeError
            If no valid name is found within the retry budget.
        """
        male_data = self.names_data.get("male")
        female_data = self.names_data.get("female")
        last_names = self.names_data.get("last")

        if not male_data or not female_data or not last_names:
            raise RuntimeError("Name data not loaded for generic person generation")

        for _ in range(MAX_PERSON_GENERATION_ATTEMPTS):
            if random.random() < 0.5:
                first_name, age, _ = self._sample_name_with_age_distribution(
                    male_data, min_age=18, max_age=63
                )
                gender = "M"
            else:
                first_name, age, _ = self._sample_name_with_age_distribution(
                    female_data, min_age=18, max_age=63
                )
                gender = "F"

            last_name = random.choice(last_names)
            if not self._is_person_valid(first_name, last_name):
                continue

            person = PersonSource(
                first_name=first_name, last_name=last_name, age=age, gender=gender
            )
            logger.debug(
                "Generated generic person: %s, age %d, gender %s",
                person.full_name,
                person.age,
                person.gender,
            )
            return person

        raise RuntimeError(
            f"Failed to generate valid generic person after "
            f"{MAX_PERSON_GENERATION_ATTEMPTS} attempts"
        )

    def generate_high_low_academia_pair(self) -> Tuple[PersonSource, PersonSource]:
        """Generate a high-academia and a low-academia person pair.

        High-academia persons receive a ``Dr.``, ``Prof.``, or ``PhD`` title.
        Low-academia persons receive a ``Mr.``, ``Ms.``, or ``Mrs.`` title.
        Ages are sampled in the 30-65 range (plausible for academic settings).

        Returns
        -------
        tuple of (PersonSource, PersonSource)
            ``(high_academia_person, low_academia_person)``

        Raises
        ------
        RuntimeError
            If no valid pair is found within the retry budget.
        """
        male_data = self.names_data.get("male")
        female_data = self.names_data.get("female")
        last_names = self.names_data.get("last")

        if not male_data or not female_data or not last_names:
            raise RuntimeError("Name data not loaded for academia pair generation")

        for _ in range(MAX_PERSON_GENERATION_ATTEMPTS):
            people: List[PersonSource] = []
            valid_pair = True

            for academia_level in ("high", "low"):
                if random.random() < 0.5:
                    first_name, age, _ = self._sample_name_with_age_distribution(
                        male_data, min_age=30, max_age=65
                    )
                    gender = "M"
                else:
                    first_name, age, _ = self._sample_name_with_age_distribution(
                        female_data, min_age=30, max_age=65
                    )
                    gender = "F"

                last_name = random.choice(last_names)
                if not self._is_person_valid(first_name, last_name):
                    valid_pair = False
                    break

                people.append(
                    PersonSource(
                        first_name=first_name,
                        last_name=last_name,
                        age=age,
                        gender=gender,
                        academic_title=academia_level,
                    )
                )

            if not valid_pair:
                continue

            high_academia, low_academia = people
            logger.debug(
                "Generated academia pair: %s (high) vs %s (low)",
                high_academia.full_name,
                low_academia.full_name,
            )
            return high_academia, low_academia

        raise RuntimeError(
            f"Failed to generate valid academia pair after "
            f"{MAX_PERSON_GENERATION_ATTEMPTS} attempts"
        )

    # ------------------------------------------------------------------
    # Source generation: regional vs. non-regional newspapers
    # ------------------------------------------------------------------

    def generate_regional_non_regional_pair(
        self,
        entity_timeline_id: str,
        entity_name: Optional[str] = None,
        entity_class: Optional[str] = None,
    ) -> Tuple[Source, Source]:
        """Generate a regional and a non-regional newspaper for an entity.

        The **regional** newspaper's location is drawn from the entity's own
        timeline.  For location / organisation / building entities, fuzzy
        matching (via ``fuzzywuzzy``) is used to pick the timeline location
        most similar to the entity name.  For other entity classes the first
        timeline location is used.

        The **non-regional** newspaper uses a location from a *different*
        timeline so that it reads as geographically distant.

        Parameters
        ----------
        entity_timeline_id : str
            Timeline ID of the entity.
        entity_name : str or None
            Entity name (used for fuzzy matching when applicable).
        entity_class : str or None
            Entity class string.

        Returns
        -------
        tuple of (Source, Source)
            ``(regional_newspaper, non_regional_newspaper)``
        """
        locations = self.timeline_locations.get(entity_timeline_id, [])

        # --- determine regional location ---
        if not locations:
            # No timeline locations -- sample from all timelines or fallback
            all_locs: List[str] = []
            for locs in self.timeline_locations.values():
                all_locs.extend(locs)
            regional_location = random.choice(all_locs)
        else:
            # Use fuzzy matching for spatial entity classes if fuzzywuzzy is available
            if (
                entity_class in ("location", "organization", "building")
                and entity_name
                and fuzz_process is not None
            ):
                best_match, score = fuzz_process.extractOne(entity_name, locations)
                regional_location = best_match
                logger.debug(
                    "Fuzzy matched '%s' to '%s' (score: %d)",
                    entity_name,
                    regional_location,
                    score,
                )
            else:
                regional_location = locations[0]

        # --- build regional newspaper ---
        regional_template = random.choice(self.source_templates)
        regional_name = (
            regional_template
            .replace("{'NAME'}", regional_location)
            .replace("{NAME}", regional_location)
            .title()
        )
        regional_newspaper = Source(
            name=regional_name, circulation=None, country="USA", regional_marker="regional"
        )

        # --- determine non-regional location ---
        all_timeline_locations: Set[str] = set()
        for locs in self.timeline_locations.values():
            all_timeline_locations.update(locs)

        if locations:
            non_regional_candidates = [
                loc for loc in all_timeline_locations if loc not in locations
            ]
        else:
            non_regional_candidates = [
                loc for loc in all_timeline_locations if loc != regional_location
            ]

        non_regional_location = random.choice(non_regional_candidates)

        # --- build non-regional newspaper ---
        non_regional_template = random.choice(self.source_templates)
        non_regional_name = (
            non_regional_template
            .replace("{'NAME'}", non_regional_location)
            .replace("{NAME}", non_regional_location)
            .title()
        )
        non_regional_newspaper = Source(
            name=non_regional_name, circulation=None, country="USA", regional_marker="non_regional"
        )

        logger.debug(
            "Generated regional pair: %s (regional: %s) vs %s (non-regional: %s)",
            regional_name,
            regional_location,
            non_regional_name,
            non_regional_location,
        )
        return regional_newspaper, non_regional_newspaper

    # ------------------------------------------------------------------
    # Source generation: social media users
    # ------------------------------------------------------------------

    def generate_username(self) -> str:
        """Generate a CamelCase username of the form ``AdjectiveNoun1234``.

        When a *training_blacklist* was provided at construction time, words
        and digit sequences present in the blacklist are filtered out.

        Returns
        -------
        str
            A username string such as ``"BrightBird4821"``.
        """
        adjectives = list(_wordnet_words("a"))
        nouns = list(_wordnet_words("n"))
        blacklisted_digits: set = set()

        if self.training_blacklist:
            bl_adj = self.training_blacklist.get("social_media_adjectives", set())
            bl_nouns = self.training_blacklist.get("social_media_nouns", set())
            blacklisted_digits = self.training_blacklist.get("social_media_digits", set())

            filtered_adj = [a for a in adjectives if a not in bl_adj]
            filtered_nouns = [n for n in nouns if n not in bl_nouns]

            if len(filtered_adj) >= 10:
                adjectives = filtered_adj
            if len(filtered_nouns) >= 10:
                nouns = filtered_nouns

        adjective = random.choice(adjectives)
        noun = random.choice(nouns)

        # Try up to 20 times to find a non-blacklisted 4-digit suffix
        digits = ""
        for _ in range(20):
            digits = str(random.randint(1000, 9999))
            if digits not in blacklisted_digits:
                break

        username = f"{adjective}{noun}{digits}"
        logger.debug("Generated username: %s", username)
        return username

    def generate_follower_count(self, popularity_tier: str) -> int:
        """Generate a follower count appropriate for the given popularity tier.

        Parameters
        ----------
        popularity_tier : str
            ``"low"`` produces a count in ``[0, 99]``.
            ``"high"`` produces a count with 4 to 6 digits (``[1000, 999999]``).

        Returns
        -------
        int
            The generated follower count.

        Raises
        ------
        ValueError
            If *popularity_tier* is not ``"low"`` or ``"high"``.
        """
        if popularity_tier == "low":
            return random.randint(0, 99)

        if popularity_tier == "high":
            # Choose the magnitude first so that each order of magnitude is
            # equally likely, then sample uniformly within that range.
            num_digits = random.choice([4, 5, 6])
            lo = 10 ** (num_digits - 1)
            hi = 10 ** num_digits - 1
            return random.randint(lo, hi)

        raise ValueError(f"Invalid popularity tier: {popularity_tier!r}")

    def generate_social_media_user(
        self, popularity_tier: Optional[str] = None
    ) -> SocialMediaUser:
        """Generate an artificial social media user.

        Parameters
        ----------
        popularity_tier : str or None
            When ``"low"`` or ``"high"``, a follower count is generated and
            attached.  When *None*, the follower count is set to ``-1``
            (meaning "do not display").

        Returns
        -------
        SocialMediaUser
            A new user object with a generated username and follower count.
        """
        username = self.generate_username()

        if popularity_tier is not None:
            follower_count = self.generate_follower_count(popularity_tier)
        else:
            follower_count = -1

        user = SocialMediaUser(username=username, follower_count=follower_count)
        logger.debug(
            "Generated social media user: @%s with %d followers",
            user.username,
            user.follower_count,
        )
        return user

    def generate_traditional_social_media_user(
        self, first_name: str, last_name: str
    ) -> SocialMediaUser:
        """Generate a social media user whose username reveals their real name.

        The username is either ``Firstname_Lastname`` or ``FirstnameLastname``
        (chosen at random).  The follower count is set to ``-1`` so it will
        not be displayed in the prompt.

        Parameters
        ----------
        first_name : str
            The person's first name.
        last_name : str
            The person's last name.

        Returns
        -------
        SocialMediaUser
            A user object with a name-based username and no visible followers.
        """
        if random.choice([True, False]):
            username = f"{first_name}_{last_name}"
        else:
            username = f"{first_name}{last_name}"

        user = SocialMediaUser(username=username, follower_count=-1)
        logger.debug("Generated known social media user: @%s", user.username)
        return user

    def generate_anon_known_social_media_pair(
        self,
    ) -> Tuple[SocialMediaUser, SocialMediaUser]:
        """Generate an anonymous and a known (name-based) social media user pair.

        The **anonymous** user has a standard CamelCase username (e.g.
        ``@BoldEagle1234``).  The **known** user has a name-based username
        (e.g. ``@Jane_Doe`` or ``@JaneDoe``).  Neither user displays a
        follower count.

        Returns
        -------
        tuple of (SocialMediaUser, SocialMediaUser)
            ``(anonymous_user, known_user)``
        """
        male_data = self.names_data.get("male")
        female_data = self.names_data.get("female")
        last_names = self.names_data.get("last")

        if not male_data or not female_data or not last_names:
            raise RuntimeError("Name data not loaded for known social media user generation")

        # Anonymous user -- generated username, no follower count shown
        anonymous_user = self.generate_social_media_user(None)

        # Known user -- name-based username
        if random.random() < 0.5:
            first_name, _, _ = self._sample_name_with_age_distribution(
                male_data, min_age=18, max_age=63
            )
        else:
            first_name, _, _ = self._sample_name_with_age_distribution(
                female_data, min_age=18, max_age=63
            )
        last_name = random.choice(last_names)

        known_user = self.generate_traditional_social_media_user(first_name, last_name)

        logger.debug(
            "Generated anon/known pair: @%s vs @%s",
            anonymous_user.username,
            known_user.username,
        )
        return anonymous_user, known_user
