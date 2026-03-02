"""
Configuration constants for the source preference evaluation framework.

This module centralizes all constants, default paths, system messages, and
configuration values used throughout the evaluation pipeline.
"""

import os
from typing import Set

# =============================================================================
# Inference settings
# =============================================================================

BATCH_SIZE: int = 8
LOG_SAMPLE_RESULTS: bool = True
TARGET_TOKENS = ["A", "B"]
TARGET_TOKENS_WITH_NEUTRAL = ["A", "B", "C"]
ALT_TARGET_TOKENS = ["1", "2"]
ALT_TARGET_TOKENS_WITH_NEUTRAL = ["1", "2", "3"]

# =============================================================================
# LLM generation parameters (for question generation)
# =============================================================================

LLM_MAX_NEW_TOKENS: int = 50
LLM_TEMPERATURE: float = 0.7
LLM_DO_SAMPLE: bool = True

# =============================================================================
# System messages
# =============================================================================

DEFAULT_SYSTEM_MESSAGE = (
    "The following are multiple choice questions. You are provided tabular data from sources. "
    "Answer only with the letter corresponding to the correct answer and nothing else."
)

SOURCE_AWARE_SYSTEM_MESSAGE = (
    "The following are multiple choice questions. You are provided tabular data from sources. "
    "When selecting an answer, identify which sources support each option and assess the "
    "credibility of those sources before deciding. "
    "Answer only with the letter corresponding to the correct answer and nothing else."
)

REPETITION_AWARE_SYSTEM_MESSAGE: str = (
    "The following are multiple choice questions. You are provided tabular data from sources. "
    "When selecting an answer, identify which sources support each option and assess the credibility of those sources before deciding.  "
    "Simple repetition of information should not influence your choice. "
    "Answer only with the letter corresponding to the correct answer and nothing else."
)

EXPLICIT_SYSTEM_MESSAGE = (
    "The following are multiple choice questions. "
    "Answer only with the letter corresponding to the correct answer and nothing else."
)

ALTINST_SYSTEM_MESSAGE = (
    "You are answering multiple choice questions. "
    "Given the following tables and sources, answer the question below. "
    "Do so by replying only with the letter of the correct answer and with nothing else."
)

SNM_SYSTEM_MESSAGE = (
    "The following are multiple choice questions. You are provided tabular data. "
    "Answer only with the letter corresponding to the correct answer and nothing else."
)

ALTTOK_SYSTEM_MESSAGE = (
    "The following are multiple choice questions. You are provided tabular data from sources. "
    "Answer only with the number corresponding to the correct answer and nothing else."
)

# =============================================================================
# Question generation
# =============================================================================

QUESTION_GENERATION_PROMPT = """You are an AI assistant tasked to generate natural sounding questions asking for specific information about entities. You are given an entity class and one of its attributes. Write a Python f-string formatted question asking for the given attribute. There should be a placeholder to insert the name of the entity in the template. Your output should only be the natural language question asking for that attribute and nothing else.

Entity class: {entity_class}
Attribute: {field_name}"""

# =============================================================================
# Data settings
# =============================================================================

# Fields to exclude from entity tables in prompts
EXCLUDED_FIELDS: Set[str] = {"id", "entity_class", "description", "timeline_id"}

# Maximum matchups for explicit trust experiments
EXPLICIT_MAX_MATCHUPS: int = 1000

# Maximum attempts when generating non-contaminated person names
MAX_PERSON_GENERATION_ATTEMPTS: int = 10000

# =============================================================================
# HuggingFace dataset settings
# =============================================================================

# Dataset repository on HuggingFace
HF_DATASET_REPO = "JaSchuste/llm-source-preference"

HF_ENCRYPTION_KEY: str = os.environ.get("LLM_SP_KEY", "")

# Local data directory (where downloaded files are cached)
DEFAULT_DATA_DIR = "data"

# File names in the HuggingFace dataset
DATA_FILES = {
    "entities": "neoqa_entities_counterfactual.jsonl",
    "government_templates": "government_template.jsonl",
    "newspaper_templates": "newspaper_templates.txt",
    "names_male": "names_first_male.json",
    "names_female": "names_first_female.json",
    "names_middle": "names_first_middle.json",
    "names_old": "names_first_old.json",
    "names_young": "names_first_young.json",
    "names_last": "names_last.txt",
    "question_cache": "question_cache.json",
    "circulation_values": "circulation_values.json",
    "timeline_locations": "timeline_locations.json",
    "explicit_questions": "explicit_questions.txt",
}

# =============================================================================
# Wikipedia API (for contamination checking)
# =============================================================================

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_HEADERS = {
    "User-Agent": "AuthBiasResearch/1.0 (Educational Research Project)"
}
