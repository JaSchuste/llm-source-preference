"""
Prompt building module for the source preference evaluation framework.

Constructs natural language questions and multi-table prompts for evaluating
how LLMs respond to information presented by sources of varying authority.
Supports two-table, three-table (2tm majority), implicit majority (1tm), and
explicit trust prompt formats with options for neutral answers, alternative
token schemes, and varied source display modes.
"""

import logging
from typing import List, Dict, Optional, Any, Union

import torch

from helpers.models import (
    Entity,
    Source,
    SocialMediaUser,
    GovernmentSource,
    PersonSource,
    SourceGroup,
    Conflict,
)
from helpers.config import (
    EXCLUDED_FIELDS,
    DEFAULT_SYSTEM_MESSAGE,
    QUESTION_GENERATION_PROMPT,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_DO_SAMPLE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helper utilities
# =============================================================================

def format_source_info(
    source: Union[Source, SocialMediaUser, GovernmentSource, PersonSource],
    show_circulation: bool = False,
    experiment_type: Optional[str] = None,
) -> str:
    """Format a source object for display in a prompt.

    The output varies by source type and experiment context:

    * **PersonSource** -- In demographic experiments (``experiment_type`` contains
      ``"male_female"`` or ``"old_young"``) the full name including age and gender
      is returned.  In academic experiments (``"academic"`` in ``experiment_type``)
      the academic-title variant is used.  Otherwise the plain first/last name is
      returned.
    * **SocialMediaUser** -- ``"@username (X followers)"`` when the follower count
      is non-negative, otherwise just ``"@username"``.
    * **GovernmentSource** -- The source name.
    * **Source** -- If *show_circulation* is ``True`` and the source has a
      circulation value, ``"name (circulation: X)"`` is returned; otherwise just
      the name.

    Args:
        source: The source to format.
        show_circulation: Whether to include circulation / follower counts.
        experiment_type: Optional string hint about the experiment (e.g.
            ``"male_female"``, ``"old_young"``, ``"academic"``).

    Returns:
        A human-readable string describing the source.
    """
    if isinstance(source, PersonSource):
        exp = (experiment_type or "").lower()
        # Demographic experiments: show full name with age/gender
        if any(kw in exp for kw in ("male_female", "old_young", "middle_vs")):
            return source.name  # e.g. "FirstName LastName (M), aged 35"
        # Academic experiments: show title variant
        if "academic" in exp:
            return source.name  # e.g. "Dr. FirstName LastName"
        # Generic person experiments: plain name without demographics
        return source.simple_name  # "FirstName LastName"

    if isinstance(source, SocialMediaUser):
        if source.follower_count >= 0:
            return f"{source.name} ({source.follower_count:,} followers)"
        return source.name

    if isinstance(source, GovernmentSource):
        return source.name

    # Regular Source (newspaper etc.)
    if show_circulation and hasattr(source, "has_circulation") and source.has_circulation:
        return f"{source.name} (circulation: {source.circulation:,})"
    return source.name


def create_entity_table(
    entity: Entity,
    conflict_field: str,
    conflict_value: str,
    regional_location: Optional[str] = None,
) -> Dict[str, str]:
    """Build a field -> value mapping for a single entity table row set.

    The returned dictionary is ordered as: Name first, then optionally a
    Location row (for regionality experiments), followed by all non-excluded
    entity attributes.  The *conflict_field* is overridden with
    *conflict_value*; all other attributes use their first stored value.

    Args:
        entity: The entity whose attributes populate the table.
        conflict_field: The field that carries the conflicting value.
        conflict_value: The value to insert for *conflict_field*.
        regional_location: If provided, inserted as a ``Location`` row.

    Returns:
        Ordered ``dict`` mapping display field names to display values.
    """
    table: Dict[str, str] = {}

    # Name is always the first row
    table["Name"] = entity.name

    # Optional location row for regionality experiments
    if regional_location:
        table["Location"] = regional_location

    # All other non-excluded attributes
    for field_name, values in entity.attributes.items():
        if field_name in EXCLUDED_FIELDS:
            continue

        if field_name == conflict_field:
            display_value = _format_value(conflict_value)
        else:
            display_value = _format_value(values[0] if values else "")

        display_field = field_name.replace("_", " ").title()
        table[display_field] = display_value

    return table


def to_markdown_table(data: Dict[str, str]) -> str:
    """Convert a field/value dictionary to a Markdown table string.

    Args:
        data: Mapping of field names to values.

    Returns:
        A string containing a Markdown table with ``Field`` and ``Value``
        columns, including header and separator rows.
    """
    header = "| Field | Value |\n|-------|-------|"
    if not data:
        return header
    rows = [f"| {key} | {value} |" for key, value in data.items()]
    return header + "\n" + "\n".join(rows)


def format_answer_choices(
    value_a: str,
    value_b: str,
    neutral: bool = False,
    alttok: bool = False,
    neutral_position: int = 2,
) -> str:
    """Format the multiple-choice answer block appended to every prompt.

    Args:
        value_a: Display text for the first real answer.
        value_b: Display text for the second real answer.
        neutral: Whether to include a third "Can't be answered" option.
        alttok: If ``True``, use ``(1)``, ``(2)``, ``(3)`` instead of
            ``(A)``, ``(B)``, ``(C)``.
        neutral_position: Where to place the neutral option when *neutral* is
            ``True``.  ``0`` = first, ``1`` = middle, ``2`` = last (default).

    Returns:
        Multi-line string with formatted answer choices.
    """
    tokens = ["1", "2", "3"] if alttok else ["A", "B", "C"]
    neutral_text = "Can't be answered"

    if not neutral:
        return f"({tokens[0]}) {value_a}\n({tokens[1]}) {value_b}"

    if neutral_position == 0:
        return (
            f"({tokens[0]}) {neutral_text}\n"
            f"({tokens[1]}) {value_a}\n"
            f"({tokens[2]}) {value_b}"
        )
    if neutral_position == 1:
        return (
            f"({tokens[0]}) {value_a}\n"
            f"({tokens[1]}) {neutral_text}\n"
            f"({tokens[2]}) {value_b}"
        )
    # neutral_position == 2 (default)
    return (
        f"({tokens[0]}) {value_a}\n"
        f"({tokens[1]}) {value_b}\n"
        f"({tokens[2]}) {neutral_text}"
    )


# =============================================================================
# Internal helpers
# =============================================================================

def _format_value(value: str) -> str:
    """Capitalise the first character while lower-casing the rest."""
    if not value:
        return value
    value_str = str(value)
    if len(value_str) <= 1:
        return value_str.upper()
    return value_str[0].upper() + value_str[1:].lower()


def _fix_template_format(template: str, field: str) -> str:
    """Normalise an LLM-generated question template to use ``{name}``."""
    template = template.strip("\"'")
    template = template.replace("{entity_name}", "{name}")
    template = template.replace("{{name}}", "{name}")

    if "{name}" not in template and "name}" not in template:
        formatted_field = field.replace("_", " ")
        return f"What is {{name}}'s {formatted_field}?"
    return template


# =============================================================================
# Question generation
# =============================================================================

def generate_question(
    entity: Entity,
    field: str,
    question_cache: Dict[str, str],
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> str:
    """Generate a natural-language question asking about an entity's field.

    The function first checks the *question_cache* for a cached template (keyed
    by ``"{entity_class}_{field}"``).  If no cached template exists and a
    *model* / *tokenizer* pair is provided the LLM is used to generate one;
    otherwise a simple fallback template is used.  The resulting template is
    cached and returned with the entity name substituted in.

    Args:
        entity: The entity the question is about.
        field: The attribute field to ask about.
        question_cache: Mutable dictionary used as a persistent cache.
        model: Optional HuggingFace language model for generation.
        tokenizer: Matching tokenizer for *model*.

    Returns:
        A natural-language question string with the entity name inserted.
    """
    entity_class = entity.entity_class.value
    cache_key = f"{entity_class}_{field}"

    # Check cache first
    if cache_key in question_cache:
        template = question_cache[cache_key]
        return template.replace("{name}", entity.name)

    # Try LLM generation
    template = None
    if model is not None and tokenizer is not None:
        try:
            template = _generate_with_llm(entity_class, field, model, tokenizer)
        except Exception as exc:
            logger.warning(
                "LLM question generation failed for %s_%s: %s",
                entity_class,
                field,
                exc,
            )

    # Fallback
    if template is None:
        formatted_field = field.replace("_", " ")
        template = f"What is {{name}}'s {formatted_field}?"

    question_cache[cache_key] = template
    return template.replace("{name}", entity.name)


def _generate_with_llm(
    entity_class: str, field: str, model: Any, tokenizer: Any
) -> str:
    """Use the loaded LLM to produce a question template."""
    prompt = QUESTION_GENERATION_PROMPT.format(
        entity_class=entity_class,
        field_name=field.replace("_", " "),
    )

    chat_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(chat_input, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            do_sample=LLM_DO_SAMPLE,
            temperature=LLM_TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    template = generated_text.strip()
    return _fix_template_format(template, field)


# =============================================================================
# Two-table prompt builder
# =============================================================================

def build_two_table_prompt(
    entity: Entity,
    conflict_field: str,
    value_a: str,
    value_b: str,
    source_a_info: Union[Source, SocialMediaUser, GovernmentSource, PersonSource],
    source_b_info: Union[Source, SocialMediaUser, GovernmentSource, PersonSource],
    question: str,
    neutral: bool = False,
    alttok: bool = False,
    neutral_position: int = 2,
    snm: bool = False,
    both_no_source: bool = False,
    source_a_is_no_source: bool = False,
    source_b_is_no_source: bool = False,
    show_circulation: bool = False,
    regional_location: Optional[str] = None,
) -> str:
    """Build a complete two-table prompt with conflicting source information.

    The prompt has the following structure::

        Context:
        Table A (Source: <source>):
        | Field | Value |
        |-------|-------|
        | Name  | ...   |
        ...

        Table B (Source: <source>):
        | Field | Value |
        |-------|-------|
        ...

        <question>
        (A) <value_a>
        (B) <value_b>

    Special display modes:

    * **both_no_source** -- Omit source labels entirely from both tables.
    * **snm** (source-not-mentioned) -- Use ``"(from XYZ)"`` instead of
      ``"(Source: XYZ)"`` and hide the label when a source is "No source
      available".
    * **neutral** -- Add a ``"Can't be answered"`` option at the position
      specified by *neutral_position*.
    * **alttok** -- Use ``(1)``, ``(2)``, ``(3)`` instead of ``(A)``, ``(B)``,
      ``(C)``.

    Args:
        entity: The entity whose attributes populate the tables.
        conflict_field: The attribute that has conflicting values.
        value_a: Value shown in Table A for *conflict_field*.
        value_b: Value shown in Table B for *conflict_field*.
        source_a_info: Source object for Table A.
        source_b_info: Source object for Table B.
        question: The natural-language question to pose.
        neutral: Include a neutral answer option.
        alttok: Use numeric tokens ``(1)``/``(2)``/``(3)``.
        neutral_position: Position of the neutral option (0/1/2).
        snm: Source-not-mentioned mode.
        both_no_source: Both sources are absent (control condition).
        source_a_is_no_source: Source A should be treated as absent.
        source_b_is_no_source: Source B should be treated as absent.
        show_circulation: Display circulation/follower numbers.
        regional_location: Optional location string for regionality tables.

    Returns:
        The fully formatted prompt string.
    """
    # Build entity tables
    table_a = create_entity_table(entity, conflict_field, value_a, regional_location)
    table_b = create_entity_table(entity, conflict_field, value_b, regional_location)

    table_a_md = to_markdown_table(table_a)
    table_b_md = to_markdown_table(table_b)

    prompt = "Context:\n"

    # ------------------------------------------------------------------
    # Case 1: both sources absent (control condition)
    # ------------------------------------------------------------------
    if both_no_source:
        prompt += f"Table A:\n{table_a_md}\n\n"
        prompt += f"Table B:\n{table_b_md}\n\n"

    # ------------------------------------------------------------------
    # Case 2: SNM mode
    # ------------------------------------------------------------------
    elif snm:
        src_a_display = format_source_info(source_a_info, show_circulation)
        src_b_display = format_source_info(source_b_info, show_circulation)

        # Table A
        if source_a_is_no_source:
            prompt += f"Table A:\n{table_a_md}\n\n"
        else:
            prompt += f"Table A (from {src_a_display}):\n{table_a_md}\n\n"

        # Table B
        if source_b_is_no_source:
            prompt += f"Table B:\n{table_b_md}\n\n"
        else:
            prompt += f"Table B (from {src_b_display}):\n{table_b_md}\n\n"

    # ------------------------------------------------------------------
    # Case 3: standard (source in caption)
    # ------------------------------------------------------------------
    else:
        src_a_display = format_source_info(source_a_info, show_circulation)
        src_b_display = format_source_info(source_b_info, show_circulation)
        prompt += f"Table A (Source: {src_a_display}):\n{table_a_md}\n\n"
        prompt += f"Table B (Source: {src_b_display}):\n{table_b_md}\n\n"

    # Question and answer choices
    prompt += f"\n{question}\n"
    prompt += format_answer_choices(value_a, value_b, neutral, alttok, neutral_position)
    return prompt


# =============================================================================
# Verbose majority prompt (3 tables)
# =============================================================================

def build_2tm_prompt(
    entity: Entity,
    conflict_field: str,
    value_high: str,
    value_low: str,
    high_auth_source_info: Union[Source, SocialMediaUser, GovernmentSource, PersonSource],
    low_auth_source_infos: Union[SourceGroup, List[Any]],
    table_order: List[str],
    answer_order: str,
    question: str,
    neutral: bool = False,
    alttok: bool = False,
    neutral_position: int = 2,
    regional_location: Optional[str] = None,
) -> str:
    """Build a three-table prompt for verbose majority experiments.

    Three tables are displayed in the order specified by *table_order*.  Each
    element is either ``'H'`` (high-authority source / value) or ``'L'``
    (low-authority / majority source / value).  The answer options are ordered
    according to *answer_order*: ``'AH_BL'`` puts the high-authority value as
    choice A and the low-authority value as choice B, while ``'AL_BH'`` does
    the opposite.

    Args:
        entity: Entity whose attributes populate the tables.
        conflict_field: The conflicting attribute field.
        value_high: Value supported by the high-authority source.
        value_low: Value supported by the low-authority (majority) sources.
        high_auth_source_info: The single high-authority source object.
        low_auth_source_infos: ``SourceGroup`` or list of low-authority sources
            (expects at least as many entries as ``'L'`` slots in *table_order*).
        table_order: e.g. ``['H', 'L', 'L']`` -- order of high/low tables.
        answer_order: ``'AH_BL'`` or ``'AL_BH'``.
        question: The natural-language question to pose.
        neutral: Include a neutral answer option.
        alttok: Use numeric tokens.
        neutral_position: Position of the neutral option.
        regional_location: Optional location for regionality experiments.

    Returns:
        The fully formatted prompt string.
    """
    # Normalise low_auth_source_infos to a list
    if isinstance(low_auth_source_infos, SourceGroup):
        low_sources = low_auth_source_infos.sources
    else:
        low_sources = list(low_auth_source_infos)

    table_high = create_entity_table(entity, conflict_field, value_high, regional_location)
    table_low = create_entity_table(entity, conflict_field, value_low, regional_location)

    table_high_md = to_markdown_table(table_high)
    table_low_md = to_markdown_table(table_low)

    prompt = "Context:\n"

    low_idx = 0
    for i, table_type in enumerate(table_order):
        label = chr(65 + i)  # A, B, C, ...

        if table_type == "H":
            source_display = format_source_info(high_auth_source_info, show_circulation=True)
            md = table_high_md
        else:  # "L"
            low_source = low_sources[low_idx]
            source_display = format_source_info(low_source, show_circulation=True)
            md = table_low_md
            low_idx += 1

        prompt += f"Table {label} (Source: {source_display}):\n{md}\n\n"

    # Determine answer value assignment
    if answer_order == "AH_BL":
        val_a, val_b = value_high, value_low
    else:  # "AL_BH"
        val_a, val_b = value_low, value_high

    prompt += f"\n{question}\n"
    prompt += format_answer_choices(val_a, val_b, neutral, alttok, neutral_position)
    return prompt


# =============================================================================
# Implicit majority prompt (2 tables with source lists)
# =============================================================================

def build_1tm_prompt(
    entity: Entity,
    conflict_field: str,
    value_high: str,
    value_low: str,
    high_source_info: Union[Source, SocialMediaUser, GovernmentSource, PersonSource],
    low_source_infos: Union[SourceGroup, List[Any]],
    swap_tables: bool,
    question: str,
    neutral: bool = False,
    alttok: bool = False,
    neutral_position: int = 2,
    regional_location: Optional[str] = None,
) -> str:
    """Build a two-table prompt where sources are shown as comma-separated lists.

    Unlike the verbose majority prompt which uses three separate tables, this
    variant presents two tables with the source attribution listing multiple
    sources (the majority side) in a single comma-separated string.

    Args:
        entity: Entity whose attributes populate the tables.
        conflict_field: The conflicting attribute field.
        value_high: Value from the high-authority (single) source.
        value_low: Value from the low-authority (majority) sources.
        high_source_info: The single high-authority source object.
        low_source_infos: ``SourceGroup`` or list of low-authority sources.
        swap_tables: If ``False``, authority is in Table A and majority in
            Table B.  If ``True``, the positions are swapped.
        question: The natural-language question to pose.
        neutral: Include a neutral answer option.
        alttok: Use numeric tokens.
        neutral_position: Position of the neutral option.
        regional_location: Optional location for regionality experiments.

    Returns:
        The fully formatted prompt string.
    """
    # Normalise low sources
    if isinstance(low_source_infos, SourceGroup):
        low_sources = low_source_infos.sources
    else:
        low_sources = list(low_source_infos)

    table_high = create_entity_table(entity, conflict_field, value_high, regional_location)
    table_low = create_entity_table(entity, conflict_field, value_low, regional_location)

    table_high_md = to_markdown_table(table_high)
    table_low_md = to_markdown_table(table_low)

    # Format source strings
    high_source_str = format_source_info(high_source_info, show_circulation=True)
    low_source_names = [
        format_source_info(s, show_circulation=True) for s in low_sources
    ]
    low_source_str = ", ".join(low_source_names)

    prompt = "Context:\n"

    # Determine table content and answer ordering based on swap_tables
    if not swap_tables:
        val_a, val_b = value_high, value_low
    else:
        val_a, val_b = value_low, value_high

    if not swap_tables:
        first_md, first_src = table_high_md, high_source_str
        second_md, second_src = table_low_md, low_source_str
    else:
        first_md, first_src = table_low_md, low_source_str
        second_md, second_src = table_high_md, high_source_str

    prompt += f"Table A (Sources: {first_src}):\n{first_md}\n\n"
    prompt += f"Table B (Sources: {second_src}):\n{second_md}\n\n"

    prompt += f"\n{question}\n"
    prompt += format_answer_choices(val_a, val_b, neutral, alttok, neutral_position)
    return prompt


# =============================================================================
# Explicit trust prompt (no tables)
# =============================================================================

def build_explicit_prompt(
    question_text: str,
    source_a_info: Union[str, Source, SocialMediaUser, GovernmentSource, PersonSource, SourceGroup],
    source_b_info: Union[str, Source, SocialMediaUser, GovernmentSource, PersonSource, SourceGroup],
    neutral: bool = False,
) -> str:
    """Build an explicit trust-evaluation prompt without tables.

    The prompt simply presents the question and two source options::

        <question>
        (A) <source_a>
        (B) <source_b>

    With an optional neutral third choice.

    Args:
        question_text: The trust question (e.g. "Who do you trust more?").
        source_a_info: First source (object or pre-formatted string).
        source_b_info: Second source (object or pre-formatted string).
        neutral: Whether to add a ``"(C) Neither/Can't determine"`` option.

    Returns:
        The formatted prompt string.
    """
    # Allow both pre-formatted strings and source objects
    if isinstance(source_a_info, str):
        src_a = source_a_info
    else:
        src_a = format_source_info(source_a_info, show_circulation=True)

    if isinstance(source_b_info, str):
        src_b = source_b_info
    else:
        src_b = format_source_info(source_b_info, show_circulation=True)

    prompt = f"{question_text}\n(A) {src_a}\n(B) {src_b}"

    if neutral:
        prompt += "\n(C) Neither/Can't determine"

    return prompt


# =============================================================================
# Chat formatting
# =============================================================================

def format_as_chat(
    prompt: str,
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
) -> List[Dict[str, str]]:
    """Wrap a prompt into the chat-message list format expected by HuggingFace.

    Args:
        prompt: The user-facing prompt content.
        system_message: The system instruction prepended to the conversation.

    Returns:
        A list of two message dictionaries (system + user).
    """
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
