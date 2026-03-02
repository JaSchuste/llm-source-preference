"""
Generate alternative field values for NeoQA entities.

Usage::

    python generate_entity_field_alternatives.py \\
        --input  neoqa_entities.jsonl \\
        --output neoqa_entities_with_alternatives.jsonl \\
        --model  Qwen/Qwen2-72B-Instruct
"""

import argparse
import json
import logging
import random
import re
from datetime import datetime, timedelta

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXCLUDED_FIELDS = {
    "name", "id", "entity_class", "description", "timeline_id", "gender", "spouse"
}
NUM_ALTERNATIVES = 4
MAX_RETRIES = 2

CATEGORICAL_ALTERNATIVES = {
    "material": [
        "stone and timber", "brick and wood", "steel and concrete",
        "glass and aluminum", "bamboo and steel", "limestone and glass",
        "sandstone and oak", "ceramic and metal", "slate and pine",
        "brick and concrete", "glass and steel", "wood and concrete",
        "stone and glass", "timber and concrete", "brick and stone",
        "wood and aluminum", "concrete and aluminum", "glass and timber",
        "stone and steel", "brick and steel", "concrete and glass",
        "timber and glass", "brick and timber", "stone and concrete",
        "wood and steel", "glass and copper", "concrete and copper",
        "steel and aluminum", "concrete and stone", "wood and brick",
    ],
    "marital_status": [
        "single", "married", "divorced", "widowed", "separated",
        "in a domestic partnership", "in a civil partnership", "engaged",
        "cohabiting",
    ],
    "eye_color": [
        "brown", "blue", "green", "hazel", "grey", "amber",
        "black", "dark brown", "light brown", "dark blue",
        "light blue", "emerald", "golden brown",
    ],
    "hair_color": [
        "black", "brown", "blonde", "red", "gray", "white",
        "dark brown", "light brown", "dirty blonde", "strawberry blonde",
        "auburn", "chestnut", "platinum blonde", "raven black",
        "silver", "green dyed", "blue dyed", "pink dyed",
    ],
}

PROMPT_TEMPLATE = (
    "You are an AI assistant tasked with creating fictional entities based on "
    "provided information. Your goal is to generate detailed, coherent, and "
    "realistic alternatives for values of existing locations, persons, "
    "organisations, products, art, buildings and events. You will be given "
    "information about one entity in JSON format and a field for which you are "
    "supposed to generate reasonable, realistic and plausible alternative values. "
    "For example, make sure that professions fit the education level and "
    "background of the original entity. If these values are entities themselves, "
    "make sure they are fictional.\n\n"
    "Reply with exactly four alternative values, each on a separate line, "
    'prefixed with "ALT: ". Do not include any other text.\n\n'
    "Example format:\n"
    "ALT: Alternative value 1\n"
    "ALT: Alternative value 2\n"
    "ALT: Alternative value 3\n"
    "ALT: Alternative value 4\n\n"
    "Entity Information:\n{entity_json}\n\n"
    "Target field:\n{target_field}\n"
)

# ---------------------------------------------------------------------------
# Precision helpers
# ---------------------------------------------------------------------------

def _count_trailing_zeros(n: int) -> int:
    """Return the number of trailing decimal zeros in *n*."""
    if n == 0:
        return 0
    n = abs(n)
    count = 0
    while n % 10 == 0:
        n //= 10
        count += 1
    return count


def _snap_to_precision(value: float, original: int) -> int:
    """Round *value* to the precision allowed relative to *original*.

    Alternatives may be at most one digit more precise than *original*, i.e.
    must be a multiple of ``10 ** max(0, trailing_zeros(original) - 1)``.

    Example: original=1_500_000 has 5 trailing zeros → unit=10_000; any
    multiple of 10_000 is a valid alternative (1,550,000 ✓, 1,678,000 ✗).
    """
    tz = _count_trailing_zeros(original)
    unit = max(1, 10 ** max(0, tz - 1))
    result = int(round(value / unit)) * unit
    return max(unit, result)


def _vary_integer(base: int, n: int = NUM_ALTERNATIVES, variation: float = 0.2) -> list:
    """Return *n* distinct integers near *base*, each snapped to allowed precision.

    Uses multiplicative variation of ±*variation*.  Retries up to 20 times
    per sample to find a value not already in the result set.
    """
    results = []
    used = {base}
    for _ in range(n):
        for _ in range(20):
            delta = random.uniform(-variation, variation)
            candidate = _snap_to_precision(max(1, base * (1 + delta)), base)
            if candidate not in used:
                used.add(candidate)
                results.append(candidate)
                break
        else:
            # Fallback: step away by one precision unit
            tz = _count_trailing_zeros(base)
            unit = max(1, 10 ** max(0, tz - 1))
            for offset in [1, -1, 2, -2, 3, -3, 4, -4]:
                candidate = base + offset * unit
                if candidate > 0 and candidate not in used:
                    used.add(candidate)
                    results.append(candidate)
                    break
            else:
                results.append(base)
    return results


# ---------------------------------------------------------------------------
# Numeric / categorical alternative generators
# ---------------------------------------------------------------------------

def _is_numeric_field(field_name: str, value: str) -> bool:
    """Return True if this field can be handled without an LLM."""
    if not isinstance(value, str):
        return False
    if field_name.lower() in CATEGORICAL_ALTERNATIVES:
        return True
    if field_name.lower() == "price":
        return True
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", value.strip()):
        return True
    count_fields = {"number_of_participants", "number_of_employees", "capacity"}
    if field_name.lower() in count_fields and re.match(r"^[\d,]+$", value.strip()):
        return True
    numeric_patterns = [
        r"^[\d,]+\s*(people|employees|participants)$",
        r"^\$[\d,]+(?:\.\d{2})?(?:\s*million)?$",
        r"^[\d,]+\s*(?:sq\.?\s*(?:miles|feet|kilometers|meters)|acres)$",
        r"^[\d,]+\s*(?:feet|meters)$",
        r"^\d+\'[\d\"]*\"?$",
        r"^[\d,]+\s*(?:lbs?|kg)$",
        r"^[\d,]+$",
        r"^\d+\s*(?:hours?|days?|minutes?)$",
        r"^\d{4}$",
    ]
    return any(re.match(p, value.strip(), re.IGNORECASE) for p in numeric_patterns)


def _fmt_int(n: int) -> str:
    return f"{n:,}" if n >= 1000 else str(n)


def _generate_categorical(field_name: str, original: str) -> list:
    pool = CATEGORICAL_ALTERNATIVES[field_name.lower()]
    options = [o for o in pool if o.lower() != original.lower()]
    selected = random.sample(options, min(NUM_ALTERNATIVES, len(options)))
    while len(selected) < NUM_ALTERNATIVES:
        selected.append(random.choice(pool))
    return selected


def _generate_price(original: str) -> list:
    low = original.lower().strip()

    # Non-numeric / free prices
    if low in {"free", "free for membership", "not applicable", "n/a", "na", "varies by package"}:
        pool = [
            "free", "$0.00", "complimentary", "no charge",
            "free with registration", "varies by package",
            "contact for pricing", "free with in-app purchases",
        ]
        options = [o for o in pool if o.lower() != low]
        return random.sample(options, min(NUM_ALTERNATIVES, len(options)))

    # Subscription / per-period descriptions
    if any(kw in low for kw in ("subscription", "per month", "per year", "freemium", "in-app")):
        pool = [
            "free with in-app purchases", "$4.99 per month subscription",
            "one-time purchase of $59.99", "freemium model with premium features",
            "free trial, then $9.99/month", "$2.99 ad-free version",
            "subscription: $19.99/year", "free with ads, $4.99 without ads",
            "$14.99 premium monthly", "annual subscription $99.99",
        ]
        options = [o for o in pool if o.lower() != low]
        return random.sample(options, min(NUM_ALTERNATIVES, len(options)))

    # Custom currency with units: "45,000 Asvelian credits"
    m = re.match(r"^([\d,]+(?:\.\d{2})?)\s+(.+)$", original.strip())
    if m:
        amount_str, unit = m.groups()
        base_int = int(float(amount_str.replace(",", "")))
        return [
            (f"{v:,} {unit}" if v >= 1000 else f"{v} {unit}")
            for v in _vary_integer(base_int)
        ]

    # Standard dollar with cents: "$19.99"
    m = re.match(r"^\$([\d,]+)\.(\d{2})$", original.strip())
    if m:
        base = float(m.group(1).replace(",", "") + "." + m.group(2))
        results, used = [], {base}
        for _ in range(NUM_ALTERNATIVES):
            for _ in range(20):
                new = round(max(0.01, base * (1 + random.uniform(-0.2, 0.2))), 2)
                if new not in used:
                    used.add(new)
                    results.append(f"${new:,.2f}" if new >= 1000 else f"${new:.2f}")
                    break
        return results

    # Standard dollar integer: "$450,000"
    m = re.match(r"^\$([\d,]+)$", original.strip())
    if m:
        base_int = int(m.group(1).replace(",", ""))
        return [f"${_fmt_int(v)}" for v in _vary_integer(base_int)]

    return []


def _generate_numeric_alternatives(field_name: str, original: str) -> list:
    """Return up to NUM_ALTERNATIVES alternatives without using an LLM."""
    if field_name.lower() in CATEGORICAL_ALTERNATIVES:
        return _generate_categorical(field_name, original)

    if field_name.lower() == "price":
        return _generate_price(original)

    # Date: YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", original.strip())
    if m:
        base = datetime(*map(int, m.groups()))
        return [
            (base + timedelta(days=random.randint(-365, 365))).strftime("%Y-%m-%d")
            for _ in range(NUM_ALTERNATIVES)
        ]

    # Year (4-digit standalone)
    if re.match(r"^\d{4}$", original.strip()):
        base = int(original)
        used, alts = {base}, []
        for _ in range(NUM_ALTERNATIVES):
            for _ in range(20):
                new = max(1850, min(2025, base + random.randint(-30, 30)))
                if new not in used:
                    used.add(new)
                    alts.append(str(new))
                    break
            else:
                alts.append(str(random.randint(max(1850, base - 30), min(2025, base + 30))))
        return alts

    # Height: feet/inches "5'5\""
    m = re.match(r"^(\d+)'(\d*)\"?$", original.strip())
    if m:
        total = int(m.group(1)) * 12 + (int(m.group(2)) if m.group(2) else 0)
        alts = []
        for _ in range(NUM_ALTERNATIVES):
            new_total = max(12, int(total * (1 + random.uniform(-0.2, 0.2))))
            f, i = divmod(new_total, 12)
            alts.append(f"{f}'{i}\"" if i else f"{f}'0\"")
        return alts

    # Weight: "135 lbs" / "1.5 kg"
    m = re.match(r"^([\d.]+)\s*(lbs?|kg)$", original.strip(), re.IGNORECASE)
    if m:
        base, unit = float(m.group(1)), m.group(2)
        alts = []
        for _ in range(NUM_ALTERNATIVES):
            new = max(0.1, base * (1 + random.uniform(-0.2, 0.2)))
            alts.append(f"{new:.1f} {unit}" if "kg" in unit.lower() else f"{int(new)} {unit}")
        return alts

    # "$X million" / "$X.Y million"
    m = re.match(r"^\$([\d,]+(?:\.\d+)?)\s*million$", original.strip(), re.IGNORECASE)
    if m:
        base_cents = int(float(m.group(1).replace(",", "")) * 1_000_000)
        return [f"${v / 1_000_000:.1f} million" for v in _vary_integer(base_cents)]

    # Number with units: "1,200 people"
    m = re.match(r"^([\d,]+)\s+(.+)$", original.strip())
    if m:
        base_int = int(m.group(1).replace(",", ""))
        unit = m.group(2)
        return [
            (f"{v:,} {unit}" if v >= 1000 else f"{v} {unit}")
            for v in _vary_integer(base_int)
        ]

    # Pure integer
    if re.match(r"^[\d,]+$", original.strip()):
        base_int = int(original.replace(",", ""))
        return [_fmt_int(v) for v in _vary_integer(base_int)]

    return []


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

def _entity_json_for_prompt(entity: dict) -> str:
    data = {}
    for k, v in entity.items():
        if k in {"id", "entity_class"}:
            continue
        if k == "name" or k not in EXCLUDED_FIELDS:
            data[k] = v[0] if isinstance(v, list) else v
    return json.dumps(data, ensure_ascii=False, indent=2)


def _parse_llm_response(text: str, original: str, entity_id: str, field: str) -> list:
    alts = [
        line[5:].strip()
        for line in text.splitlines()
        if line.strip().startswith("ALT: ") and line.strip()[5:].strip() != original
    ]
    if len(alts) < NUM_ALTERNATIVES:
        logger.warning(
            "Only %d/%d alternatives from LLM for entity=%s field=%s",
            len(alts), NUM_ALTERNATIVES, entity_id, field,
        )
    return alts[:NUM_ALTERNATIVES]


def _generate_llm_alternatives(
    model, tokenizer, entity: dict, field: str, original: str, log_first: list
) -> list:
    """Call the LLM; returns up to NUM_ALTERNATIVES alternative strings."""
    entity_id = entity.get("id", "unknown")
    prompt = PROMPT_TEMPLATE.format(
        entity_json=_entity_json_for_prompt(entity),
        target_field=field,
    )
    if not log_first:
        logger.info("First LLM prompt (entity=%s, field=%s):\n%s", entity_id, field, prompt)
        log_first.append(True)

    for attempt in range(MAX_RETRIES + 1):
        try:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=64,
                    do_sample=True,
                    top_k=50,
                    temperature=1.0 + attempt * 0.1,
                    top_p=0.85,
                    pad_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
            alts = _parse_llm_response(decoded, original, entity_id, field)
            if alts:
                return alts
            if attempt < MAX_RETRIES:
                logger.warning(
                    "Attempt %d yielded no alternatives for entity=%s field=%s, retrying...",
                    attempt + 1, entity_id, field,
                )
        except Exception as e:
            logger.error(
                "Generation attempt %d failed for entity=%s field=%s: %s",
                attempt + 1, entity_id, field, e,
            )
    logger.error("All LLM attempts failed for entity=%s field=%s", entity_id, field)
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate alternative field values for NeoQA entities."
    )
    parser.add_argument("--input", default="neoqa_entities.jsonl")
    parser.add_argument("--output", default="neoqa_entities_with_alternatives.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2-72B-Instruct")
    parser.add_argument("--log-file", default="entity_generation.log")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )

    logger.info("Loading model %s ...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.eval()

    logger.info("Loading entities from %s ...", args.input)
    with open(args.input, encoding="utf-8") as f:
        entities = [json.loads(line) for line in f]

    log_first: list = []
    updated = []
    numeric_count = llm_count = 0

    for entity in tqdm(entities, desc="Processing entities"):
        new_entity = entity.copy()
        for field, value in entity.items():
            if field in EXCLUDED_FIELDS or isinstance(value, list):
                continue
            try:
                if _is_numeric_field(field, value):
                    alts = _generate_numeric_alternatives(field, value)
                    numeric_count += 1
                else:
                    alts = _generate_llm_alternatives(
                        model, tokenizer, entity, field, value, log_first
                    )
                    llm_count += 1
            except Exception as e:
                logger.error(
                    "Error for entity=%s field=%s: %s", entity.get("id", "?"), field, e
                )
                alts = []

            if not alts:
                new_entity[field] = [value]
            else:
                # Pad to NUM_ALTERNATIVES if LLM returned fewer
                while len(alts) < NUM_ALTERNATIVES:
                    alts.append(alts[0])
                new_entity[field] = [value] + alts[:NUM_ALTERNATIVES]

        updated.append(new_entity)

    logger.info(
        "Done: %d numeric fields, %d LLM fields", numeric_count, llm_count
    )
    logger.info("Saving to %s ...", args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        for entity in updated:
            f.write(json.dumps(entity, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
