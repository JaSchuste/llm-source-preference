"""
Data loading utilities for the source preference evaluation framework.

This module handles loading all data files from a local directory. The data
is expected to be downloaded from the HuggingFace Hub (see download_dataset).

Typical usage:
    from helpers.data_loader import load_entities, load_source_templates

    entities = load_entities("data/")
    templates = load_source_templates("data/")
"""

import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from helpers.models import Entity
from helpers.config import DATA_FILES, DEFAULT_DATA_DIR, HF_DATASET_REPO, HF_ENCRYPTION_KEY


# =============================================================================
# Key derivation
# =============================================================================

def _derive_fernet_key(passphrase: str) -> bytes:
    """Derive a 32-byte Fernet key from *passphrase* using PBKDF2-SHA256.

    Uses a fixed public salt (``b"llm-source-preference"``) and 100 000
    iterations.  Both upload and download call this function so the key is
    always identical for a given passphrase.
    """
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"llm-source-preference",
        iterations=100_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))


# =============================================================================
# Dataset download
# =============================================================================

def download_dataset(
    repo_id: str = HF_DATASET_REPO,
    data_dir: str = DEFAULT_DATA_DIR,
    passphrase: str = HF_ENCRYPTION_KEY,
) -> None:
    """Download and decrypt the dataset from HuggingFace Hub.

    Downloads each encrypted ``.enc`` file from *repo_id*, derives a Fernet
    key from *passphrase* via PBKDF2-SHA256, decrypts every file, and writes
    the plaintext to *data_dir*.  Files that are already present are skipped.

    Args:
        repo_id: HuggingFace dataset repository ID
            (e.g. ``"JaSchuste/llm-source-preference"``).
        data_dir: Local directory where data files will be stored.
        passphrase: Encryption passphrase (``HF_ENCRYPTION_KEY`` in
            ``helpers/config.py``).  Must match the value used during upload.

    Raises:
        ImportError: If ``cryptography`` or ``huggingface_hub`` are not installed.
    """
    if not passphrase:
        raise ValueError(
            "Encryption passphrase not set. "
            "Export the LLM_SP_KEY environment variable before running:\n"
            "  export LLM_SP_KEY='<passphrase>'"
        )

    try:
        from cryptography.fernet import Fernet
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Missing dependency for dataset download. "
            "Run: pip install cryptography huggingface_hub"
        ) from exc

    fernet = Fernet(_derive_fernet_key(passphrase))
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for label, filename in DATA_FILES.items():
        out_path = data_path / filename
        if out_path.exists():
            print(f"[skip] {filename} already exists.")
            continue

        enc_filename = filename + ".enc"
        print(f"Downloading {enc_filename} ...")
        enc_path = hf_hub_download(
            repo_id=repo_id,
            filename=enc_filename,
            repo_type="dataset",
        )
        with open(enc_path, "rb") as f:
            encrypted_data = f.read()

        decrypted_data = fernet.decrypt(encrypted_data)
        with open(out_path, "wb") as f:
            f.write(decrypted_data)
        print(f"  → saved to {out_path}")

    print(f"Dataset ready in '{data_dir}'.")


# =============================================================================
# Entity loading
# =============================================================================

def load_entities(data_dir: str = DEFAULT_DATA_DIR) -> List[Entity]:
    """Load NeoQA entities from the JSONL file.

    Each line in the file is a JSON object representing one entity. Fields
    beyond the metadata (id, name, entity_class, timeline_id, description)
    are treated as attributes with potentially multiple conflicting values.

    Args:
        data_dir: Directory containing the data files.

    Returns:
        List of Entity objects parsed from the file.

    Raises:
        FileNotFoundError: If the entities file does not exist.
    """
    filepath = Path(data_dir) / DATA_FILES["entities"]
    if not filepath.exists():
        raise FileNotFoundError(
            f"Entities file not found: {filepath}. "
            f"Run download_dataset() or place the file manually."
        )

    entities = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entities.append(Entity.from_dict(data))

    return entities


# =============================================================================
# Question cache
# =============================================================================

def load_question_cache(data_dir: str = DEFAULT_DATA_DIR) -> Dict[str, str]:
    """Load the question cache mapping (field_key -> generated question).

    The cache stores previously generated questions so they are consistent
    across runs and do not require re-generation via an LLM.

    Args:
        data_dir: Directory containing the data files.

    Returns:
        Dictionary mapping cache keys to question strings.
        Returns an empty dict if the file does not exist.
    """
    filepath = Path(data_dir) / DATA_FILES["question_cache"]
    if not filepath.exists():
        return {}

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_question_cache(cache: Dict[str, str], data_dir: str = DEFAULT_DATA_DIR) -> None:
    """Save the question cache to disk.

    Args:
        cache: Dictionary mapping cache keys to question strings.
        data_dir: Directory containing the data files.
    """
    filepath = Path(data_dir) / DATA_FILES["question_cache"]
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# =============================================================================
# Source templates
# =============================================================================

def load_source_templates(data_dir: str = DEFAULT_DATA_DIR) -> List[str]:
    """Load newspaper name templates.

    Each line in the file is a template string containing a {'NAME'}
    placeholder that will be replaced with a location name to generate
    realistic newspaper names (e.g. "The {NAME} Times").

    Args:
        data_dir: Directory containing the data files.

    Returns:
        List of template strings.

    Raises:
        FileNotFoundError: If the templates file does not exist.
    """
    filepath = Path(data_dir) / DATA_FILES["newspaper_templates"]
    if not filepath.exists():
        raise FileNotFoundError(
            f"Newspaper templates file not found: {filepath}. "
            f"Run download_dataset() or place the file manually."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        templates = [line.strip() for line in f if line.strip()]

    return templates


# =============================================================================
# Timeline locations
# =============================================================================

def load_timeline_locations(data_dir: str = DEFAULT_DATA_DIR) -> Dict[str, Any]:
    """Load timeline location data.

    This maps timeline IDs to location names used for generating
    geographically appropriate newspaper names.

    Args:
        data_dir: Directory containing the data files.

    Returns:
        Dictionary of timeline location data.
        Returns an empty dict if the file does not exist.
    """
    filepath = Path(data_dir) / DATA_FILES["timeline_locations"]
    if not filepath.exists():
        return {}

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Government templates
# =============================================================================

def load_government_templates(data_dir: str = DEFAULT_DATA_DIR) -> Dict[str, Any]:
    """Load government/institutional source templates.

    The file contains a single JSON object where keys are entity class
    names and values are template structures for generating government
    source names (e.g. "Ministry of {entity_class}" patterns).

    Args:
        data_dir: Directory containing the data files.

    Returns:
        Dictionary mapping entity classes to their government templates.

    Raises:
        FileNotFoundError: If the templates file does not exist.
    """
    filepath = Path(data_dir) / DATA_FILES["government_templates"]
    if not filepath.exists():
        raise FileNotFoundError(
            f"Government templates file not found: {filepath}. "
            f"Run download_dataset() or place the file manually."
        )

    # The file is named .jsonl but contains a single JSON object.
    # Strip trailing commas before ] or } to handle non-standard JSON.
    import re
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r",(\s*[\]\}])", r"\1", content)
    return json.loads(content)


# =============================================================================
# Circulation values
# =============================================================================

def load_circulation_values(data_dir: str = DEFAULT_DATA_DIR) -> List[int]:
    """Load newspaper circulation values.

    These are real-world circulation numbers used to assign realistic
    circulation data to generated newspaper sources.

    Args:
        data_dir: Directory containing the data files.

    Returns:
        List of integer circulation values.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(data_dir) / DATA_FILES["circulation_values"]
    if not filepath.exists():
        raise FileNotFoundError(
            f"Circulation values file not found: {filepath}. "
            f"Run download_dataset() or place the file manually."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Person names
# =============================================================================

def load_names(data_dir: str = DEFAULT_DATA_DIR) -> Dict[str, Any]:
    """Load all person name lists used for generating person sources.

    Loads first names grouped by gender and age, plus last names.
    The JSON files contain lists of name strings. The last names file
    is a plain text file with one name per line.

    Args:
        data_dir: Directory containing the data files.

    Returns:
        Dictionary with keys:
            - "male": list of male first names
            - "female": list of female first names
            - "young": list of first names associated with younger people
            - "middle": list of first names for middle-aged people
            - "old": list of first names associated with older people
            - "last": list of last names
    """
    data_path = Path(data_dir)
    names = {}

    # Load first name lists from JSON files
    name_categories = {
        "male": DATA_FILES["names_male"],
        "female": DATA_FILES["names_female"],
        "young": DATA_FILES["names_young"],
        "middle": DATA_FILES["names_middle"],
        "old": DATA_FILES["names_old"],
    }

    for category, filename in name_categories.items():
        filepath = data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Name file not found: {filepath}. "
                f"Run download_dataset() or place the file manually."
            )
        with open(filepath, "r", encoding="utf-8") as f:
            names[category] = json.load(f)

    # Load last names from plain text file (one per line)
    last_names_path = data_path / DATA_FILES["names_last"]
    if not last_names_path.exists():
        raise FileNotFoundError(
            f"Last names file not found: {last_names_path}. "
            f"Run download_dataset() or place the file manually."
        )
    with open(last_names_path, "r", encoding="utf-8") as f:
        names["last"] = [line.strip() for line in f if line.strip()]

    return names


# =============================================================================
# Explicit trust questions
# =============================================================================

def load_explicit_questions(data_dir: str = DEFAULT_DATA_DIR) -> List[str]:
    """Load explicit trust questions (one per line).

    These are used in the explicit trust experiment where the model is
    directly asked which source it trusts more, without a factual question.

    Args:
        data_dir: Directory containing the data files.

    Returns:
        List of question strings. Returns an empty list if the file
        does not exist (the explicit experiment is optional).
    """
    filepath = Path(data_dir) / "explicit_questions.txt"
    if not filepath.exists():
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    return questions
