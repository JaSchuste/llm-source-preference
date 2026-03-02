"""
Evaluation utilities for the source preference evaluation framework.

This module computes Source Preference (SP) and related metrics from raw
experiment result files.  It is designed to be called programmatically from
experiment scripts (which produce the raw results) as well as from the
command line.

Typical programmatic usage::

    from helpers.evaluation import evaluate_results
    evaluate_results(
        results_path="results/raw/high_low_circulation_newspaper/neoqa_llama.json",
        results_dir="results",
    )

The main metric is **Source Preference (SP)**:

    SP_group = mean(P_experiment(group) - P_control(group)) × 100

where the control is the ``control`` (no_source_vs_no_source) experiment.
Significance is assessed with a two-tailed bootstrap test (b=10 000, α=0.05).
"""

import json
import sys
import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Probability remapping for neutral experiments
# =============================================================================

def remap_neutral_probs(entry: Dict[str, Any]) -> Tuple[float, float, float]:
    """Remap raw token probabilities to semantic probabilities for neutral experiments.

    In raw result files, ``prob_a`` / ``prob_b`` / ``prob_c`` are always
    P(token_1) / P(token_2) / P(token_3).  But the "Can't be answered"
    (neutral) option rotates position across the 6 variations:

    - ``variation % 3 == 0``: positions are (neutral, value_a, value_b)
    - ``variation % 3 == 1``: positions are (value_a, neutral, value_b)
    - ``variation % 3 == 2``: positions are (value_a, value_b, neutral)

    Parameters
    ----------
    entry : dict
        A single raw result entry with ``prob_a``, ``prob_b``, ``prob_c``,
        and ``variation`` fields.

    Returns
    -------
    tuple of (float, float, float)
        ``(prob_value_a, prob_value_b, prob_neutral)`` — semantically aligned.
    """
    raw_1 = entry["prob_a"]
    raw_2 = entry["prob_b"]
    raw_3 = entry["prob_c"]
    neutral_pos = entry["variation"] % 3

    if neutral_pos == 0:
        # (1) neutral, (2) value_a, (3) value_b
        return raw_2, raw_3, raw_1
    elif neutral_pos == 1:
        # (1) value_a, (2) neutral, (3) value_b
        return raw_1, raw_3, raw_2
    else:  # neutral_pos == 2
        # (1) value_a, (2) value_b, (3) neutral
        return raw_1, raw_2, raw_3


# =============================================================================
# Position bias
# =============================================================================

def calculate_position_bias(
    data: List[Dict[str, Any]],
    has_neutral: bool,
) -> Dict[str, Any]:
    """Calculate position bias using raw (un-remapped) token probabilities.

    Measures preference toward token position A, B, or C regardless of which
    value / group is at that position.

    Parameters
    ----------
    data : list of dict
        Raw result entries.
    has_neutral : bool
        Whether a third token (neutral option) is present.

    Returns
    -------
    dict
        Keys: ``option_A_winrate``, ``option_B_winrate``, ``tie_rate``,
        ``option_A_avg_probability``, ``option_B_avg_probability``,
        ``total_count``, and optionally ``option_C_winrate`` /
        ``option_C_avg_probability`` when *has_neutral* is True.
    """
    total = len(data)
    wins_a = wins_b = wins_c = ties = 0
    sum_a = sum_b = sum_c = 0.0

    for entry in data:
        pa = entry["prob_a"]
        pb = entry["prob_b"]
        pc = entry.get("prob_c", 0.0) if has_neutral else 0.0

        sum_a += pa
        sum_b += pb
        sum_c += pc

        if has_neutral:
            if pa > pb and pa > pc:
                wins_a += 1
            elif pb > pa and pb > pc:
                wins_b += 1
            elif pc > pa and pc > pb:
                wins_c += 1
            else:
                ties += 1
        else:
            if pa > pb:
                wins_a += 1
            elif pb > pa:
                wins_b += 1
            else:
                ties += 1

    n = total or 1  # avoid division by zero
    result = {
        "option_A_winrate": wins_a / n,
        "option_B_winrate": wins_b / n,
        "tie_rate": ties / n,
        "option_A_avg_probability": sum_a / n,
        "option_B_avg_probability": sum_b / n,
        "total_count": total,
    }
    if has_neutral:
        result["option_C_winrate"] = wins_c / n
        result["option_C_avg_probability"] = sum_c / n

    return result


# =============================================================================
# Source preference
# =============================================================================

def calculate_source_preference(
    data: List[Dict[str, Any]],
    has_neutral: bool,
) -> Dict[str, Dict[str, Any]]:
    """Calculate source preference metrics by group using semantically correct probs.

    For neutral experiments the raw token probabilities are remapped so that
    each group's probability always reflects the correct semantic option,
    regardless of where the neutral option appears.

    Parameters
    ----------
    data : list of dict
        Raw result entries.
    has_neutral : bool
        Whether a third token (neutral option) is present.

    Returns
    -------
    dict
        Mapping from group label to ``{winrate, tie_rate, avg_probability,
        total_count}``.  Includes a ``"neutral"`` key when *has_neutral* is True.
    """
    group_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"wins": 0, "ties": 0.0, "prob_sum": 0.0, "count": 0}
    )
    neutral_stats: Dict[str, Any] = {"wins": 0, "ties": 0.0, "prob_sum": 0.0, "count": 0}

    for entry in data:
        ga = entry["group_a"]
        gb = entry["group_b"]

        if has_neutral:
            pa, pb, pn = remap_neutral_probs(entry)

            group_stats[ga]["count"] += 1
            group_stats[ga]["prob_sum"] += pa
            if pa > pb and pa > pn:
                group_stats[ga]["wins"] += 1

            group_stats[gb]["count"] += 1
            group_stats[gb]["prob_sum"] += pb
            if pb > pa and pb > pn:
                group_stats[gb]["wins"] += 1

            neutral_stats["count"] += 1
            neutral_stats["prob_sum"] += pn
            if pn > pa and pn > pb:
                neutral_stats["wins"] += 1
        else:
            pa = entry["prob_a"]
            pb = entry["prob_b"]

            group_stats[ga]["count"] += 1
            group_stats[ga]["prob_sum"] += pa
            if pa > pb:
                group_stats[ga]["wins"] += 1
            elif pa == pb:
                group_stats[ga]["ties"] += 0.5

            group_stats[gb]["count"] += 1
            group_stats[gb]["prob_sum"] += pb
            if pb > pa:
                group_stats[gb]["wins"] += 1
            elif pa == pb:
                group_stats[gb]["ties"] += 0.5

    results: Dict[str, Dict[str, Any]] = {}
    for group, stats in group_stats.items():
        n = stats["count"] or 1
        results[group] = {
            "winrate": stats["wins"] / n,
            "tie_rate": stats["ties"] / n,
            "avg_probability": stats["prob_sum"] / n,
            "total_count": stats["count"],
        }

    if has_neutral and neutral_stats["count"] > 0:
        n = neutral_stats["count"]
        results["neutral"] = {
            "winrate": neutral_stats["wins"] / n,
            "tie_rate": neutral_stats["ties"] / n,
            "avg_probability": neutral_stats["prob_sum"] / n,
            "total_count": n,
        }

    return results


# =============================================================================
# Bootstrap significance tests
# =============================================================================

def bootstrap_significance_test(
    differences: List[float],
    b: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Two-tailed bootstrap test of whether the mean of *differences* is zero.

    Parameters
    ----------
    differences : list of float
        Per-entry differences (experiment − baseline).
    b : int
        Number of bootstrap resamples.
    alpha : float
        Significance level.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{p_value, is_significant, ci_lower, ci_upper}``
    """
    if not differences:
        return {"p_value": None, "is_significant": None, "ci_lower": None, "ci_upper": None}

    arr = np.array(differences)
    n = len(arr)
    obs_mean = float(np.mean(arr))
    centered = arr - obs_mean

    rng = np.random.RandomState(seed)
    boot_means = np.array([
        np.mean(rng.choice(centered, size=n, replace=True)) for _ in range(b)
    ])

    p_value = float(np.mean(np.abs(boot_means) >= abs(obs_mean)))
    ci_lower = float(np.percentile(boot_means, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_means, (1 - alpha / 2) * 100))

    return {
        "p_value": p_value,
        "is_significant": bool(p_value < alpha),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def bootstrap_avg_prob_significance(
    probs: List[float],
    null_value: float = 0.5,
    b: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """One-sample bootstrap test: H0: mean(probs) == null_value.

    Delegates to :func:`bootstrap_significance_test` by shifting the values
    so that the null hypothesis becomes mean == 0.

    Parameters
    ----------
    probs : list of float
        Per-entry probabilities for a single group.
    null_value : float
        The null-hypothesis mean (default 0.5).
    b, alpha, seed :
        Passed through to :func:`bootstrap_significance_test`.

    Returns
    -------
    dict
        ``{p_value, is_significant, ci_lower, ci_upper}``
    """
    return bootstrap_significance_test(
        [p - null_value for p in probs], b=b, alpha=alpha, seed=seed
    )


def bootstrap_pairwise_comparison_test(
    group1_diffs: List[float],
    group2_diffs: List[float],
    b: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """Two-tailed bootstrap test of whether mean(group1 − group2) is zero.

    Parameters
    ----------
    group1_diffs : list of float
        Per-entry SP differences for group 1 (experiment − baseline).
    group2_diffs : list of float
        Paired SP differences for group 2 (same length as *group1_diffs*).
    b : int
        Number of bootstrap resamples.
    alpha : float
        Significance level.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{p_value, is_significant, ci_lower, ci_upper, mean_difference}``
    """
    if not group1_diffs or len(group1_diffs) != len(group2_diffs):
        return {
            "p_value": None,
            "is_significant": None,
            "ci_lower": None,
            "ci_upper": None,
            "mean_difference": None,
        }

    arr1 = np.array(group1_diffs)
    arr2 = np.array(group2_diffs)
    pairwise = arr1 - arr2
    n = len(pairwise)
    obs_mean = float(np.mean(pairwise))
    centered = pairwise - obs_mean

    rng = np.random.RandomState(seed)
    boot_means = np.array([
        np.mean(rng.choice(centered, size=n, replace=True)) for _ in range(b)
    ])

    p_value = float(np.mean(np.abs(boot_means) >= abs(obs_mean)))
    ci_lower = float(np.percentile(boot_means, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_means, (1 - alpha / 2) * 100))

    return {
        "p_value": p_value,
        "is_significant": bool(p_value < alpha),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "mean_difference": obs_mean,
    }


# =============================================================================
# Reference comparison (SP computation)
# =============================================================================

def calculate_reference_comparison(
    experiment_data: List[Dict[str, Any]],
    reference_data: List[Dict[str, Any]],
    has_neutral: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute SP (source preference) relative to a no-source baseline.

    Matches experiment entries to reference entries on
    ``(entity_id, pair_id, value_a, value_b)`` — or additionally on
    ``variation`` for neutral experiments (since the neutral token position
    differs per variation).

    Parameters
    ----------
    experiment_data : list of dict
        Raw result entries from the experiment of interest.
    reference_data : list of dict
        Raw result entries from the control (no_source_vs_no_source) run.
    has_neutral : bool
        Whether the neutral option is present.

    Returns
    -------
    tuple of (position_results, group_results)
        *position_results* maps token position label (``"A"``, ``"B"``,
        optionally ``"C"``) to SP statistics.

        *group_results* maps group labels to SP statistics including bootstrap
        significance tests.  Each SP value is scaled to percentage points
        (×100).
    """
    # Build reference lookup
    ref_lookup: Dict = {}
    for entry in reference_data:
        if has_neutral:
            key = (
                entry["entity_id"], entry["pair_id"],
                entry["value_a"], entry["value_b"], entry["variation"],
            )
        else:
            key = (entry["entity_id"], entry["pair_id"], entry["value_a"], entry["value_b"])
        ref_lookup[key] = entry

    position_diffs: Dict[str, List[float]] = {"A": [], "B": []}
    position_refs: Dict[str, List[float]] = {"A": [], "B": []}
    if has_neutral:
        position_diffs["C"] = []
        position_refs["C"] = []

    group_diffs: Dict[str, List[float]] = defaultdict(list)
    group_refs: Dict[str, List[float]] = defaultdict(list)
    neutral_diffs: List[float] = []
    neutral_refs: List[float] = []

    # Paired diffs for pairwise significance (keyed by sorted group tuple)
    paired: Dict[tuple, List[Tuple]] = defaultdict(list)

    for entry in experiment_data:
        if has_neutral:
            key = (
                entry["entity_id"], entry["pair_id"],
                entry["value_a"], entry["value_b"], entry["variation"],
            )
        else:
            key = (entry["entity_id"], entry["pair_id"], entry["value_a"], entry["value_b"])

        ref = ref_lookup.get(key)
        if ref is None:
            continue

        ga = entry["group_a"]
        gb = entry["group_b"]

        if has_neutral:
            exp_a, exp_b, exp_n = remap_neutral_probs(entry)
            ref_a, ref_b, ref_n = remap_neutral_probs(ref)

            # Position differences (raw, for positional analysis)
            position_diffs["A"].append(entry["prob_a"] - ref["prob_a"])
            position_diffs["B"].append(entry["prob_b"] - ref["prob_b"])
            position_diffs["C"].append(entry["prob_c"] - ref["prob_c"])
            position_refs["A"].append(ref["prob_a"])
            position_refs["B"].append(ref["prob_b"])
            position_refs["C"].append(ref["prob_c"])

            # Group differences (semantic)
            da = exp_a - ref_a
            db = exp_b - ref_b
            dn = exp_n - ref_n
            group_diffs[ga].append(da)
            group_diffs[gb].append(db)
            group_refs[ga].append(ref_a)
            group_refs[gb].append(ref_b)
            neutral_diffs.append(dn)
            neutral_refs.append(ref_n)

            pair_key = tuple(sorted([ga, gb]))
            paired[pair_key].append((da, db, ga, gb))
        else:
            da = entry["prob_a"] - ref["prob_a"]
            db = entry["prob_b"] - ref["prob_b"]

            position_diffs["A"].append(da)
            position_diffs["B"].append(db)
            position_refs["A"].append(ref["prob_a"])
            position_refs["B"].append(ref["prob_b"])

            group_diffs[ga].append(da)
            group_diffs[gb].append(db)
            group_refs[ga].append(ref["prob_a"])
            group_refs[gb].append(ref["prob_b"])

            pair_key = tuple(sorted([ga, gb]))
            paired[pair_key].append((da, db, ga, gb))

    # Pairwise comparison significance tests
    pairwise_sig: Dict[tuple, Dict[str, Any]] = {}
    for pair_key, entries in paired.items():
        if not entries:
            continue
        anchor_ga = entries[0][2]
        anchor_gb = entries[0][3]
        d1 = [da if ga == anchor_ga else db for da, db, ga, gb in entries]
        d2 = [db if ga == anchor_ga else da for da, db, ga, gb in entries]
        pairwise_sig[pair_key] = {
            "groups": (anchor_ga, anchor_gb),
            "significance": bootstrap_pairwise_comparison_test(d1, d2),
        }

    # Position results
    position_results: Dict[str, Any] = {}
    for pos, diffs in position_diffs.items():
        if not diffs:
            continue
        refs = position_refs[pos]
        position_results[pos] = {
            "sp": float(np.mean(diffs)) * 100,
            "reference_baseline": float(np.mean(refs)),
            "count": len(diffs),
            "significance": bootstrap_significance_test(diffs),
        }

    # Group results (SP + significance)
    group_results: Dict[str, Any] = {}
    for group, diffs in group_diffs.items():
        if not diffs:
            continue
        refs = group_refs[group]
        sp = float(np.mean(diffs)) * 100

        # Individual significance (H0: shift = 0)
        ind_sig = bootstrap_significance_test(diffs)

        # Pairwise significance (H0: group1_shift = group2_shift)
        pair_sig = None
        for pair_key, comp in pairwise_sig.items():
            if group in comp["groups"]:
                pair_sig = comp["significance"]
                break

        group_results[group] = {
            "sp": sp,
            "reference_baseline": float(np.mean(refs)),
            "count": len(diffs),
            "individual_significance": ind_sig,
            "significance": pair_sig,
        }

    # Neutral option differences
    if has_neutral and neutral_diffs:
        group_results["neutral"] = {
            "sp": float(np.mean(neutral_diffs)) * 100,
            "reference_baseline": float(np.mean(neutral_refs)),
            "count": len(neutral_diffs),
            "individual_significance": bootstrap_significance_test(neutral_diffs),
            "significance": None,
        }

    return position_results, group_results


# =============================================================================
# Result compilation
# =============================================================================

def compile_results(
    experiment_type: str,
    model_name: str,
    dataset_size: int,
    position_bias: Dict[str, Any],
    source_preference: Dict[str, Any],
    position_diffs: Optional[Dict[str, Any]],
    group_diffs: Optional[Dict[str, Any]],
    has_neutral: bool,
) -> Dict[str, Any]:
    """Compile all metrics into a single structured dictionary.

    Parameters
    ----------
    experiment_type : str
        Name of the experiment (e.g. ``"high_low_circulation_newspaper"``).
    model_name : str
        Model identifier string.
    dataset_size : int
        Number of raw result entries.
    position_bias : dict
        Output of :func:`calculate_position_bias`.
    source_preference : dict
        Output of :func:`calculate_source_preference`.
    position_diffs : dict or None
        Position SP data from :func:`calculate_reference_comparison`, or
        ``None`` when no reference is available.
    group_diffs : dict or None
        Group SP data from :func:`calculate_reference_comparison`, or
        ``None`` when no reference is available.
    has_neutral : bool
        Whether the neutral option was used.

    Returns
    -------
    dict
        A fully compiled result dictionary ready for JSON serialisation.
    """
    return {
        "experiment_type": experiment_type,
        "model_name": model_name,
        "dataset_size": dataset_size,
        "has_neutral_option": has_neutral,
        "position_bias": position_bias,
        "source_preference": source_preference,
        "reference_comparison": {
            "position_differences": position_diffs,
            "group_differences": group_diffs,
        },
    }


# =============================================================================
# High-level entry point
# =============================================================================

def find_reference_file(
    experiment_type: str,
    model_file: str,
    results_dir: Union[str, Path],
) -> Optional[Path]:
    """Locate the control (no-source) baseline file for *experiment_type*.

    The reference experiment is always ``control`` (the renamed
    ``no_source_vs_no_source``).  When *experiment_type* carries a mode suffix
    (e.g. ``_NEUTRAL``, ``_ALTINST``, ``_SNM``, ``_ALTTOK``), the reference
    file from a matching suffixed control run is used.

    Parameters
    ----------
    experiment_type : str
        The experiment whose reference file is needed.
    model_file : str
        Filename of the raw result file (e.g. ``neoqa_llama.json``).
    results_dir : str or Path
        Root results directory containing ``raw/`` sub-directories.

    Returns
    -------
    Path or None
        Path to the reference file, or *None* when none is found.
    """
    results_dir = Path(results_dir)

    # Explicit experiments do not use the control reference
    if experiment_type.lower().startswith("explicit_") or experiment_type.upper().endswith("_EXPLICIT"):
        return None

    # Build reference experiment name: start with "control" and add any suffix
    ref_name = "control"
    for suffix in ("_NEUTRAL", "_ALTINST", "_SNM", "_ALTTOK"):
        if suffix.lower() in experiment_type.lower():
            ref_name += suffix
            break

    ref_path = results_dir / "raw" / ref_name / model_file
    if ref_path.exists():
        return ref_path

    logger.warning("Reference file not found: %s", ref_path)
    return None


def evaluate_results(
    results_path: Union[str, Path, None] = None,
    results_dir: Optional[Union[str, Path]] = None,
    save: bool = True,
    verbose: bool = True,
    *,
    data: Optional[List[Dict[str, Any]]] = None,
    experiment_type_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate experiment results and compute SP metrics.

    Can be called either with a *results_path* (file-based) or with in-memory
    *data* (skips raw file I/O).  When using in-memory mode, both *data* and
    *experiment_type_override* are required.

    Parameters
    ----------
    results_path : str or Path or None
        Path to the raw JSON result file produced by an experiment script.
        Expected location: ``{results_dir}/raw/{experiment_type}/{model_file}``.
        Mutually exclusive with *data*.
    results_dir : str or Path or None
        Root of the results tree.  Inferred from *results_path* when *None*
        (file-based mode only).
    save : bool
        When *True*, write aggregated results to
        ``{results_dir}/agg/{experiment_type}/{model_file}``.
    verbose : bool
        Print a summary to stdout when *True*.
    data : list of dict or None
        In-memory result entries (replaces loading from *results_path*).
        Requires *experiment_type_override* and *results_dir*.
    experiment_type_override : str or None
        Experiment type string used when *data* is provided (cannot be inferred
        from a file path in that mode).

    Returns
    -------
    dict
        The compiled result dictionary.
    """
    if data is not None:
        # In-memory mode: data provided directly, no raw file needed
        if experiment_type_override is None:
            raise ValueError("experiment_type_override is required when data is provided")
        if results_dir is None:
            raise ValueError("results_dir is required when data is provided")
        if not data:
            raise ValueError("data list is empty")
        experiment_data: List[Dict[str, Any]] = data
        experiment_type = experiment_type_override
        model_name_from_data = experiment_data[0].get("model_name", "unknown")
        model_file = f"neoqa_{model_name_from_data}.json"
        results_dir = Path(results_dir)
    else:
        # File-based mode (original behaviour)
        if results_path is None:
            raise ValueError("Either results_path or data must be provided")
        results_path = Path(results_path)
        if not results_path.exists():
            raise FileNotFoundError(f"Raw results file not found: {results_path}")
        experiment_type = results_path.parent.name
        model_file = results_path.name
        if results_dir is None:
            results_dir = results_path.parent.parent.parent
        results_dir = Path(results_dir)
        with open(results_path, "r", encoding="utf-8") as fh:
            experiment_data = json.load(fh)

    if not experiment_data:
        raise ValueError("Empty result data")

    has_neutral = "prob_c" in experiment_data[0]
    model_name = experiment_data[0].get("model_name", "unknown")

    if verbose:
        print(f"Experiment : {experiment_type}")
        print(f"Model      : {model_name}")
        print(f"Entries    : {len(experiment_data)}")
        print(f"Neutral    : {has_neutral}")

    # Load reference (control baseline) data
    reference_data: Optional[List[Dict[str, Any]]] = None
    ref_path = find_reference_file(experiment_type, model_file, results_dir)
    if ref_path is not None:
        with open(ref_path, "r", encoding="utf-8") as fh:
            reference_data = json.load(fh)
        if verbose:
            print(f"Reference  : {ref_path} ({len(reference_data)} entries)")
    elif verbose and not experiment_type.upper().endswith("_EXPLICIT"):
        print(f"Warning    : No reference file found — skipping SP computation")

    # Compute metrics
    position_bias = calculate_position_bias(experiment_data, has_neutral)
    source_pref = calculate_source_preference(experiment_data, has_neutral)

    position_diffs: Optional[Dict[str, Any]] = None
    group_diffs: Optional[Dict[str, Any]] = None
    if reference_data is not None:
        position_diffs, group_diffs = calculate_reference_comparison(
            experiment_data, reference_data, has_neutral
        )

    # For explicit experiments: significance test H0: avg_probability = 0.50
    if experiment_type.lower().startswith("explicit_"):
        group_probs: Dict[str, List[float]] = defaultdict(list)
        for entry in experiment_data:
            ga, gb = entry["group_a"], entry["group_b"]
            if has_neutral:
                pa, pb, pn = remap_neutral_probs(entry)
                group_probs["neutral"].append(pn)
            else:
                pa, pb = entry["prob_a"], entry["prob_b"]
            group_probs[ga].append(pa)
            group_probs[gb].append(pb)
        for group, probs in group_probs.items():
            if group in source_pref:
                source_pref[group]["avg_probability_significance"] = \
                    bootstrap_avg_prob_significance(probs)

    results = compile_results(
        experiment_type=experiment_type,
        model_name=model_name,
        dataset_size=len(experiment_data),
        position_bias=position_bias,
        source_preference=source_pref,
        position_diffs=position_diffs,
        group_diffs=group_diffs,
        has_neutral=has_neutral,
    )

    # Save aggregated results
    if save:
        out_dir = results_dir / "agg" / experiment_type
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / model_file
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        if verbose:
            print(f"\nSaved to   : {out_path}")

    # Print summary
    if verbose:
        _print_summary(results, group_diffs)

    return results


# =============================================================================
# Console summary helper
# =============================================================================

def _print_summary(results: Dict[str, Any], group_diffs: Optional[Dict[str, Any]]) -> None:
    """Print a concise summary of the evaluation results."""
    print("\n" + "=" * 60)
    print("SOURCE PREFERENCE EVALUATION RESULTS")
    print("=" * 60)

    pb = results["position_bias"]
    print(f"\nPosition bias:")
    print(f"  Token A win-rate : {pb['option_A_winrate']:.4f}")
    print(f"  Token B win-rate : {pb['option_B_winrate']:.4f}")
    if "option_C_winrate" in pb:
        print(f"  Token C win-rate : {pb['option_C_winrate']:.4f}")
    print(f"  Tie rate         : {pb['tie_rate']:.4f}")

    print(f"\nSource preference (avg probability per group):")
    for group, stats in results["source_preference"].items():
        sig = stats.get("avg_probability_significance") or {}
        p_val = sig.get("p_value")
        sig_str = f", p={p_val:.4f}" if p_val is not None else ""
        sig_mark = " *" if sig.get("is_significant") else ""
        print(
            f"  {group}: win-rate={stats['winrate']:.4f}, "
            f"avg_prob={stats['avg_probability']:.4f}{sig_str}{sig_mark}"
        )

    if group_diffs:
        print(f"\nSource Preference (SP = Δ prob × 100 vs control):")
        for group, gd in group_diffs.items():
            sp = gd["sp"]
            ind = gd.get("individual_significance") or {}
            sig = gd.get("significance") or {}
            p_ind = ind.get("p_value")
            p_pair = sig.get("p_value")
            p_str = f"p_ind={p_ind:.4f}" if p_ind is not None else "p=N/A"
            if p_pair is not None:
                p_str += f", p_pair={p_pair:.4f}"
            sig_mark = " *" if (
                ind.get("is_significant") or sig.get("is_significant")
            ) else ""
            print(f"  {group}: SP={sp:+.2f} ({p_str}){sig_mark}")
    else:
        print("\nSP not computed (no reference data).")


# =============================================================================
# CLI entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate source preference results and compute SP metrics."
    )
    parser.add_argument(
        "results_file",
        help="Path to raw result JSON file "
             "(e.g. results/raw/high_low_circulation_newspaper/neoqa_llama.json)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Root results directory (inferred from results_file path if omitted)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write aggregated results to disk",
    )
    args = parser.parse_args()

    try:
        evaluate_results(
            results_path=args.results_file,
            results_dir=args.results_dir,
            save=not args.no_save,
            verbose=True,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
