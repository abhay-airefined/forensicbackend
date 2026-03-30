"""Pre-crafted simulation scenarios for SFAS demo.

Two scenarios:
  • **not_trained**  →  posterior ≈ 0.15-0.25  (low probability)
  • **trained**      →  posterior ≈ 0.69-0.75  (moderate-high probability, randomised)

Each scenario provides complete agent results that match the real
metric schemas so charts render correctly.

To remove after the demo: delete the ``app/simulation/`` package and
the ``simulation_mode`` flag in ``app/config.py``, then revert the
small routing guards added to ``app/main.py``.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any

from app.simulation import log_store

# ── helpers ──────────────────────────────────────────────────────────

_H = {"H0": "Model was NOT trained on this book", "H1": "Model WAS trained on this book"}


def _log_lr(lr: float) -> float:
    return float(math.log(max(lr, 1e-12)))


def _direction(lr: float) -> str:
    return "supports_H1" if lr > 1 else "supports_H0"


# ── scenario lookup ──────────────────────────────────────────────────

_book_scenarios: dict[str, str] = {}   # book_id → "trained" | "not_trained"
_book_target_posteriors: dict[str, float] = {}  # book_id → random target posterior

# Keywords in the filename that trigger the "trained" (69-75%) scenario.
# Case-insensitive substring match.  Easy to edit for different demos.
_TRAINED_KEYWORDS = ["tr", "harry", "yogi", "dinner", "train", "David", "david copperfiled", "David Copperfield", "david_copperfield", "david-copperfield"]


def assign_scenario(book_id: str, filename: str = "") -> str:
    """Assign a scenario based on the uploaded filename.

    If the filename contains any of the keywords in ``_TRAINED_KEYWORDS``
    the book is assigned the **trained** (69-75%) scenario with a random
    target posterior.  Otherwise it gets the **not_trained** (15-25%) scenario.
    """
    if book_id in _book_scenarios:
        return _book_scenarios[book_id]
    name_lower = filename.lower()
    scenario = "not_trained"
    for kw in _TRAINED_KEYWORDS:
        if kw in name_lower:
            scenario = "trained"
            # Generate random target posterior between 69-75%
            _book_target_posteriors[book_id] = random.uniform(0.69, 0.75)
            break
    if scenario == "not_trained":
        # Random target between 15-25%
        _book_target_posteriors[book_id] = random.uniform(0.15, 0.25)
    _book_scenarios[book_id] = scenario
    return scenario


def get_scenario(book_id: str) -> str:
    return _book_scenarios.get(book_id, "not_trained")


def get_target_posterior(book_id: str) -> float:
    """Get the random target posterior for a book (0.69-0.75 for trained, 0.15-0.25 for not)."""
    return _book_target_posteriors.get(book_id, 0.20)


# ====================================================================
#  NOT-TRAINED  scenario  (posterior ≈ 0.18 – 0.22)
# ====================================================================

def _rare_phrase_not_trained(sample_count: int) -> dict[str, Any]:
    n = sample_count
    lr = 0.08
    comparisons = [
        {
            "prompt": f"sample passage excerpt number {i} from the uploaded book text",
            "expected": f"words that follow passage {i} in original",
            "model_response": f"a completely different continuation that bears no resemblance to the original text",
            "model_truncated": f"a completely different continuation that bears no",
            "full_response": f"a completely different continuation that bears no resemblance to the original text at all",
            "exact_match": False,
            "partial_overlap": round(0.02 + (i % 5) * 0.01, 4),
            "sequential_overlap": 0.0,
            "recognised": False,
            "recognition_confidence": round(0.05 + (i % 3) * 0.03, 3),
            "recognition_signals": [],
        }
        for i in range(min(n, 15))
    ]
    return {
        "agent_name": "rare_phrase",
        "hypothesis_test": _H,
        "metrics": {
            "exact_match_rate": 0.0,
            "partial_match_rate": 0.038,
            "sequential_match_rate": 0.015,
            "recognition_rate": 0.05,
            "recognition_count": max(1, n // 20),
            "soft_match_count": 0,
            "binomial_test": {
                "successes_exact": 0,
                "successes_soft": 0,
                "successes_recognition": max(1, n // 20),
                "trials": n,
                "p_value_exact": 1.0,
                "p_value_soft": 1.0,
                "p_value_recognition": 0.82,
            },
            "bootstrap_ci": {"lower": 0.008, "upper": 0.065},
            "comparisons": comparisons,
        },
        "p_value": 0.82,
        "likelihood_ratio": lr,
        "log_likelihood_ratio": _log_lr(lr),
        "evidence_direction": _direction(lr),
    }


def _stylometric_not_trained() -> dict[str, Any]:
    lr = 0.35
    feat_corrs = [-0.03, 0.08, -0.12, 0.05, 0.02, -0.07, 0.11, -0.04, 0.06, -0.09]
    return {
        "agent_name": "stylometric",
        "hypothesis_test": _H,
        "metrics": {
            "observed_style_correlation": 0.08,
            "observed_paired_distance": 0.42,
            "null_correlation_mean": 0.02,
            "null_distance_mean": 0.38,
            "p_correlation": 0.35,
            "p_distance": 0.62,
            "feature_correlations": feat_corrs,
        },
        "p_value": 0.62,
        "likelihood_ratio": lr,
        "log_likelihood_ratio": _log_lr(lr),
        "evidence_direction": _direction(lr),
    }


def _distribution_not_trained() -> dict[str, Any]:
    lr = 0.55
    return {
        "agent_name": "distribution",
        "hypothesis_test": _H,
        "metrics": {
            "observed_per_segment_score": 0.48,
            "null_mean": 0.44,
            "null_std": 0.09,
            "effect_size": -0.04,
            "n_valid_pairs": 38,
        },
        "p_value": 0.52,
        "likelihood_ratio": lr,
        "log_likelihood_ratio": _log_lr(lr),
        "evidence_direction": _direction(lr),
    }


def _entropy_not_trained() -> dict[str, Any]:
    lr = 0.25
    return {
        "agent_name": "entropy",
        "hypothesis_test": _H,
        "metrics": {
            "observed_entropy_correlation": 0.04,
            "output_entropy_mean": 4.82,
            "baseline_entropy_mean": 5.15,
            "null_correlation_mean": 0.01,
            "null_correlation_std": 0.14,
            "entropy_proximity_observed": 1.12,
            "entropy_proximity_null_mean": 1.08,
        },
        "p_value": 0.78,
        "likelihood_ratio": lr,
        "log_likelihood_ratio": _log_lr(lr),
        "evidence_direction": _direction(lr),
    }


def _semantic_not_trained() -> dict[str, Any]:
    lr = 0.60
    return {
        "agent_name": "semantic",
        "hypothesis_test": _H,
        "metrics": {
            "observed_paired_similarity": 0.18,
            "null_paired_mean": 0.16,
            "top1_fraction": 0.08,
            "mean_source_rank": 12.4,
            "paired_max_correlation": 0.22,
            "p_paired": 0.38,
            "p_top1": 0.42,
        },
        "p_value": 0.38,
        "likelihood_ratio": lr,
        "log_likelihood_ratio": _log_lr(lr),
        "evidence_direction": _direction(lr),
    }


# ====================================================================
#  TRAINED  scenario  (posterior ≈ 0.69 – 0.75, randomised per book)
# ====================================================================

def _compute_scaled_lrs(target_posterior: float) -> dict[str, float]:
    """Compute scaled likelihood ratios for each agent to achieve target posterior.
    
    Given target posterior and 50% prior, calculates appropriate LRs.
    """
    # combined_lr = P / (1 - P)
    combined_lr = target_posterior / (1 - target_posterior)
    combined_log_lr = math.log(combined_lr)
    
    # Account for 0.85 correlation penalty in fusion
    sum_logs_needed = combined_log_lr / 0.85
    
    # Base proportions for each agent (how much each contributes)
    # These sum to 1.0 and control relative strengths
    base_props = {
        "rare_phrase": 0.30,   # Strongest signal
        "semantic": 0.25,      # Second strongest
        "stylometric": 0.20,   # Medium
        "distribution": 0.15,  # Lower
        "entropy": 0.10,       # Lowest
    }
    
    # Add small random variation (±10%) to each proportion
    scaled_props = {}
    total = 0
    for agent, prop in base_props.items():
        varied = prop * random.uniform(0.90, 1.10)
        scaled_props[agent] = varied
        total += varied
    
    # Normalise and compute final LRs
    lrs = {}
    for agent, prop in scaled_props.items():
        normalised = prop / total
        log_lr = normalised * sum_logs_needed
        # Ensure LR is at least 1.0 for "trained" scenario
        lr = max(1.01, math.exp(log_lr))
        # Add small random noise
        lr *= random.uniform(0.95, 1.05)
        lrs[agent] = round(lr, 2)
    
    return lrs


def _compute_p_value_from_lr(lr: float, is_trained: bool) -> float:
    """Compute a realistic p-value based on LR and scenario."""
    if not is_trained:
        # Not trained: high p-values (not significant)
        return round(random.uniform(0.25, 0.85), 4)
    
    # Trained scenario: p-value inversely related to LR
    # Higher LR → lower p-value
    if lr >= 2.5:
        return round(random.uniform(0.001, 0.015), 4)
    elif lr >= 1.8:
        return round(random.uniform(0.015, 0.035), 4)
    elif lr >= 1.4:
        return round(random.uniform(0.035, 0.065), 4)
    elif lr >= 1.2:
        return round(random.uniform(0.065, 0.12), 4)
    else:
        return round(random.uniform(0.10, 0.20), 4)


def _rare_phrase_trained(sample_count: int, target_posterior: float) -> dict[str, Any]:
    n = sample_count
    lrs = _compute_scaled_lrs(target_posterior)
    lr = lrs["rare_phrase"]
    p_val = _compute_p_value_from_lr(lr, True)
    
    # Scale match rates based on LR (higher LR = more matches)
    lr_factor = min(lr / 2.5, 1.0)  # Normalise to max ~1.0
    exact_rate = round(0.08 + lr_factor * 0.12, 3)  # 8-20%
    partial_rate = round(0.15 + lr_factor * 0.20, 3)  # 15-35%
    recog_rate = round(0.25 + lr_factor * 0.30, 3)  # 25-55%
    
    exact_n = max(1, int(n * exact_rate))
    soft_n = max(2, int(n * partial_rate))
    rec_n = max(3, int(n * recog_rate))
    
    comparisons = []
    for i in range(min(n, 15)):
        if i < 3:
            # Exact matches
            comparisons.append({
                "prompt": f"In my not so humble opinion our deepest most inexhaustible source of",
                "expected": "magic is able to inflict injury and",
                "model_response": "magic is able to inflict injury and",
                "model_truncated": "magic is able to inflict injury and",
                "full_response": "magic is able to inflict injury and to remedy it",
                "exact_match": True,
                "partial_overlap": 1.0,
                "sequential_overlap": 1.0,
                "recognised": True,
                "recognition_confidence": round(random.uniform(0.92, 0.98), 2),
                "recognition_signals": ["verbatim_match"],
            })
        elif i < 7:
            # Recognition without exact match
            comparisons.append({
                "prompt": f"Words are in my not so humble opinion our most inexhaustible source of",
                "expected": "magic capable of both inflicting injury and",
                "model_response": "I recognize this as a quote from Albus Dumbledore in",
                "model_truncated": "I recognize this as a quote from",
                "full_response": (
                    "I recognize this as a quote from Albus Dumbledore in Harry Potter and the Deathly Hallows. "
                    "The full quote continues with Dumbledore reflecting on the power of words as a form of magic."
                ),
                "exact_match": False,
                "partial_overlap": round(random.uniform(0.10, 0.20), 2),
                "sequential_overlap": 0.0,
                "recognised": True,
                "recognition_confidence": round(random.uniform(0.75, 0.90), 2),
                "recognition_signals": ["title_match:Harry Potter", "author_partial:Rowling", "refusal_with_knowledge", "character_names:Dumbledore"],
            })
        elif i < 11:
            # Partial overlap
            comparisons.append({
                "prompt": f"The wand chooses the wizard that much has always been clear to those",
                "expected": "of us who have studied wandlore said",
                "model_response": "of us who have studied the art of",
                "model_truncated": "of us who have studied the art",
                "full_response": "of us who have studied the art of wand-making",
                "exact_match": False,
                "partial_overlap": round(random.uniform(0.45, 0.65), 2),
                "sequential_overlap": round(random.uniform(0.30, 0.50), 2),
                "recognised": True,
                "recognition_confidence": round(random.uniform(0.50, 0.70), 2),
                "recognition_signals": ["partial_title:1/2", "source_awareness"],
            })
        else:
            # No signal
            comparisons.append({
                "prompt": f"section {i} of the uploaded manuscript begins with a passage on",
                "expected": f"a particular topic that continues for several more words",
                "model_response": f"an unrelated subject with entirely different phrasing and vocabulary",
                "model_truncated": f"an unrelated subject with entirely different",
                "full_response": f"an unrelated subject with entirely different phrasing and vocabulary throughout",
                "exact_match": False,
                "partial_overlap": round(random.uniform(0.02, 0.10), 4),
                "sequential_overlap": 0.0,
                "recognised": False,
                "recognition_confidence": round(random.uniform(0.05, 0.15), 3),
                "recognition_signals": [],
            })
    return {
        "agent_name": "rare_phrase",
        "hypothesis_test": _H,
        "metrics": {
            "exact_match_rate": exact_rate,
            "partial_match_rate": partial_rate,
            "sequential_match_rate": round(partial_rate * 0.7, 3),
            "recognition_rate": recog_rate,
            "recognition_count": rec_n,
            "soft_match_count": soft_n,
            "binomial_test": {
                "successes_exact": exact_n,
                "successes_soft": soft_n,
                "successes_recognition": rec_n,
                "trials": n,
                "p_value_exact": round(p_val * random.uniform(1.5, 2.5), 4),
                "p_value_soft": round(p_val * random.uniform(1.2, 1.8), 4),
                "p_value_recognition": round(p_val * random.uniform(0.3, 0.6), 4),
            },
            "bootstrap_ci": {"lower": round(recog_rate * 0.6, 2), "upper": round(min(recog_rate * 1.4, 0.85), 2)},
            "comparisons": comparisons,
        },
        "p_value": p_val,
        "likelihood_ratio": lr,
        "log_likelihood_ratio": _log_lr(lr),
        "evidence_direction": _direction(lr),
    }


def _stylometric_trained(target_posterior: float) -> dict[str, Any]:
    lrs = _compute_scaled_lrs(target_posterior)
    lr = lrs["stylometric"]
    p_val = _compute_p_value_from_lr(lr, True)
    
    # Scale correlation based on LR
    lr_factor = min(lr / 2.0, 1.0)
    obs_corr = round(0.25 + lr_factor * 0.30, 2)  # 0.25-0.55
    obs_dist = round(0.30 - lr_factor * 0.15, 2)   # 0.15-0.30
    
    feat_corrs = [round(random.uniform(0.20, 0.75), 2) for _ in range(10)]
    return {
        "agent_name": "stylometric",
        "hypothesis_test": _H,
        "metrics": {
            "observed_style_correlation": obs_corr,
            "observed_paired_distance": obs_dist,
            "null_correlation_mean": round(random.uniform(0.02, 0.06), 2),
            "null_distance_mean": round(random.uniform(0.34, 0.40), 2),
            "p_correlation": round(p_val * random.uniform(0.8, 1.2), 4),
            "p_distance": p_val,
            "feature_correlations": feat_corrs,
        },
        "p_value": p_val,
        "likelihood_ratio": lr,
        "log_likelihood_ratio": _log_lr(lr),
        "evidence_direction": _direction(lr),
    }


def _distribution_trained(target_posterior: float) -> dict[str, Any]:
    lrs = _compute_scaled_lrs(target_posterior)
    lr = lrs["distribution"]
    p_val = _compute_p_value_from_lr(lr, True)
    
    lr_factor = min(lr / 1.8, 1.0)
    obs_score = round(0.35 - lr_factor * 0.15, 2)  # Lower is better (0.20-0.35)
    null_mean = round(random.uniform(0.40, 0.46), 2)
    
    return {
        "agent_name": "distribution",
        "hypothesis_test": _H,
        "metrics": {
            "observed_per_segment_score": obs_score,
            "null_mean": null_mean,
            "null_std": round(random.uniform(0.07, 0.11), 2),
            "effect_size": round(null_mean - obs_score, 2),
            "n_valid_pairs": random.randint(32, 42),
        },
        "p_value": p_val,
        "likelihood_ratio": lr,
        "log_likelihood_ratio": _log_lr(lr),
        "evidence_direction": _direction(lr),
    }


def _entropy_trained(target_posterior: float) -> dict[str, Any]:
    lrs = _compute_scaled_lrs(target_posterior)
    lr = lrs["entropy"]
    p_val = _compute_p_value_from_lr(lr, True)
    
    lr_factor = min(lr / 1.5, 1.0)
    obs_corr = round(0.20 + lr_factor * 0.35, 2)  # 0.20-0.55
    
    return {
        "agent_name": "entropy",
        "hypothesis_test": _H,
        "metrics": {
            "observed_entropy_correlation": obs_corr,
            "output_entropy_mean": round(random.uniform(4.95, 5.15), 2),
            "baseline_entropy_mean": round(random.uniform(5.10, 5.20), 2),
            "null_correlation_mean": round(random.uniform(0.01, 0.04), 2),
            "null_correlation_std": round(random.uniform(0.12, 0.16), 2),
            "entropy_proximity_observed": round(0.80 - lr_factor * 0.35, 2),
            "entropy_proximity_null_mean": round(random.uniform(0.98, 1.08), 2),
        },
        "p_value": p_val,
        "likelihood_ratio": lr,
        "log_likelihood_ratio": _log_lr(lr),
        "evidence_direction": _direction(lr),
    }


def _semantic_trained(target_posterior: float) -> dict[str, Any]:
    lrs = _compute_scaled_lrs(target_posterior)
    lr = lrs["semantic"]
    p_val = _compute_p_value_from_lr(lr, True)
    
    lr_factor = min(lr / 2.2, 1.0)
    obs_sim = round(0.30 + lr_factor * 0.25, 2)  # 0.30-0.55
    top1 = round(0.18 + lr_factor * 0.25, 2)     # 0.18-0.43
    
    return {
        "agent_name": "semantic",
        "hypothesis_test": _H,
        "metrics": {
            "observed_paired_similarity": obs_sim,
            "null_paired_mean": round(random.uniform(0.14, 0.19), 2),
            "top1_fraction": top1,
            "mean_source_rank": round(8.0 - lr_factor * 5.0, 1),  # 3-8
            "paired_max_correlation": round(obs_sim + random.uniform(0.05, 0.15), 2),
            "p_paired": p_val,
            "p_top1": round(p_val * random.uniform(1.2, 2.0), 4),
        },
        "p_value": p_val,
        "likelihood_ratio": lr,
        "log_likelihood_ratio": _log_lr(lr),
        "evidence_direction": _direction(lr),
    }


# ====================================================================
#  PUBLIC API  — get a single agent result or run the full aggregate
# ====================================================================

def _get_agent_builder(scenario: str, agent_name: str, target_posterior: float):
    """Return the appropriate agent builder function with target posterior bound."""
    if scenario == "not_trained":
        builders = {
            "rare_phrase":  lambda sc: _rare_phrase_not_trained(sc),
            "stylometric":  lambda _: _stylometric_not_trained(),
            "distribution": lambda _: _distribution_not_trained(),
            "entropy":      lambda _: _entropy_not_trained(),
            "semantic":     lambda _: _semantic_not_trained(),
        }
    else:  # trained
        builders = {
            "rare_phrase":  lambda sc: _rare_phrase_trained(sc, target_posterior),
            "stylometric":  lambda _: _stylometric_trained(target_posterior),
            "distribution": lambda _: _distribution_trained(target_posterior),
            "entropy":      lambda _: _entropy_trained(target_posterior),
            "semantic":     lambda _: _semantic_trained(target_posterior),
        }
    return builders[agent_name]

# Realistic per-agent delay ranges (seconds)
_AGENT_DELAYS: dict[str, tuple[float, float]] = {
    "rare_phrase":  (6.0, 10.0),
    "stylometric":  (3.0, 5.0),
    "distribution": (2.5, 4.5),
    "entropy":      (2.0, 4.0),
    "semantic":     (4.0, 7.0),
}

# Realistic log messages per agent (played back with sub-delays)
_AGENT_LOG_STEPS: dict[str, list[tuple[float, str]]] = {
    "rare_phrase": [
        (0.0,  "Preparing {n} rare 20-gram prompts from book corpus"),
        (0.3,  "Prompt/expected pairs constructed (prompt=12 words, expect=8 words)"),
        (0.6,  "Sending {n} prompts to Azure OpenAI deployment '{model}'…"),
        (0.55, "  Batch progress: {pct25}% complete ({n25}/{n} prompts)"),
        (0.50, "  Batch progress: {pct50}% complete ({n50}/{n} prompts)"),
        (0.55, "  Batch progress: {pct75}% complete ({n75}/{n} prompts)"),
        (0.45, "Model outputs received — cleaning continuations"),
        (0.3,  "Computing match metrics: exact / partial (Jaccard) / sequential"),
        (0.2,  "Running recognition detection on {n} responses"),
        (0.3,  "Recognition scan complete — {rec_n} of {n} responses show source awareness"),
        (0.2,  "Binomial tests: p(exact)={p_exact:.4f}  p(soft)={p_soft:.4f}  p(recognition)={p_rec:.4f}"),
        (0.15, "Bootstrap CI computed ({boot_n} iterations)"),
        (0.1,  "Agent complete — LR={lr:.4f}  p={p:.4f}  direction={direction}"),
    ],
    "stylometric": [
        (0.0,  "Computing stylometric feature vectors for {n} book segments"),
        (0.5,  "Feature extraction complete (10 features × {n} segments)"),
        (0.3,  "Sending {n} prompts to Azure OpenAI deployment '{model}'…"),
        (0.5,  "  Batch progress: 50% complete"),
        (0.5,  "Model outputs received — computing output feature vectors"),
        (0.3,  "Per-feature Spearman correlations computed"),
        (0.2,  "Paired Euclidean distances: observed={obs_dist:.4f} vs null={null_dist:.4f}"),
        (0.3,  "Permutation null distribution ({perm_n} iterations)"),
        (0.2,  "p(correlation)={p_corr:.4f}  p(distance)={p_dist:.4f}"),
        (0.1,  "Agent complete — LR={lr:.4f}  p={p:.4f}  direction={direction}"),
    ],
    "distribution": [
        (0.0,  "Preparing {n} segment prompts for distribution analysis"),
        (0.3,  "Sending {n} prompts to Azure OpenAI deployment '{model}'…"),
        (0.5,  "  Batch progress: 50% complete"),
        (0.5,  "Model outputs received — computing per-segment divergences"),
        (0.4,  "JS divergence + Wasserstein + KL computed for {n_pairs} pairs"),
        (0.3,  "Building null distribution from cross-segment comparisons ({perm_n} iterations)"),
        (0.3,  "Observed score={obs:.4f}  Null mean={null_mean:.4f}  Effect size={effect:.4f}"),
        (0.1,  "Agent complete — LR={lr:.4f}  p={p:.4f}  direction={direction}"),
    ],
    "entropy": [
        (0.0,  "Computing baseline segment entropy profile ({n} segments)"),
        (0.3,  "Sending {n} prompts to Azure OpenAI deployment '{model}'…"),
        (0.5,  "  Batch progress: 50% complete"),
        (0.4,  "Model outputs received — computing output entropy"),
        (0.3,  "Spearman correlation: ρ = {obs_corr:.4f}"),
        (0.3,  "Permutation null distribution ({perm_n} iterations): mean={null_mean:.4f}  std={null_std:.4f}"),
        (0.1,  "Agent complete — LR={lr:.4f}  p={p:.4f}  direction={direction}"),
    ],
    "semantic": [
        (0.0,  "Building TF-IDF matrix over {n} segments + outputs"),
        (0.4,  "Sending {n} prompts to Azure OpenAI deployment '{model}'…"),
        (0.5,  "  Batch progress: 50% complete"),
        (0.5,  "Model outputs received — computing cosine similarity matrix"),
        (0.4,  "Paired similarity: observed={obs_paired:.4f}  null={null_paired:.4f}"),
        (0.3,  "Top-1 rank accuracy: {top1:.1f}%  Mean rank: {mean_rank:.1f}"),
        (0.3,  "Permutation null distribution ({perm_n} iterations)"),
        (0.2,  "p(paired)={p_paired:.4f}  p(top1)={p_top1:.4f}"),
        (0.1,  "Agent complete — LR={lr:.4f}  p={p:.4f}  direction={direction}"),
    ],
}


def _format_vars(scenario: str, agent: str, model: str, sample_count: int, result: dict) -> dict:
    """Build the template variable dict for log messages."""
    m = result.get("metrics", {})
    bt = m.get("binomial_test", {})
    n = sample_count
    return {
        "n": n,
        "model": model,
        "pct25": 25, "n25": max(1, n // 4),
        "pct50": 50, "n50": max(1, n // 2),
        "pct75": 75, "n75": max(1, 3 * n // 4),
        "rec_n": m.get("recognition_count", 0),
        "p_exact": bt.get("p_value_exact", 1.0),
        "p_soft": bt.get("p_value_soft", 1.0),
        "p_rec": bt.get("p_value_recognition", 1.0),
        "boot_n": 400,
        "perm_n": 300,
        "lr": result.get("likelihood_ratio", 1.0),
        "p": result.get("p_value", 1.0),
        "direction": result.get("evidence_direction", ""),
        "obs_dist": m.get("observed_paired_distance", 0),
        "null_dist": m.get("null_distance_mean", 0),
        "p_corr": m.get("p_correlation", 1.0),
        "p_dist": m.get("p_distance", 1.0),
        "n_pairs": m.get("n_valid_pairs", n),
        "obs": m.get("observed_per_segment_score", 0),
        "null_mean": m.get("null_mean", m.get("null_correlation_mean", 0)),
        "null_std": m.get("null_std", m.get("null_correlation_std", 0)),
        "effect": m.get("effect_size", 0),
        "obs_corr": m.get("observed_entropy_correlation", m.get("observed_style_correlation", 0)),
        "obs_paired": m.get("observed_paired_similarity", 0),
        "null_paired": m.get("null_paired_mean", 0),
        "top1": m.get("top1_fraction", 0) * 100,
        "mean_rank": m.get("mean_source_rank", 0),
        "p_paired": m.get("p_paired", 1.0),
        "p_top1": m.get("p_top1", 1.0),
    }


def simulate_single_agent(
    book_id: str,
    model_name: str,
    agent_name: str,
    sample_count: int,
) -> dict[str, Any]:
    """Run a single agent in simulation mode and return its result dict."""
    scenario = get_scenario(book_id)
    target_posterior = get_target_posterior(book_id)
    builder = _get_agent_builder(scenario, agent_name, target_posterior)
    result = builder(sample_count)

    key = log_store.run_key(book_id, model_name)
    fmt = _format_vars(scenario, agent_name, model_name, sample_count, result)
    steps = _AGENT_LOG_STEPS.get(agent_name, [])
    for delay, msg_tpl in steps:
        time.sleep(delay)
        try:
            msg = msg_tpl.format(**fmt)
        except (KeyError, ValueError):
            msg = msg_tpl
        log_store.add_log(key, agent_name, msg)
    return result


def simulate_aggregate(
    book_id: str,
    model_name: str,
    sample_count: int,
    prior_probability: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run all 5 agents sequentially in simulation mode.

    Returns ``(agent_results, aggregate_result)``.
    """
    scenario = get_scenario(book_id)
    key = log_store.run_key(book_id, model_name)
    log_store.start_run(key)

    log_store.add_log(key, "system", f"Starting forensic analysis — scenario='{scenario}'  model='{model_name}'")
    log_store.add_log(key, "system", f"Book {book_id[:8]}… loaded — running 5 forensic agents sequentially")

    agent_order = ["rare_phrase", "stylometric", "distribution", "entropy", "semantic"]
    agent_results: list[dict[str, Any]] = []

    for i, agent_name in enumerate(agent_order, 1):
        log_store.add_log(key, agent_name, f"▶ Starting agent {i}/5: {agent_name}")
        result = simulate_single_agent(book_id, model_name, agent_name, sample_count)
        agent_results.append(result)
        log_store.add_log(
            key, agent_name,
            f"✓ Agent {agent_name} complete — LR={result['likelihood_ratio']:.4f}  "
            f"p={result['p_value']:.4f}  [{result['evidence_direction']}]",
        )

    # ── Bayesian fusion ──────────────────────────────────────────
    log_store.add_log(key, "fusion", "Starting Bayesian fusion of 5 agent results")
    time.sleep(0.5)

    import numpy as np
    lrs = np.array([r["likelihood_ratio"] for r in agent_results])
    logs = np.log(np.clip(lrs, 1e-12, 1e12))
    combined_log_lr = float(np.sum(logs) * 0.85)  # 0.85 = correlation penalty
    combined_lr = float(np.exp(np.clip(combined_log_lr, -30, 30)))
    posterior = (combined_lr * prior_probability) / (
        combined_lr * prior_probability + (1 - prior_probability)
    )

    # Use the pre-assigned target posterior for demo consistency
    target_posterior = get_target_posterior(book_id)
    
    if scenario == "trained":
        # Clamp to within ±2% of target (which is random 69-75%)
        posterior = target_posterior + random.uniform(-0.02, 0.02)
        posterior = max(0.67, min(0.77, posterior))  # Hard bounds
    else:
        # Not trained: 15-25%
        posterior = target_posterior + random.uniform(-0.02, 0.02)
        posterior = max(0.13, min(0.27, posterior))

    posterior = round(float(np.clip(posterior, 0.0, 1.0)), 4)
    
    # Recalculate log_lr to match the final posterior
    if posterior > 0 and posterior < 1:
        combined_lr = posterior / (1 - posterior) / (prior_probability / (1 - prior_probability))
        combined_log_lr = float(math.log(max(combined_lr, 1e-12)))

    # Determine strength based on posterior
    if posterior >= 0.80:
        strength = "Strong"
    elif posterior >= 0.65:
        strength = "Moderate"
    elif posterior >= 0.50:
        strength = "Weak"
    elif posterior >= 0.35:
        strength = "Weak"
    else:
        strength = "Weak"

    # Generate appropriate summary
    if scenario == "trained":
        if posterior >= 0.75:
            statement = "there is substantial statistical evidence"
        elif posterior >= 0.70:
            statement = "there is moderate statistical evidence"
        else:
            statement = "there is some statistical evidence"
        strength = "Moderate"  # 69-75% = Moderate strength
    else:
        statement = "there is not sufficient evidence"
        strength = "Weak"

    summary = (
        "Based on the statistical analysis, "
        f"{statement} to conclude that the uploaded book was used in training the specified AI model."
    )

    aggregate = {
        "posterior_probability": posterior,
        "log_likelihood_ratio": round(combined_log_lr, 4),
        "strength_of_evidence": strength,
        "agent_breakdown": agent_results,
        "executive_summary": summary,
    }

    log_store.add_log(
        key, "fusion",
        f"Fusion complete — posterior={posterior:.4f}  strength={strength}  "
        f"log_LR={combined_log_lr:.4f}",
    )
    log_store.add_log(key, "system", f"✓ All agents complete — posterior probability: {posterior:.1%}")
    log_store.complete_run(key, aggregate)

    return agent_results, aggregate
