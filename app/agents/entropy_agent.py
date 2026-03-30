from __future__ import annotations

import logging
import time

import numpy as np
from scipy.stats import spearmanr

from app.config import settings
from app.datasets.builders import shannon_entropy
from app.models.schemas import AgentResponse
from app.models.storage import BookRecord
from app.preprocessing.text_pipeline import word_tokenize
from app.utils.model_gateway import generate_model_continuations
from app.utils.statistics import clip_lr, permutation_pvalue, safe_log_lr

logger = logging.getLogger("sfas.agent.entropy")


def run(book: BookRecord, model_name: str, sample_count: int) -> AgentResponse:
    sample_count = min(sample_count, len(book.segment_tokens))
    prompts = [" ".join(seg[:10]) for seg in book.segment_tokens[:sample_count]]

    t0 = time.perf_counter()
    outputs = generate_model_continuations(model_name, prompts, book.tokens, max_tokens=60)
    logger.info("Model outputs received in %.2fs", time.perf_counter() - t0)

    # ── Filter empty outputs ────────────────────────────────────
    valid_pairs = [(i, o) for i, o in enumerate(outputs) if o.strip()]
    if len(valid_pairs) < 3:
        logger.warning(
            "Too few valid outputs (%d) — returning neutral result",
            len(valid_pairs),
        )
        return AgentResponse(
            agent_name="entropy",
            hypothesis_test={
                "H0": "Model was NOT trained on this book",
                "H1": "Model WAS trained on this book",
            },
            metrics={"error": "too_few_valid_outputs", "valid_count": len(valid_pairs)},
            p_value=1.0,
            likelihood_ratio=1.0,
            log_likelihood_ratio=0.0,
            evidence_direction="supports_H0",
        )

    valid_indices = [p[0] for p in valid_pairs]
    valid_outputs = [p[1] for p in valid_pairs]

    output_entropy = np.array(
        [shannon_entropy(word_tokenize(o)) for o in valid_outputs], dtype=float
    )
    baseline = np.array(book.datasets["entropy"]["segment_entropy"], dtype=float)
    paired_baseline = baseline[valid_indices]

    logger.info(
        "Entropy stats: output_mean=%.4f  baseline_mean=%.4f  paired_mean=%.4f  n=%d",
        float(np.mean(output_entropy)),
        float(np.mean(baseline)),
        float(np.mean(paired_baseline)),
        len(valid_indices),
    )

    # ── Primary metric: Entropy profile correlation ─────────────
    #
    # If the model was trained on this book, its per-output entropy
    # should *correlate* with the entropy of the source segment it
    # was prompted from.  Under H0 (not trained) the correlation
    # should be near zero.
    observed_corr, _ = spearmanr(output_entropy, paired_baseline)
    observed_corr = float(np.nan_to_num(observed_corr, nan=0.0))

    # Null distribution: permute the segment entropy assignment
    rng = np.random.default_rng(settings.random_seed)
    null_corrs = []
    for _ in range(settings.permutation_iterations):
        shuffled = rng.permutation(baseline)[: len(output_entropy)]
        c, _ = spearmanr(output_entropy, shuffled)
        null_corrs.append(float(np.nan_to_num(c, nan=0.0)))
    null_arr = np.array(null_corrs)

    # Higher correlation → more evidence for H1
    p_value = permutation_pvalue(observed_corr, null_arr, greater=True)

    # ── Secondary metric: Normalised entropy proximity ──────────
    out_z = (output_entropy - np.mean(output_entropy)) / (
        np.std(output_entropy) + 1e-9
    )
    base_z = (paired_baseline - np.mean(baseline)) / (np.std(baseline) + 1e-9)
    observed_proximity = float(np.mean(np.abs(out_z - base_z)))

    null_proximities = []
    for _ in range(settings.bootstrap_iterations):
        shuf = rng.permutation(baseline)[: len(output_entropy)]
        shuf_z = (shuf - np.mean(baseline)) / (np.std(baseline) + 1e-9)
        null_proximities.append(float(np.mean(np.abs(out_z - shuf_z))))
    null_prox_arr = np.array(null_proximities)

    # ── Likelihood ratio from correlation ───────────────────────
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr) + 1e-9)
    z_score = (observed_corr - null_mean) / null_std
    lr = float(np.exp(np.clip(z_score, -8, 8)))
    lr = clip_lr(lr, settings.lr_min, settings.lr_max)

    logger.info(
        "Entropy result: corr=%.4f  null_mean=%.4f  z=%.4f  LR=%.4f  p=%.4f "
        "proximity=%.4f (null=%.4f)",
        observed_corr,
        null_mean,
        z_score,
        lr,
        p_value,
        observed_proximity,
        float(np.mean(null_prox_arr)),
    )

    return AgentResponse(
        agent_name="entropy",
        hypothesis_test={
            "H0": "Model was NOT trained on this book",
            "H1": "Model WAS trained on this book",
        },
        metrics={
            "observed_entropy_correlation": observed_corr,
            "output_entropy_mean": float(np.mean(output_entropy)),
            "baseline_entropy_mean": float(np.mean(baseline)),
            "null_correlation_mean": null_mean,
            "null_correlation_std": null_std,
            "entropy_proximity_observed": observed_proximity,
            "entropy_proximity_null_mean": float(np.mean(null_prox_arr)),
        },
        p_value=p_value,
        likelihood_ratio=lr,
        log_likelihood_ratio=safe_log_lr(lr),
        evidence_direction="supports_H1" if lr > 1 else "supports_H0",
    )
