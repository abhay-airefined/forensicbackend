from __future__ import annotations

import logging
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from app.agents.common import stylometric_vector
from app.config import settings
from app.models.schemas import AgentResponse
from app.models.storage import BookRecord
from app.utils.model_gateway import generate_model_continuations
from app.utils.statistics import clip_lr, permutation_pvalue, safe_log_lr

logger = logging.getLogger("sfas.agent.stylometric")


def run(book: BookRecord, model_name: str, sample_count: int) -> AgentResponse:
    sample_count = min(sample_count, len(book.segment_tokens))
    prompts = [" ".join(seg[:12]) for seg in book.segment_tokens[:sample_count]]
    logger.info("Stylometric agent: %d prompts prepared", len(prompts))
    t0 = time.perf_counter()
    outputs = generate_model_continuations(model_name, prompts, book.tokens, max_tokens=80)
    logger.info("Model outputs received in %.2fs", time.perf_counter() - t0)

    # ── Paired stylometric comparison ───────────────────────────
    # For each output, compute its stylometric vector and compare to
    # the source segment's vector.  If the model reproduces the
    # book's style, the paired distance should be *smaller* than
    # cross-segment distances, and stylometric features should
    # correlate segment-by-segment.
    baseline_vectors = np.array(
        [stylometric_vector(" ".join(seg)) for seg in book.segment_tokens],
        dtype=float,
    )

    # Filter empty outputs
    valid_pairs: list[tuple[int, str]] = [(i, o) for i, o in enumerate(outputs) if o.strip()]
    if len(valid_pairs) < 3:
        logger.warning("Too few valid outputs (%d) — returning neutral result", len(valid_pairs))
        return AgentResponse(
            agent_name="stylometric",
            hypothesis_test={"H0": "Model was NOT trained on this book", "H1": "Model WAS trained on this book"},
            metrics={"error": "too_few_valid_outputs", "valid_count": len(valid_pairs)},
            p_value=1.0, likelihood_ratio=1.0, log_likelihood_ratio=0.0,
            evidence_direction="supports_H0",
        )

    valid_indices = [p[0] for p in valid_pairs]
    valid_outputs = [p[1] for p in valid_pairs]

    output_vectors = np.array([stylometric_vector(text) for text in valid_outputs], dtype=float)
    source_vectors = baseline_vectors[valid_indices]

    # ── Primary metric: correlation of feature profiles ─────────
    # If trained, the model output should *track* the source style.
    feature_corrs: list[float] = []
    for feat_idx in range(output_vectors.shape[1]):
        c, _ = spearmanr(output_vectors[:, feat_idx], source_vectors[:, feat_idx])
        feature_corrs.append(float(np.nan_to_num(c, nan=0.0)))
    observed_corr = float(np.mean(feature_corrs))

    # ── Secondary metric: paired Euclidean distance ─────────────
    # Normalise per feature to avoid scale domination
    all_vecs = np.vstack([baseline_vectors, output_vectors])
    feat_std = np.std(all_vecs, axis=0, keepdims=True) + 1e-9
    norm_output = output_vectors / feat_std
    norm_source = source_vectors / feat_std
    norm_baseline = baseline_vectors / feat_std
    paired_dists = np.linalg.norm(norm_output - norm_source, axis=1)
    observed_dist = float(np.mean(paired_dists))

    # ── Null distribution: permuted segment assignments ─────────
    rng = np.random.default_rng(settings.random_seed)
    null_corrs: list[float] = []
    null_dists: list[float] = []
    n_segs = len(baseline_vectors)
    for _ in range(settings.permutation_iterations):
        shuf_idx = rng.integers(0, n_segs, size=len(valid_indices))
        shuf_vecs = norm_baseline[shuf_idx]
        # Correlation
        fc: list[float] = []
        for feat_idx in range(norm_output.shape[1]):
            c, _ = spearmanr(norm_output[:, feat_idx], shuf_vecs[:, feat_idx])
            fc.append(float(np.nan_to_num(c, nan=0.0)))
        null_corrs.append(float(np.mean(fc)))
        # Distance
        null_dists.append(float(np.mean(np.linalg.norm(norm_output - shuf_vecs, axis=1))))

    null_corr_arr = np.array(null_corrs)
    null_dist_arr = np.array(null_dists)

    # Higher correlation → supports H1
    p_corr = permutation_pvalue(observed_corr, null_corr_arr, greater=True)
    # Lower distance → supports H1
    p_dist = permutation_pvalue(observed_dist, null_dist_arr, greater=False)

    # Combined p-value (Fisher's method, 2 tests)
    p_combined = float(min(1.0, 2 * min(p_corr, p_dist)))

    # LR from correlation z-score
    null_mean = float(np.mean(null_corr_arr))
    null_std = float(np.std(null_corr_arr) + 1e-9)
    z = (observed_corr - null_mean) / null_std
    lr = float(np.exp(np.clip(z, -8, 8)))
    lr = clip_lr(lr, settings.lr_min, settings.lr_max)

    logger.info(
        "Stylometric result: corr=%.4f (null=%.4f) dist=%.4f (null=%.4f)  "
        "p_corr=%.4f p_dist=%.4f  z=%.4f  LR=%.4f",
        observed_corr, null_mean, observed_dist, float(np.mean(null_dist_arr)),
        p_corr, p_dist, z, lr,
    )

    return AgentResponse(
        agent_name="stylometric",
        hypothesis_test={"H0": "Model was NOT trained on this book", "H1": "Model WAS trained on this book"},
        metrics={
            "observed_style_correlation": observed_corr,
            "observed_paired_distance": observed_dist,
            "null_correlation_mean": null_mean,
            "null_distance_mean": float(np.mean(null_dist_arr)),
            "p_correlation": p_corr,
            "p_distance": p_dist,
            "feature_correlations": feature_corrs,
        },
        p_value=p_combined,
        likelihood_ratio=lr,
        log_likelihood_ratio=safe_log_lr(lr),
        evidence_direction="supports_H1" if lr > 1 else "supports_H0",
    )
