from __future__ import annotations

import logging
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.models.schemas import AgentResponse
from app.models.storage import BookRecord
from app.utils.model_gateway import generate_model_continuations
from app.utils.statistics import clip_lr, permutation_pvalue, safe_log_lr

logger = logging.getLogger("sfas.agent.semantic")


def run(book: BookRecord, model_name: str, sample_count: int) -> AgentResponse:
    sample_count = min(sample_count, len(book.segment_tokens))
    prompts = [" ".join(seg[:18]) for seg in book.segment_tokens[:sample_count]]
    logger.info("Semantic agent: %d prompts prepared", len(prompts))
    t0 = time.perf_counter()
    outputs = generate_model_continuations(model_name, prompts, book.tokens, max_tokens=90)
    logger.info("Model outputs received in %.2fs", time.perf_counter() - t0)

    # ── Filter empty / content-filtered outputs ─────────────────
    valid_pairs: list[tuple[int, str]] = [(i, o) for i, o in enumerate(outputs) if o.strip()]
    if len(valid_pairs) < 3:
        logger.warning("Too few valid outputs (%d) — returning neutral", len(valid_pairs))
        return AgentResponse(
            agent_name="semantic",
            hypothesis_test={"H0": "Model was NOT trained on this book", "H1": "Model WAS trained on this book"},
            metrics={"error": "too_few_valid_outputs", "valid_count": len(valid_pairs)},
            p_value=1.0, likelihood_ratio=1.0, log_likelihood_ratio=0.0,
            evidence_direction="supports_H0",
        )

    valid_indices = [p[0] for p in valid_pairs]
    valid_outputs = [p[1] for p in valid_pairs]

    # ── Build TF-IDF over all segments + outputs jointly ────────
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    corpus = list(book.segments) + valid_outputs
    mat = vectorizer.fit_transform(corpus)
    seg_mat = mat[: len(book.segments)]
    out_mat = mat[len(book.segments) :]

    # ── Paired similarity: each output ↔ its source segment ────
    # If trained on this book, the model should produce text that is
    # most similar to the *source* segment from which the prompt came.
    sim_matrix = cosine_similarity(out_mat, seg_mat)  # (n_valid, n_segments)

    paired_sims = np.array([
        sim_matrix[k, valid_indices[k]]
        for k in range(len(valid_indices))
    ])
    observed_paired = float(np.mean(paired_sims))

    # ── Rank metric: how often is the source the top-ranked seg? ─
    ranks = []
    for k, src_idx in enumerate(valid_indices):
        row = sim_matrix[k]
        rank = int(np.sum(row >= row[src_idx]))  # 1 = top
        ranks.append(rank)
    mean_rank = float(np.mean(ranks))
    top1_frac = float(np.mean([1 if r == 1 else 0 for r in ranks]))

    # ── Correlation across outputs: paired sim should correlate ─
    # Build a second feature: max sim to any segment (quality)
    max_sims = np.max(sim_matrix, axis=1)
    corr, _ = spearmanr(paired_sims, max_sims)
    observed_corr = float(np.nan_to_num(corr, nan=0.0))

    # ── Null: shuffle segment assignments ───────────────────────
    rng = np.random.default_rng(settings.random_seed)
    n_segs = len(book.segments)
    null_paired: list[float] = []
    null_top1: list[float] = []
    for _ in range(settings.permutation_iterations):
        shuf = rng.integers(0, n_segs, size=len(valid_indices))
        null_sims = np.array([sim_matrix[k, shuf[k]] for k in range(len(valid_indices))])
        null_paired.append(float(np.mean(null_sims)))
        # Top-1 for null
        null_ranks = [int(np.sum(sim_matrix[k] >= sim_matrix[k, shuf[k]])) for k in range(len(valid_indices))]
        null_top1.append(float(np.mean([1 if r == 1 else 0 for r in null_ranks])))

    null_arr = np.array(null_paired)
    null_top1_arr = np.array(null_top1)

    # Higher paired similarity → supports H1
    p_paired = permutation_pvalue(observed_paired, null_arr, greater=True)
    # Higher top-1 fraction → supports H1
    p_top1 = permutation_pvalue(top1_frac, null_top1_arr, greater=True)

    # Combined p-value (simple Bonferroni-style)
    p_combined = float(min(1.0, 2 * min(p_paired, p_top1)))

    # LR from z-score of paired similarity
    null_mean = float(np.mean(null_arr))
    null_std = float(np.std(null_arr) + 1e-9)
    z = (observed_paired - null_mean) / null_std
    lr = float(np.exp(np.clip(z, -8, 8)))
    lr = clip_lr(lr, settings.lr_min, settings.lr_max)

    logger.info(
        "Semantic result: paired_sim=%.4f (null=%.4f)  top1=%.2f%%  "
        "mean_rank=%.1f  p_paired=%.4f  p_top1=%.4f  z=%.4f  LR=%.4f",
        observed_paired, null_mean, top1_frac * 100, mean_rank,
        p_paired, p_top1, z, lr,
    )

    return AgentResponse(
        agent_name="semantic",
        hypothesis_test={"H0": "Model was NOT trained on this book", "H1": "Model WAS trained on this book"},
        metrics={
            "observed_paired_similarity": observed_paired,
            "null_paired_mean": null_mean,
            "top1_fraction": top1_frac,
            "mean_source_rank": mean_rank,
            "paired_max_correlation": observed_corr,
            "p_paired": p_paired,
            "p_top1": p_top1,
        },
        p_value=p_combined,
        likelihood_ratio=lr,
        log_likelihood_ratio=safe_log_lr(lr),
        evidence_direction="supports_H1" if lr > 1 else "supports_H0",
    )
