from __future__ import annotations

import logging
import time
from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist, jensenshannon

from app.config import settings
from app.models.schemas import AgentResponse
from app.models.storage import BookRecord
from app.preprocessing.text_pipeline import word_tokenize
from app.utils.model_gateway import generate_model_continuations
from app.utils.statistics import clip_lr, permutation_pvalue, safe_log_lr

logger = logging.getLogger("sfas.agent.distribution")


# ── Helpers ─────────────────────────────────────────────────────────


def _js_from_counters(c1: Counter, c2: Counter) -> float:
    """JS distance between two token-frequency Counters.

    Builds a pair-local vocabulary so output words that are absent from
    the book vocabulary are still accounted for.
    """
    keys = set(c1) | set(c2)
    if not keys:
        return 0.0
    total1 = sum(c1.values()) or 1
    total2 = sum(c2.values()) or 1
    ordered = sorted(keys)
    p = np.array([c1.get(w, 0) / total1 for w in ordered], dtype=np.float64)
    q = np.array([c2.get(w, 0) / total2 for w in ordered], dtype=np.float64)
    d = jensenshannon(p, q)
    return 0.0 if np.isnan(d) else float(d)


def _vocab_overlap(c1: Counter, c2: Counter) -> float:
    """Jaccard coefficient of the two vocabularies."""
    s1, s2 = set(c1), set(c2)
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0


def _cosine_sim(c1: Counter, c2: Counter) -> float:
    """Cosine similarity on raw frequency vectors."""
    keys = set(c1) | set(c2)
    if not keys:
        return 0.0
    ordered = sorted(keys)
    a = np.array([c1.get(w, 0) for w in ordered], dtype=np.float64)
    b = np.array([c2.get(w, 0) for w in ordered], dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def _neutral_response(reason: str, detail: int = 0) -> AgentResponse:
    """Return a neutral/inconclusive response for edge cases."""
    return AgentResponse(
        agent_name="distribution",
        hypothesis_test={
            "H0": "Model was NOT trained on this book",
            "H1": "Model WAS trained on this book",
        },
        metrics={"error": reason, "count": detail},
        p_value=1.0,
        likelihood_ratio=1.0,
        log_likelihood_ratio=0.0,
        evidence_direction="supports_H0",
    )


# ── Main entry point ────────────────────────────────────────────────


def run(book: BookRecord, model_name: str, sample_count: int) -> AgentResponse:
    t_start = time.perf_counter()
    n_segs = len(book.segment_tokens)
    sample_count = min(sample_count, n_segs)

    if sample_count < 3:
        logger.warning("Too few segments (%d) for distribution analysis", n_segs)
        return _neutral_response("too_few_segments", n_segs)

    # ── 1. Prompt the model ─────────────────────────────────────
    # Use 20 leading tokens per segment for richer context.
    prompts = [" ".join(seg[:20]) for seg in book.segment_tokens[:sample_count]]

    logger.info("Distribution agent: %d prompts prepared", len(prompts))
    t0 = time.perf_counter()
    outputs = generate_model_continuations(
        model_name, prompts, book.tokens, max_tokens=80,
    )
    logger.info("Model outputs received in %.2fs", time.perf_counter() - t0)

    # ── 2. Build Counters for segments and outputs ──────────────
    seg_counters: list[Counter] = [Counter(seg) for seg in book.segment_tokens]

    valid_indices: list[int] = []
    out_counters: list[Counter] = []
    out_lengths: list[int] = []

    for i, raw in enumerate(outputs):
        text = (raw or "").strip()
        if not text:
            continue
        toks = word_tokenize(text)
        if len(toks) < 3:
            continue
        out_counters.append(Counter(toks))
        out_lengths.append(len(toks))
        valid_indices.append(i)

    n_valid = len(valid_indices)
    if n_valid < 2:
        logger.warning("Only %d valid output(s) — returning neutral", n_valid)
        return _neutral_response("too_few_valid_outputs", n_valid)

    avg_out_len = int(np.mean(out_lengths))

    # ── 3. Observed metrics (output ↔ source segment) ──────────
    t1 = time.perf_counter()
    obs_js: list[float] = []
    obs_overlap: list[float] = []
    obs_cosine: list[float] = []

    for k, seg_idx in enumerate(valid_indices):
        sc, oc = seg_counters[seg_idx], out_counters[k]
        obs_js.append(_js_from_counters(sc, oc))
        obs_overlap.append(_vocab_overlap(sc, oc))
        obs_cosine.append(_cosine_sim(sc, oc))

    obs_js_mean = float(np.mean(obs_js))
    obs_overlap_mean = float(np.mean(obs_overlap))
    obs_cosine_mean = float(np.mean(obs_cosine))

    logger.info(
        "Observed: JS=%.4f  overlap=%.4f  cosine=%.4f  (%d pairs, %.2fs)",
        obs_js_mean, obs_overlap_mean, obs_cosine_mean,
        n_valid, time.perf_counter() - t1,
    )

    # ── 4. Pre-compute pairwise JS matrix for the null ─────────
    #
    # Build normalised frequency vectors over the book-level
    # vocabulary for every segment (full-length) and every segment
    # truncated to the typical output length.  A single
    # scipy.spatial.distance.cdist call then produces the complete
    # inter-segment JS distance matrix — the null loop becomes
    # pure random indexing, which is orders of magnitude faster
    # than the previous nested-loop approach.
    t2 = time.perf_counter()

    all_words: set[str] = set()
    for seg in book.segment_tokens:
        all_words.update(seg)
    vocab_list = sorted(all_words)
    vocab_idx = {w: i for i, w in enumerate(vocab_list)}
    V = len(vocab_list)

    seg_freq = np.zeros((n_segs, V), dtype=np.float64)
    trunc_freq = np.zeros((n_segs, V), dtype=np.float64)

    for i, seg in enumerate(book.segment_tokens):
        for w in seg:
            seg_freq[i, vocab_idx[w]] += 1
        for w in seg[:avg_out_len]:
            trunc_freq[i, vocab_idx[w]] += 1
        s_total = seg_freq[i].sum()
        t_total = trunc_freq[i].sum()
        if s_total > 0:
            seg_freq[i] /= s_total
        if t_total > 0:
            trunc_freq[i] /= t_total

    # Full pairwise JS distance: full-segment × truncated-segment
    pairwise = cdist(seg_freq, trunc_freq, metric="jensenshannon")
    np.nan_to_num(pairwise, copy=False, nan=0.0)

    logger.info(
        "Pairwise JS (%dx%d, vocab=%d) in %.2fs",
        n_segs, n_segs, V, time.perf_counter() - t2,
    )

    # ── 5. Null distribution via random indexing ────────────────
    t3 = time.perf_counter()
    rng = np.random.default_rng(settings.random_seed)
    n_perms = settings.permutation_iterations
    null_means = np.empty(n_perms, dtype=np.float64)

    for p_idx in range(n_perms):
        a = rng.integers(0, n_segs, size=n_valid)
        b = rng.integers(0, n_segs, size=n_valid)
        null_means[p_idx] = pairwise[a, b].mean()

    null_mean = float(null_means.mean())
    null_std = float(null_means.std() + 1e-9)

    logger.info(
        "Null: mean=%.4f  std=%.4f  (%d perms, %.2fs)",
        null_mean, null_std, n_perms, time.perf_counter() - t3,
    )

    # ── 6. Statistical inference ────────────────────────────────
    # Lower JS distance ⇒ model output closer to source ⇒ H1
    p_value = permutation_pvalue(obs_js_mean, null_means, greater=False)

    z = (null_mean - obs_js_mean) / null_std
    effect = null_mean - obs_js_mean

    lr = float(np.exp(np.clip(z, -10, 10)))
    lr = clip_lr(lr, settings.lr_min, settings.lr_max)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Distribution done: JS_obs=%.4f JS_null=%.4f z=%.3f "
        "LR=%.4f p=%.4f  [%.2fs]",
        obs_js_mean, null_mean, z, lr, p_value, elapsed,
    )

    return AgentResponse(
        agent_name="distribution",
        hypothesis_test={
            "H0": "Model was NOT trained on this book",
            "H1": "Model WAS trained on this book",
        },
        metrics={
            "observed_per_segment_score": obs_js_mean,
            "observed_js_distance": obs_js_mean,
            "observed_vocab_overlap": obs_overlap_mean,
            "observed_cosine_similarity": obs_cosine_mean,
            "null_mean": null_mean,
            "null_std": null_std,
            "effect_size": effect,
            "z_score": z,
            "n_valid_pairs": n_valid,
            "avg_output_tokens": avg_out_len,
        },
        p_value=p_value,
        likelihood_ratio=lr,
        log_likelihood_ratio=safe_log_lr(lr),
        evidence_direction="supports_H1" if lr > 1 else "supports_H0",
    )
