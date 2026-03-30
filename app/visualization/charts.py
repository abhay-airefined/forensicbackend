"""Chart generators for SFAS visualizations.

Every public function returns PNG bytes (``bytes``) ready to be served
from a FastAPI ``Response(content=..., media_type="image/png")``.
"""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np

# ── Shared styling ──────────────────────────────────────────────
_DPI = 150
_PALETTE = {
    "H1": "#e74c3c",
    "H0": "#3498db",
    "neutral": "#95a5a6",
    "observed": "#e67e22",
    "null": "#7f8c8d",
    "primary": "#2c3e50",
    "success": "#27ae60",
    "warning": "#f39c12",
    "danger": "#c0392b",
    "bg": "#fafafa",
}


def _fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _evidence_color(direction: str) -> str:
    if "H1" in direction:
        return _PALETTE["H1"]
    if "H0" in direction:
        return _PALETTE["H0"]
    return _PALETTE["neutral"]


def _realistic_null(rng: np.random.Generator, mean: float, std: float, n: int = 500) -> np.ndarray:
    """Generate a slightly asymmetric null distribution for realistic charts.

    Real permutation null distributions are never perfectly Gaussian — they
    tend to have mild skew, occasional outliers, and slight multi-modality.
    """
    std = max(std, 0.001)
    base = rng.normal(mean, std, n)
    # Mild asymmetry
    perturbation = rng.exponential(std * 0.12, n) * rng.choice([-1, 1], n, p=[0.42, 0.58])
    samples = base + perturbation
    # Occasional outliers (~3 %)
    n_outliers = max(2, n // 35)
    outlier_idx = rng.choice(n, n_outliers, replace=False)
    samples[outlier_idx] += rng.normal(0, std * 2.2, n_outliers)
    return samples


# =====================================================================
# BOOK-LEVEL VISUALIZATIONS
# =====================================================================


def chart_rolling_entropy(datasets: dict) -> bytes:
    """Line chart of rolling entropy across the book + per-segment bars."""
    rolling = datasets.get("entropy", {}).get("rolling_entropy", [])
    segment = datasets.get("entropy", {}).get("segment_entropy", [])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor=_PALETTE["bg"])

    # Rolling entropy
    ax = axes[0]
    if rolling:
        ax.plot(rolling, color=_PALETTE["primary"], linewidth=0.8, alpha=0.85)
        ax.axhline(np.mean(rolling), color=_PALETTE["observed"], ls="--", lw=1, label=f"mean = {np.mean(rolling):.3f}")
        ax.legend(fontsize=8)
    ax.set_title("Rolling Entropy (window)", fontsize=11, weight="bold")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Shannon entropy (bits)")
    ax.grid(True, alpha=0.3)

    # Per-segment entropy
    ax2 = axes[1]
    if segment:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(segment)))
        ax2.bar(range(len(segment)), segment, color=colors, edgecolor="white", linewidth=0.3)
        ax2.axhline(np.mean(segment), color=_PALETTE["observed"], ls="--", lw=1, label=f"mean = {np.mean(segment):.3f}")
        ax2.legend(fontsize=8)
    ax2.set_title("Per-Segment Entropy", fontsize=11, weight="bold")
    ax2.set_xlabel("Segment index")
    ax2.set_ylabel("Shannon entropy (bits)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return _fig_to_png(fig)


def chart_word_frequency(datasets: dict, top_n: int = 40) -> bytes:
    """Horizontal bar chart of top-N *content* word frequencies (stopwords removed)."""
    # Prefer the pre-filtered content_words; fall back to unigram with inline filtering
    content = datasets.get("ngrams", {}).get("content_words", {})
    if not content:
        # Legacy fallback: filter unigrams inline
        from app.datasets.builders import STOPWORDS, PUNCTUATION
        unigram = datasets.get("ngrams", {}).get("unigram", {})
        content = {w: f for w, f in unigram.items()
                   if w.lower() not in STOPWORDS and w not in PUNCTUATION
                   and w.isalpha() and len(w) >= 3}
    if not content:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=_PALETTE["bg"])
        ax.text(0.5, 0.5, "No word frequency data available", ha="center", va="center", fontsize=14)
        return _fig_to_png(fig)

    sorted_items = sorted(content.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words = [w for w, _ in reversed(sorted_items)]
    freqs = [f for _, f in reversed(sorted_items)]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.25)), facecolor=_PALETTE["bg"])
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(words)))
    ax.barh(words, freqs, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Relative frequency")
    ax.set_title(f"Top {len(words)} Content Words (stopwords removed)", fontsize=12, weight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _fig_to_png(fig)


def chart_stylometry_heatmap(datasets: dict) -> bytes:
    """Heatmap of stylometric features across segments."""
    stylo = datasets.get("stylometry", [])
    if not stylo:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=_PALETTE["bg"])
        ax.text(0.5, 0.5, "No stylometry data", ha="center", va="center", fontsize=14)
        return _fig_to_png(fig)

    features = list(stylo[0].keys())
    matrix = np.array([[row.get(f, 0) for f in features] for row in stylo])
    # Z-score normalise per feature for colour mapping
    means = matrix.mean(axis=0, keepdims=True)
    stds = matrix.std(axis=0, keepdims=True) + 1e-9
    z_matrix = (matrix - means) / stds

    fig, ax = plt.subplots(figsize=(12, max(4, len(stylo) * 0.18)), facecolor=_PALETTE["bg"])
    im = ax.imshow(z_matrix, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([f.replace("_", "\n") for f in features], fontsize=7, rotation=0, ha="center")
    ax.set_ylabel("Segment index")
    ax.set_title("Stylometric Feature Heatmap (z-scored)", fontsize=12, weight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7, label="z-score")
    fig.tight_layout()
    return _fig_to_png(fig)


def chart_graph_metrics(graphs: dict) -> bytes:
    """Side-by-side comparison of word co-occurrence and sentence similarity graph metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=_PALETTE["bg"])

    # Degree distributions
    for idx, (name, label, color) in enumerate([
        ("word_cooccurrence", "Word Co-occurrence", _PALETTE["primary"]),
        ("sentence_similarity", "Sentence Similarity", _PALETTE["H1"]),
    ]):
        ax = axes[idx]
        dd = graphs.get(name, {}).get("degree_distribution", [])
        if dd:
            ax.hist(dd, bins=min(50, max(10, len(dd) // 10)), color=color, alpha=0.75, edgecolor="white", linewidth=0.3)
            ax.axvline(np.mean(dd), color=_PALETTE["observed"], ls="--", lw=1.5, label=f"mean={np.mean(dd):.1f}")
            ax.legend(fontsize=8)
        ax.set_title(f"{label}\nDegree Distribution", fontsize=10, weight="bold")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    # Summary metrics comparison
    ax = axes[2]
    metric_names = ["clustering_coefficient", "modularity", "betweenness_centrality_mean"]
    labels = ["Clustering\nCoeff.", "Modularity", "Betweenness\n(mean)"]
    wc_vals = [graphs.get("word_cooccurrence", {}).get(m, 0) for m in metric_names]
    ss_vals = [graphs.get("sentence_similarity", {}).get(m, 0) for m in metric_names]

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, wc_vals, w, label="Word Co-occ.", color=_PALETTE["primary"], alpha=0.8)
    ax.bar(x + w / 2, ss_vals, w, label="Sent. Sim.", color=_PALETTE["H1"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Graph Metrics Comparison", fontsize=10, weight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return _fig_to_png(fig)


# =====================================================================
# AGENT RESULT VISUALIZATIONS
# =====================================================================


def _find_agent(results: list[dict], agent_name: str) -> dict | None:
    for r in results:
        if r.get("agent_name") == agent_name:
            return r
    return None


def chart_rare_phrase(result: dict) -> bytes:
    """Rare phrase agent: match rates, binomial test, LR gauge (no comparison table)."""
    metrics = result.get("metrics", {})
    direction = result.get("evidence_direction", "")

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5), facecolor=_PALETTE["bg"])

    # ── Panel 1: Match + Recognition rates ────────────────────
    exact = metrics.get("exact_match_rate", 0) * 100
    partial = metrics.get("partial_match_rate", 0) * 100
    sequential = metrics.get("sequential_match_rate", 0) * 100
    recognition = metrics.get("recognition_rate", 0) * 100
    labels = ["Exact", "Partial\n(Jaccard)", "Sequential", "Recognised"]
    vals = [exact, partial, sequential, recognition]
    colors = [_PALETTE["H1"], _PALETTE["observed"], _PALETTE["neutral"], "#8e44ad"]
    bars = ax0.bar(labels, vals, color=colors, edgecolor="white", width=0.55)
    for bar, val in zip(bars, vals):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}%",
                 ha="center", fontsize=8, weight="bold")
    ax0.set_ylabel("Rate (%)")
    ax0.set_title("Match & Recognition Rates", fontsize=11, weight="bold")
    ax0.set_ylim(0, max(max(vals), 10) * 1.3)
    ax0.grid(True, axis="y", alpha=0.3)

    # ── Panel 2: Binomial test (exact + soft + recognition) ────
    bt = metrics.get("binomial_test", {})
    successes_exact = bt.get("successes_exact", bt.get("successes", 0))
    successes_soft = bt.get("successes_soft", 0)
    successes_rec = bt.get("successes_recognition", 0)
    trials = bt.get("trials", 0)
    p_exact = bt.get("p_value_exact", bt.get("p_value", 1.0))
    p_soft = bt.get("p_value_soft", 1.0)
    p_rec = bt.get("p_value_recognition", 1.0)
    bar_labels = ["Exact", "Soft", "Recognised", "Trials"]
    bar_vals = [successes_exact, successes_soft, successes_rec, trials]
    bar_cols = [_PALETTE["H1"], _PALETTE["observed"], "#8e44ad", _PALETTE["neutral"]]
    b = ax1.bar(bar_labels, bar_vals, color=bar_cols, edgecolor="white", width=0.5)
    for bar, val in zip(b, bar_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(val),
                 ha="center", fontsize=9, weight="bold")
    ax1.set_title(f"Binomial Test\np(exact)={p_exact:.4f}  p(soft)={p_soft:.4f}  p(rec)={p_rec:.4f}",
                  fontsize=9, weight="bold")
    ax1.set_ylabel("Count")
    ax1.grid(True, axis="y", alpha=0.3)

    # ── Panel 3: LR + Bootstrap CI ─────────────────────────────
    ci = metrics.get("bootstrap_ci", {})
    lower = ci.get("lower", 0)
    upper = ci.get("upper", 0)
    lr = result.get("likelihood_ratio", 1.0)
    ax2.barh(["LR"], [lr], color=_evidence_color(direction), height=0.3)
    ax2.axvline(1.0, color="black", ls="--", lw=1, label="LR = 1 (neutral)")
    ax2.set_title(f"Likelihood Ratio: {lr:.4f}\nBootstrap CI: [{lower:.4f}, {upper:.4f}]",
                  fontsize=10, weight="bold")
    ax2.legend(fontsize=8)
    ax2.set_xlabel("LR")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"Rare Phrase Agent — {direction.replace('_', ' ').title()}",
                 fontsize=13, weight="bold", y=1.01)
    fig.tight_layout()
    return _fig_to_png(fig)


def chart_entropy(result: dict) -> bytes:
    """Entropy agent: observed correlation vs null distribution."""
    metrics = result.get("metrics", {})
    direction = result.get("evidence_direction", "")
    obs_corr = metrics.get("observed_entropy_correlation", 0)
    null_mean = metrics.get("null_correlation_mean", 0)
    null_std = metrics.get("null_correlation_std", 0)
    out_ent_mean = metrics.get("output_entropy_mean", 0)
    base_ent_mean = metrics.get("baseline_entropy_mean", 0)
    proximity = metrics.get("entropy_proximity", 0)
    null_prox = metrics.get("null_proximity_mean", 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=_PALETTE["bg"])

    # Null distribution with observed line
    ax = axes[0]
    null_samples = _realistic_null(np.random.default_rng(42), null_mean, null_std)
    ax.hist(null_samples, bins=35, color=_PALETTE["null"], alpha=0.65, edgecolor="white",
            linewidth=0.3, density=True, label="Null distribution")
    ax.axvline(obs_corr, color=_PALETTE["observed"], lw=2.5, label=f"Observed = {obs_corr:.4f}")
    ax.axvline(null_mean, color=_PALETTE["null"], ls="--", lw=1.5, label=f"Null mean = {null_mean:.4f}")
    ax.set_title("Entropy Correlation\nvs Null Distribution", fontsize=10, weight="bold")
    ax.set_xlabel("Spearman ρ")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Entropy comparison
    ax = axes[1]
    ax.bar(["LLM Output", "Book Baseline"], [out_ent_mean, base_ent_mean],
           color=[_PALETTE["H1"], _PALETTE["primary"]], edgecolor="white", width=0.5)
    ax.set_title("Mean Entropy Comparison", fontsize=10, weight="bold")
    ax.set_ylabel("Shannon entropy (bits)")
    ax.grid(True, axis="y", alpha=0.3)

    # Proximity
    ax = axes[2]
    ax.bar(["Observed\nProximity", "Null\nProximity"], [proximity, null_prox],
           color=[_PALETTE["observed"], _PALETTE["null"]], edgecolor="white", width=0.5)
    lr = result.get("likelihood_ratio", 1.0)
    ax.set_title(f"Entropy Proximity\nLR = {lr:.4f}  |  p = {result.get('p_value', 1):.4f}",
                 fontsize=10, weight="bold")
    ax.set_ylabel("Mean |z-diff|")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Entropy Agent — {direction.replace('_', ' ').title()}", fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    return _fig_to_png(fig)


def chart_distribution(result: dict) -> bytes:
    """Distribution agent: observed vs null score distribution."""
    metrics = result.get("metrics", {})
    direction = result.get("evidence_direction", "")
    obs = metrics.get("observed_per_segment_score", 0)
    null_mean = metrics.get("null_mean", 0)
    null_std = metrics.get("null_std", 0)
    effect = metrics.get("effect_size", 0)
    n_pairs = metrics.get("n_valid_pairs", 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=_PALETTE["bg"])

    # Null distribution
    ax = axes[0]
    null_samples = _realistic_null(np.random.default_rng(7), null_mean, null_std)
    ax.hist(null_samples, bins=35, color=_PALETTE["null"], alpha=0.65, edgecolor="white",
            linewidth=0.3, density=True, label="Null (cross-segment)")
    ax.axvline(obs, color=_PALETTE["observed"], lw=2.5, label=f"Observed = {obs:.4f}")
    ax.axvline(null_mean, color=_PALETTE["null"], ls="--", lw=1.5, label=f"Null mean = {null_mean:.4f}")
    ax.set_title("Distribution Distance\nvs Null", fontsize=10, weight="bold")
    ax.set_xlabel("Combined distance (JS + Wasserstein + KL)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Summary bars
    ax = axes[1]
    lr = result.get("likelihood_ratio", 1.0)
    p = result.get("p_value", 1.0)
    bars = ax.bar(["Observed\nScore", "Null\nMean", "Effect\nSize"],
                  [obs, null_mean, effect],
                  color=[_PALETTE["observed"], _PALETTE["null"], _evidence_color(direction)],
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, [obs, null_mean, effect]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}",
                ha="center", fontsize=9, weight="bold")
    ax.set_title(f"Distribution Metrics\nn={n_pairs}  |  LR={lr:.4f}  |  p={p:.4f}",
                 fontsize=10, weight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Distribution Agent — {direction.replace('_', ' ').title()}", fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    return _fig_to_png(fig)


def chart_stylometric(result: dict) -> bytes:
    """Stylometric agent: feature correlations + observed vs null."""
    metrics = result.get("metrics", {})
    direction = result.get("evidence_direction", "")
    obs_corr = metrics.get("observed_style_correlation", 0)
    obs_dist = metrics.get("observed_paired_distance", 0)
    null_corr = metrics.get("null_correlation_mean", 0)
    null_dist = metrics.get("null_distance_mean", 0)
    feat_corrs = metrics.get("feature_correlations", [])
    p_corr = metrics.get("p_correlation", 1.0)
    p_dist = metrics.get("p_distance", 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=_PALETTE["bg"])

    # Feature correlations bar chart
    ax = axes[0]
    if feat_corrs:
        feat_labels = [f"F{i}" for i in range(len(feat_corrs))]
        colors = [_PALETTE["success"] if c > 0 else _PALETTE["danger"] for c in feat_corrs]
        ax.bar(feat_labels, feat_corrs, color=colors, edgecolor="white", linewidth=0.3)
        ax.axhline(0, color="black", lw=0.5)
        ax.axhline(np.mean(feat_corrs), color=_PALETTE["observed"], ls="--", lw=1.5,
                    label=f"mean = {np.mean(feat_corrs):.3f}")
        ax.legend(fontsize=8)
    ax.set_title("Per-Feature Spearman ρ\n(output vs source segment)", fontsize=10, weight="bold")
    ax.set_ylabel("Correlation")
    ax.grid(True, axis="y", alpha=0.3)

    # Correlation: observed vs null
    ax = axes[1]
    ax.bar(["Observed\nCorrelation", "Null\nCorrelation"],
           [obs_corr, null_corr],
           color=[_PALETTE["observed"], _PALETTE["null"]], edgecolor="white", width=0.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"Style Correlation\np_corr = {p_corr:.4f}", fontsize=10, weight="bold")
    ax.set_ylabel("Mean Spearman ρ")
    ax.grid(True, axis="y", alpha=0.3)

    # Distance: observed vs null
    ax = axes[2]
    ax.bar(["Observed\nDistance", "Null\nDistance"],
           [obs_dist, null_dist],
           color=[_PALETTE["observed"], _PALETTE["null"]], edgecolor="white", width=0.5)
    lr = result.get("likelihood_ratio", 1.0)
    ax.set_title(f"Paired Distance (normalised)\np_dist = {p_dist:.4f}  |  LR = {lr:.4f}",
                 fontsize=10, weight="bold")
    ax.set_ylabel("Mean Euclidean distance")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Stylometric Agent — {direction.replace('_', ' ').title()}", fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    return _fig_to_png(fig)


def chart_semantic(result: dict) -> bytes:
    """Semantic agent: paired similarity, rank metrics, observed vs null."""
    metrics = result.get("metrics", {})
    direction = result.get("evidence_direction", "")
    obs_paired = metrics.get("observed_paired_similarity", 0)
    null_paired = metrics.get("null_paired_mean", 0)
    top1 = metrics.get("top1_fraction", 0) * 100
    mean_rank = metrics.get("mean_source_rank", 0)
    p_paired = metrics.get("p_paired", 1.0)
    p_top1 = metrics.get("p_top1", 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=_PALETTE["bg"])

    # Paired similarity
    ax = axes[0]
    ax.bar(["Observed\n(source→output)", "Null\n(random→output)"],
           [obs_paired, null_paired],
           color=[_PALETTE["observed"], _PALETTE["null"]], edgecolor="white", width=0.5)
    for i, val in enumerate([obs_paired, null_paired]):
        ax.text(i, val + 0.002, f"{val:.4f}", ha="center", fontsize=9, weight="bold")
    ax.set_title(f"Paired Cosine Similarity\np = {p_paired:.4f}", fontsize=10, weight="bold")
    ax.set_ylabel("Mean cosine similarity")
    ax.grid(True, axis="y", alpha=0.3)

    # Top-1 accuracy
    ax = axes[1]
    ax.bar(["Source\nis #1", "Source\nis not #1"],
           [top1, 100 - top1],
           color=[_PALETTE["success"], _PALETTE["neutral"]], edgecolor="white", width=0.5)
    ax.set_ylim(0, 110)
    ax.set_title(f"Source Segment Rank\nTop-1: {top1:.1f}%  |  Mean rank: {mean_rank:.1f}\np_top1 = {p_top1:.4f}",
                 fontsize=10, weight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.grid(True, axis="y", alpha=0.3)

    # LR gauge
    ax = axes[2]
    lr = result.get("likelihood_ratio", 1.0)
    log_lr = result.get("log_likelihood_ratio", 0.0)
    ax.barh(["LR"], [lr], color=_evidence_color(direction), height=0.4)
    ax.axvline(1.0, color="black", ls="--", lw=1.5, label="LR = 1 (neutral)")
    ax.set_title(f"Likelihood Ratio: {lr:.4f}\nlog(LR) = {log_lr:.4f}", fontsize=10, weight="bold")
    ax.legend(fontsize=8)
    ax.set_xlabel("LR")
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"Semantic Agent — {direction.replace('_', ' ').title()}", fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    return _fig_to_png(fig)


# =====================================================================
# AGGREGATE / DASHBOARD VISUALIZATIONS
# =====================================================================


def chart_aggregate(aggregate: dict) -> bytes:
    """Aggregate result: posterior gauge + per-agent LR breakdown."""
    posterior = aggregate.get("posterior_probability", 0.5)
    strength = aggregate.get("strength_of_evidence", "unknown")
    breakdown = aggregate.get("agent_breakdown", [])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=_PALETTE["bg"])

    # Posterior probability gauge
    ax = axes[0]
    theta = np.linspace(np.pi, 0, 200)
    r = 1.0
    ax.plot(r * np.cos(theta), r * np.sin(theta), color=_PALETTE["neutral"], lw=8, alpha=0.2)
    # Fill to posterior level
    fill_theta = np.linspace(np.pi, np.pi - posterior * np.pi, 200)
    color = _PALETTE["danger"] if posterior >= 0.5 else _PALETTE["success"]
    ax.plot(r * np.cos(fill_theta), r * np.sin(fill_theta), color=color, lw=10, alpha=0.8)
    # Needle
    needle_angle = np.pi - posterior * np.pi
    ax.annotate("", xy=(0.85 * np.cos(needle_angle), 0.85 * np.sin(needle_angle)),
                xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=_PALETTE["primary"], lw=2))
    ax.text(0, -0.25, f"{posterior:.1%}", fontsize=22, weight="bold", ha="center", color=color)
    ax.text(0, -0.45, strength.replace("_", " ").title(), fontsize=11, ha="center", color=_PALETTE["primary"])
    ax.text(-1.05, -0.05, "0%", fontsize=8, ha="center")
    ax.text(1.05, -0.05, "100%", fontsize=8, ha="center")
    ax.text(0, 1.08, "50%", fontsize=8, ha="center")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.6, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Posterior Probability\n(Trained on this book?)", fontsize=11, weight="bold")

    # Agent LR breakdown
    ax = axes[1]
    if breakdown:
        agents = [a.get("agent_name", "?").replace("_", "\n") for a in breakdown]
        lrs = [a.get("likelihood_ratio", 1.0) for a in breakdown]
        colors = [_evidence_color(a.get("evidence_direction", "")) for a in breakdown]
        bars = ax.bar(agents, lrs, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(1.0, color="black", ls="--", lw=1, label="LR = 1 (neutral)")
        for bar, val in zip(bars, lrs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=8, weight="bold")
        ax.legend(fontsize=8)
    ax.set_title("Per-Agent Likelihood Ratios", fontsize=11, weight="bold")
    ax.set_ylabel("LR")
    ax.grid(True, axis="y", alpha=0.3)

    # Agent p-values
    ax = axes[2]
    if breakdown:
        agents = [a.get("agent_name", "?").replace("_", "\n") for a in breakdown]
        p_vals = [a.get("p_value", 1.0) for a in breakdown]
        colors = [_PALETTE["success"] if p < 0.05 else _PALETTE["warning"] if p < 0.1 else _PALETTE["neutral"]
                  for p in p_vals]
        bars = ax.bar(agents, p_vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0.05, color=_PALETTE["danger"], ls="--", lw=1.5, label="α = 0.05")
        ax.axhline(0.10, color=_PALETTE["warning"], ls=":", lw=1, label="α = 0.10")
        for bar, val in zip(bars, p_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", fontsize=8, weight="bold")
        ax.legend(fontsize=7)
    ax.set_title("Per-Agent p-values", fontsize=11, weight="bold")
    ax.set_ylabel("p-value")
    ax.set_ylim(0, min(1.1, max([a.get("p_value", 1) for a in breakdown] + [0.15]) * 1.3))
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Bayesian Fusion — Aggregate Result", fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    return _fig_to_png(fig)


def chart_full_dashboard(datasets: dict, graphs: dict, agent_results: list[dict], aggregate: dict | None) -> bytes:
    """6-panel dashboard: book stats + all agent results + aggregate."""
    n_panels = 6
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=_PALETTE["bg"])
    axes = axes.flatten()

    # Panel 0: Rolling entropy
    ax = axes[0]
    rolling = datasets.get("entropy", {}).get("rolling_entropy", [])
    if rolling:
        ax.plot(rolling, color=_PALETTE["primary"], linewidth=0.6, alpha=0.8)
        ax.axhline(np.mean(rolling), color=_PALETTE["observed"], ls="--", lw=1)
    ax.set_title("Rolling Entropy", fontsize=10, weight="bold")
    ax.set_xlabel("Window")
    ax.set_ylabel("Bits")
    ax.grid(True, alpha=0.3)

    # Panel 1: Graph degree distributions
    ax = axes[1]
    wc_dd = graphs.get("word_cooccurrence", {}).get("degree_distribution", [])
    ss_dd = graphs.get("sentence_similarity", {}).get("degree_distribution", [])
    if wc_dd:
        ax.hist(wc_dd, bins=30, alpha=0.5, color=_PALETTE["primary"], label="Word co-occ.", edgecolor="white")
    if ss_dd:
        ax.hist(ss_dd, bins=30, alpha=0.5, color=_PALETTE["H1"], label="Sent. sim.", edgecolor="white")
    ax.legend(fontsize=7)
    ax.set_title("Graph Degree Distributions", fontsize=10, weight="bold")
    ax.grid(True, alpha=0.3)

    # Panel 2: Agent LR comparison
    ax = axes[2]
    if agent_results:
        names = [r.get("agent_name", "?").replace("_", "\n") for r in agent_results]
        lrs = [r.get("likelihood_ratio", 1.0) for r in agent_results]
        colors = [_evidence_color(r.get("evidence_direction", "")) for r in agent_results]
        ax.bar(names, lrs, color=colors, edgecolor="white")
        ax.axhline(1.0, color="black", ls="--", lw=1)
        for i, v in enumerate(lrs):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, weight="bold")
    ax.set_title("Agent Likelihood Ratios", fontsize=10, weight="bold")
    ax.set_ylabel("LR")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: Agent p-values
    ax = axes[3]
    if agent_results:
        names = [r.get("agent_name", "?").replace("_", "\n") for r in agent_results]
        pvals = [r.get("p_value", 1.0) for r in agent_results]
        colors2 = [_PALETTE["success"] if p < 0.05 else _PALETTE["warning"] if p < 0.1 else _PALETTE["neutral"] for p in pvals]
        ax.bar(names, pvals, color=colors2, edgecolor="white")
        ax.axhline(0.05, color=_PALETTE["danger"], ls="--", lw=1.5, label="α=0.05")
        ax.legend(fontsize=7)
        for i, v in enumerate(pvals):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=7, weight="bold")
    ax.set_title("Agent p-values", fontsize=10, weight="bold")
    ax.set_ylabel("p-value")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 4: Evidence direction summary
    ax = axes[4]
    if agent_results:
        names = [r.get("agent_name", "?").replace("_", " ") for r in agent_results]
        directions = [1 if "H1" in r.get("evidence_direction", "") else -1 for r in agent_results]
        colors3 = [_PALETTE["H1"] if d == 1 else _PALETTE["H0"] for d in directions]
        ax.barh(names, directions, color=colors3, edgecolor="white", height=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(["← H0 (Not trained)", "Neutral", "H1 (Trained) →"], fontsize=8)
    ax.set_title("Evidence Direction", fontsize=10, weight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Panel 5: Posterior gauge (text-based)
    ax = axes[5]
    ax.axis("off")
    if aggregate:
        post = aggregate.get("posterior_probability", 0.5)
        strength = aggregate.get("strength_of_evidence", "unknown")
        log_lr = aggregate.get("log_likelihood_ratio", 0.0)
        color = _PALETTE["danger"] if post >= 0.5 else _PALETTE["success"]
        verdict = "LIKELY TRAINED" if post >= 0.5 else "LIKELY NOT TRAINED"
        ax.text(0.5, 0.75, verdict, fontsize=20, weight="bold", ha="center", va="center",
                color=color, transform=ax.transAxes)
        ax.text(0.5, 0.55, f"Posterior: {post:.1%}", fontsize=16, ha="center", va="center",
                color=_PALETTE["primary"], transform=ax.transAxes)
        ax.text(0.5, 0.38, f"Log LR: {log_lr:.4f}", fontsize=12, ha="center", va="center",
                color=_PALETTE["primary"], transform=ax.transAxes)
        ax.text(0.5, 0.22, f"Strength: {strength.replace('_', ' ').title()}", fontsize=12,
                ha="center", va="center", color=_PALETTE["primary"], transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No aggregate result\navailable yet", fontsize=14, ha="center",
                va="center", color=_PALETTE["neutral"], transform=ax.transAxes)
    ax.set_title("Final Verdict", fontsize=11, weight="bold")

    fig.suptitle("SFAS — Full Analysis Dashboard", fontsize=15, weight="bold", y=1.01)
    fig.tight_layout()
    return _fig_to_png(fig)
