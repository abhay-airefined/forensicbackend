from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

FUNCTION_WORDS = {
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "it", "as", "with", "on", "was", "at", "by", "an", "be", "this", "from",
}
PUNCTUATION = {".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", '"', "`"}

# Extended stopwords for word-frequency filtering
STOPWORDS = {
    # articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "some", "any", "no", "every", "each",
    # pronouns
    "i", "me", "we", "us", "you", "he", "him", "she", "it", "they", "them",
    "who", "whom", "what", "which", "myself", "himself", "herself", "itself",
    "ourselves", "themselves", "yourself", "yourselves",
    # prepositions
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "over", "against", "along", "across", "around",
    "among", "upon", "within", "without", "toward", "towards", "onto",
    # conjunctions
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "only", "if", "then", "than", "when", "while", "although",
    "because", "since", "unless", "whether", "though",
    # auxiliaries / be / have / do
    "is", "am", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    "must", "need", "dare", "ought",
    # common adverbs / misc
    "very", "too", "also", "just", "now", "here", "there", "where",
    "how", "all", "more", "most", "other", "such", "as", "like",
    "well", "much", "even", "still", "already", "always", "never",
    "often", "ever", "quite", "rather", "really",
    # misc function words
    "up", "out", "down", "off", "away", "back", "own", "same",
    "one", "two", "first", "new", "old", "said",
    "know", "get", "go", "come", "make", "see", "take",
    "way", "thing", "things", "man", "time",
}


def content_word_frequencies(tokens: list[str], top_n: int = 100) -> dict[str, float]:
    """Return normalised frequencies of the top-N content words (no stopwords/punct)."""
    counts: Counter[str] = Counter()
    for t in tokens:
        low = t.lower()
        if low in STOPWORDS or low in PUNCTUATION or not t.isalpha() or len(t) < 3:
            continue
        counts[low] += 1
    total = sum(counts.values()) or 1
    return {w: c / total for w, c in counts.most_common(top_n)}


def ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _safe_entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = np.array([v / total for v in counter.values()], dtype=float)
    return float(-(probs * np.log2(np.clip(probs, 1e-12, 1))).sum())


def shannon_entropy(tokens: list[str]) -> float:
    return _safe_entropy(Counter(tokens))


def rolling_entropy(tokens: list[str], window: int = 100) -> list[float]:
    if not tokens:
        return []
    if len(tokens) <= window:
        return [shannon_entropy(tokens)]
    out = []
    for i in range(0, len(tokens) - window + 1, max(1, window // 5)):
        out.append(shannon_entropy(tokens[i : i + window]))
    return out


def stylometric_features(tokens: list[str], sentences: list[str]) -> dict[str, float]:
    words = [t for t in tokens if t.isalnum() or "'" in t]
    sent_lengths = [len(s.split()) for s in sentences if s.strip()]
    word_lengths = [len(w) for w in words]
    counts = Counter(tokens)
    total = max(1, len(tokens))
    function_freq = sum(counts[w] for w in FUNCTION_WORDS) / total
    punct_freq = sum(counts[p] for p in PUNCTUATION) / total
    ttr = len(set(words)) / max(1, len(words))
    return {
        "sentence_length_mean": float(np.mean(sent_lengths)) if sent_lengths else 0.0,
        "sentence_length_std": float(np.std(sent_lengths)) if sent_lengths else 0.0,
        "word_length_mean": float(np.mean(word_lengths)) if word_lengths else 0.0,
        "word_length_std": float(np.std(word_lengths)) if word_lengths else 0.0,
        "type_token_ratio": float(ttr),
        "function_word_frequency": float(function_freq),
        "punctuation_frequency": float(punct_freq),
    }


def build_segment_stylometry(segment_tokens: list[list[str]]) -> pd.DataFrame:
    rows = []
    for seg in segment_tokens:
        seg_text = " ".join(seg)
        sentences = [s.strip() for s in seg_text.split(".") if s.strip()]
        rows.append(stylometric_features(seg, sentences))
    return pd.DataFrame(rows)


def normalize_counter(counter: Counter[tuple[str, ...]] | Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {" ".join(k) if isinstance(k, tuple) else k: v / total for k, v in counter.items()}


def percentile_rare_ngrams(counter: Counter[tuple[str, ...]], percentile: float) -> list[tuple[tuple[str, ...], int]]:
    if not counter:
        return []
    values = np.array(list(counter.values()), dtype=float)
    threshold = np.percentile(values, percentile)
    return [(ng, c) for ng, c in counter.items() if c <= threshold]
