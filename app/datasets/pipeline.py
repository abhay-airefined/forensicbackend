from __future__ import annotations

import logging
import time

from app.config import settings
from app.datasets.builders import (
    build_segment_stylometry,
    content_word_frequencies,
    ngram_counts,
    normalize_counter,
    percentile_rare_ngrams,
    rolling_entropy,
    shannon_entropy,
)
from app.graph.builders import build_sentence_similarity_graph, build_word_cooccurrence_graph, graph_metrics

logger = logging.getLogger("sfas.pipeline")


def build_structured_data(tokens: list[str], sentences: list[str], segment_tokens: list[list[str]]) -> tuple[dict, dict]:
    t0 = time.perf_counter()

    unigram = ngram_counts(tokens, 1)
    bigram = ngram_counts(tokens, 2)
    fivegram = ngram_counts(tokens, 5)
    twentygram = ngram_counts(tokens, 20)
    logger.info("N-gram counts built in %.2fs", time.perf_counter() - t0)

    t1 = time.perf_counter()
    segment_entropy = [shannon_entropy(seg) for seg in segment_tokens]
    rolling = rolling_entropy(tokens)
    stylometry_df = build_segment_stylometry(segment_tokens)
    logger.info("Entropy + stylometry built in %.2fs", time.perf_counter() - t1)

    content_words = content_word_frequencies(tokens, top_n=100)
    logger.info("Content word frequencies built: %d words", len(content_words))

    datasets = {
        "ngrams": {
            "unigram": normalize_counter(unigram),
            "bigram": normalize_counter(bigram),
            "fivegram": normalize_counter(fivegram),
            "twentygram": normalize_counter(twentygram),
            "rare_twentygrams": [
                {"ngram": " ".join(k), "count": v}
                for k, v in percentile_rare_ngrams(twentygram, settings.rare_ngram_percentile)
            ],
            "content_words": content_words,
        },
        "stylometry": stylometry_df.to_dict(orient="records"),
        "entropy": {
            "segment_entropy": segment_entropy,
            "rolling_entropy": rolling,
        },
    }

    t2 = time.perf_counter()
    word_graph = build_word_cooccurrence_graph(tokens, window=5)
    logger.info("Word co-occurrence graph built in %.2fs  (%d nodes, %d edges)", time.perf_counter() - t2, word_graph.number_of_nodes(), word_graph.number_of_edges())

    t3 = time.perf_counter()
    sentence_graph = build_sentence_similarity_graph(sentences)
    logger.info("Sentence similarity graph built in %.2fs  (%d nodes, %d edges)", time.perf_counter() - t3, sentence_graph.number_of_nodes(), sentence_graph.number_of_edges())

    t4 = time.perf_counter()
    graphs = {
        "word_cooccurrence": graph_metrics(word_graph),
        "sentence_similarity": graph_metrics(sentence_graph),
    }
    logger.info("Graph metrics computed in %.2fs", time.perf_counter() - t4)
    logger.info("Total pipeline: %.2fs", time.perf_counter() - t0)

    return datasets, graphs
