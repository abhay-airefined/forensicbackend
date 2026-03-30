from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter

import numpy as np
from scipy.stats import binomtest

from app.config import settings
from app.models.schemas import AgentResponse
from app.models.storage import BookRecord
from app.utils.model_gateway import generate_model_continuations_rare_phrase
from app.utils.statistics import beta_binomial_lr, bootstrap_ci, clip_lr, safe_log_lr

logger = logging.getLogger("sfas.agent.rare_phrase")

_CONT_TAG_RE = re.compile(r"<CONT>(.*?)</CONT>", re.IGNORECASE | re.DOTALL)
_GENERIC_REFUSAL_RE = re.compile(
    r"(?:sorry|i\s+can(?:not|'t)|i\s+am\s+unable|i\s+won't|can't\s+help\s+with\s+that)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
_BOILERPLATE_RE = re.compile(
    r"free\s+ebooks|newsletter|subscribe|planet\s+ebook|www\.|http|copyright|all\s+rights\s+reserved",
    re.IGNORECASE,
)
_NOISE_TOKEN_RE = re.compile(r"^[^a-zA-Z0-9]+$")
# Page-number token: bare digits that are likely headers/footers
_PAGE_NUM_RE = re.compile(r"^\d{1,5}$")


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", token.lower())


def _extract_continuation(output: str, prompt: str, expect_len: int) -> tuple[str, bool]:
    """Extract clean continuation text and whether it looks compliant."""
    text = (output or "").strip()

    tagged = _CONT_TAG_RE.search(text)
    if tagged:
        text = tagged.group(1).strip()

    text = text.strip("\"'` ")
    if text.lower().startswith(prompt.lower()[:30]):
        text = text[len(prompt):].strip()

    for prefix in (
        "...",
        "…",
        "continuation:",
        "continued:",
        "here is",
        "sure,",
        "sure.",
        "response:",
        "output:",
    ):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()

    words = text.split()
    truncated = " ".join(words[:expect_len])
    compliant = bool(truncated) and not _GENERIC_REFUSAL_RE.search(text)
    return truncated, compliant


def _leading_match_ratio(output: str, expected: str) -> float:
    """Fraction of expected tokens matched from the start until first mismatch."""
    out_tokens = [_normalize_token(t) for t in output.split()]
    exp_tokens = [_normalize_token(t) for t in expected.split()]
    out_tokens = [t for t in out_tokens if t]
    exp_tokens = [t for t in exp_tokens if t]
    if not exp_tokens:
        return 0.0
    prefix_matches = 0
    for ow, ew in zip(out_tokens, exp_tokens):
        if ow != ew:
            break
        prefix_matches += 1
    return prefix_matches / len(exp_tokens)


def _content_tokens(text: str) -> set[str]:
    return {
        t.lower() for t in _TOKEN_RE.findall(text)
        if len(t) > 2 and t.lower() not in _COMMON_WORDS
    }


def _looks_like_page_header(tokens: list[str]) -> bool:
    """Return True if the token window contains an embedded page header.

    Common pattern in PDFs: '... highly culti david copperfield 1148 vated ...'
    The header tokens sit inside what should be flowing prose.
    Only flags numbers > 50 to avoid false positives on small in-text numbers.
    """
    for i, tok in enumerate(tokens):
        if _PAGE_NUM_RE.match(tok):
            try:
                num = int(tok)
            except ValueError:
                continue
            if num > 50:
                before = tokens[max(0, i - 3):i]
                after = tokens[i + 1:i + 4]
                alpha_around = sum(1 for t in before + after if t.isalpha())
                if alpha_around >= 2:
                    return True
    return False


def _has_hyphen_fragment(tokens: list[str]) -> bool:
    """Return True if window has leftover line-break hyphen artefacts.

    E.g. ['immedi', '-', 'ately'] or ['culti', '-', 'vated'].
    """
    for i, tok in enumerate(tokens):
        if tok in {"-", "--"}:
            if i > 0 and i < len(tokens) - 1:
                left, right = tokens[i - 1], tokens[i + 1]
                if left.isalpha() and right.isalpha() and len(left) <= 6 and len(right) <= 6:
                    return True
    return False


def _is_good_prompt_window(tokens: list[str]) -> bool:
    if len(tokens) < 20:
        return False
    phrase = " ".join(tokens)
    if _BOILERPLATE_RE.search(phrase):
        return False
    if _looks_like_page_header(tokens):
        return False
    if _has_hyphen_fragment(tokens):
        return False

    alpha_tokens = [t for t in tokens if any(ch.isalpha() for ch in t)]
    if len(alpha_tokens) < 14:
        return False

    unique_ratio = len({t.lower() for t in alpha_tokens}) / max(1, len(alpha_tokens))
    if unique_ratio < 0.5:
        return False
    return True


def _is_good_expected_window(tokens: list[str]) -> bool:
    phrase = " ".join(tokens)
    if _BOILERPLATE_RE.search(phrase):
        return False

    if any(tok in {"-", "--", "—", "–"} for tok in tokens):
        return False
    if _looks_like_page_header(tokens):
        return False
    if _has_hyphen_fragment(tokens):
        return False

    # Reject windows with bare large page-like numbers (>100)
    if any(_PAGE_NUM_RE.match(t) and int(t) > 100 for t in tokens):
        return False

    alpha_tokens = [t for t in tokens if any(ch.isalpha() for ch in t)]
    min_alpha = min(settings.rare_phrase_expected_words, settings.rare_phrase_min_expected_alpha_tokens)
    if len(alpha_tokens) < min_alpha:
        return False

    noisy_count = sum(1 for t in tokens if _NOISE_TOKEN_RE.match(t))
    if noisy_count > max(1, len(tokens) // 4):
        return False

    return True


def _window_quality_score(prompt_tokens: list[str], expected_tokens: list[str]) -> float:
    prompt_alpha = [t for t in prompt_tokens if any(ch.isalpha() for ch in t)]
    expected_alpha = [t for t in expected_tokens if any(ch.isalpha() for ch in t)]
    prompt_content = len(_content_tokens(" ".join(prompt_tokens)))
    expected_content = len(_content_tokens(" ".join(expected_tokens)))
    unique_ratio = len({_normalize_token(t) for t in prompt_alpha if _normalize_token(t)}) / max(1, len(prompt_alpha))
    punctuation_penalty = sum(1 for t in expected_tokens if _NOISE_TOKEN_RE.match(t)) * 0.2
    return (0.9 * unique_ratio) + (0.25 * prompt_content) + (0.4 * expected_content) + (0.03 * len(expected_alpha)) - punctuation_penalty


def _prompt_distance(prompt_a: str, prompt_b: str) -> float:
    a = {_normalize_token(t) for t in prompt_a.split() if _normalize_token(t)}
    b = {_normalize_token(t) for t in prompt_b.split() if _normalize_token(t)}
    if not a or not b:
        return 1.0
    jaccard = len(a & b) / max(1, len(a | b))
    return 1.0 - jaccard


def _build_prompt_expected_pairs(book: BookRecord, sample_count: int, prompt_len: int, expect_len: int) -> tuple[list[str], list[str]]:
    candidates: list[tuple[float, str, str]] = []
    seen: set[str] = set()

    rare_all = book.datasets.get("ngrams", {}).get("rare_twentygrams", [])
    for item in rare_all:
        words = item.get("ngram", "").split()
        if len(words) < prompt_len + expect_len:
            continue
        window = words[:prompt_len + expect_len]
        if not _is_good_prompt_window(window):
            continue
        prompt = " ".join(window[:prompt_len])
        expected = " ".join(window[prompt_len:prompt_len + expect_len])
        if not _is_good_expected_window(window[prompt_len:prompt_len + expect_len]):
            continue
        if prompt not in seen:
            score = _window_quality_score(window[:prompt_len], window[prompt_len:prompt_len + expect_len])
            candidates.append((score, prompt, expected))
            seen.add(prompt)

    stride = max(1, (len(book.tokens) - (prompt_len + expect_len)) // max(1, sample_count * 6))
    for i in range(0, max(0, len(book.tokens) - (prompt_len + expect_len)), stride):
        window = book.tokens[i:i + prompt_len + expect_len]
        if len(window) < prompt_len + expect_len:
            continue
        if not _is_good_prompt_window(window):
            continue
        prompt = " ".join(window[:prompt_len])
        expected = " ".join(window[prompt_len:prompt_len + expect_len])
        if not _is_good_expected_window(window[prompt_len:prompt_len + expect_len]):
            continue
        if prompt not in seen:
            score = _window_quality_score(window[:prompt_len], window[prompt_len:prompt_len + expect_len])
            candidates.append((score, prompt, expected))
            seen.add(prompt)

    if not candidates:
        fallback_pairs = []
        for i in range(0, max(0, len(book.tokens) - (prompt_len + expect_len)), stride):
            window = book.tokens[i:i + prompt_len + expect_len]
            if len(window) < prompt_len + expect_len:
                continue
            prompt = " ".join(window[:prompt_len])
            expected = " ".join(window[prompt_len:prompt_len + expect_len])
            fallback_pairs.append((prompt, expected))
            if len(fallback_pairs) >= min(sample_count, 12):
                break
        prompts = [p for p, _ in fallback_pairs]
        expected = [e for _, e in fallback_pairs]
        return prompts, expected

    candidates.sort(key=lambda item: item[0], reverse=True)

    pairs: list[tuple[str, str]] = []
    for _, prompt, expected in candidates:
        if any(_prompt_distance(prompt, existing_prompt) < settings.rare_phrase_min_prompt_jaccard_distance for existing_prompt, _ in pairs):
            continue
        pairs.append((prompt, expected))
        if len(pairs) >= sample_count:
            break

    prompts = [p for p, _ in pairs[:sample_count]]
    expected = [e for _, e in pairs[:sample_count]]
    return prompts, expected


def _compatibility_score(output: str, prompt: str, expected: str) -> float:
    if not output:
        return 0.0
    if _GENERIC_REFUSAL_RE.search(output):
        return 0.0

    leading = _leading_match_ratio(output, expected)
    out_set = _content_tokens(output)
    exp_set = _content_tokens(expected)
    prompt_tail = " ".join(prompt.split()[-6:])
    tail_set = _content_tokens(prompt_tail)

    expected_overlap = len(out_set & exp_set)
    tail_overlap = len(out_set & tail_set)
    partial = len(out_set & exp_set) / max(1, len(exp_set))
    length_bonus = 0.2 if len(output.split()) >= 4 else 0.0

    return (2.3 * leading) + (1.2 * partial) + (0.2 * min(2, expected_overlap)) + (0.15 * min(2, tail_overlap)) + length_bonus


# ── Recognition detection ───────────────────────────────────────
# Models often refuse to reproduce copyrighted text verbatim but reveal
# they *recognise* the source by naming the book/author/characters or
# describing the scene.  This is itself strong evidence of training.

_REFUSAL_PATTERNS = re.compile(
    r"(?:can[\u2019']?t|cannot|unable\s+to|won[\u2019']?t|shouldn[\u2019']?t|not\s+able\s+to)\s+"
    r"(?:continue|reproduce|provide|complete|share|generate|output|give|write|copy)",
    re.IGNORECASE,
)

_SOURCE_AWARENESS = re.compile(
    r"copyright(?:ed)?|intellectual\s+property|fair\s+use|"
    r"(?:from|in|of)\s+(?:the\s+)?(?:book|novel|story|series|work|chapter|passage)|"
    r"(?:spoken|said|written|stated|quoted)\s+by|"
    r"(?:this|that|the)\s+(?:passage|excerpt|quote|line|text)\s+(?:is|comes?|appears?)\s+(?:from|in)|"
    r"(?:the|this)\s+(?:scene|chapter|part)\s+(?:where|when|in\s+which)",
    re.IGNORECASE,
)

_COMMON_WORDS = frozenset({
    "the", "he", "she", "it", "they", "we", "you", "i", "his", "her",
    "this", "that", "but", "and", "there", "then", "mr", "mrs", "dr",
    "sir", "now", "so", "yet", "not", "no", "yes", "one", "all", "my",
    "what", "when", "where", "why", "how", "who", "if", "at", "in",
    "on", "to", "for", "by", "an", "do", "did", "was", "is", "have",
    "has", "as", "or", "be", "am", "are", "were", "would", "could",
    "should", "may", "might", "got", "from", "out", "up", "down",
    "just", "very", "well", "still", "even", "some", "any", "each",
    "every", "only", "part", "chapter", "however", "also", "said",
    "been", "had", "with", "than", "them", "their", "its", "our",
    "your", "much", "more", "most", "about", "back", "into", "over",
    "after", "before", "here", "which", "never", "always", "again",
    "too", "like", "other", "such", "will", "can", "shall", "upon",
    "though", "through",
})


def _extract_notable_names(tokens: list[str], top_n: int = 30) -> set[str]:
    """Extract likely character / proper names from the book tokens."""
    proper: Counter[str] = Counter()
    for i, tok in enumerate(tokens):
        if tok and tok[0].isupper() and len(tok) > 2 and tok.lower() not in _COMMON_WORDS:
            # Skip words at the start of a sentence (after . ? !)
            if i > 0 and tokens[i - 1][-1:] not in ".?!":
                proper[tok] += 1
    return {name for name, cnt in proper.most_common(top_n) if cnt >= 3}


def _detect_recognition(
    response: str,
    book_metadata: dict,
    notable_names: set[str],
) -> tuple[bool, float, list[str]]:
    """Detect if the model recognises the source without reproducing it.

    Returns ``(is_recognised, confidence, signal_descriptions)``.
    """
    resp_lower = response.lower()
    signals: list[str] = []
    score = 0.0

    # 1. Book title mentioned -----------------------------------
    title = book_metadata.get("title", "").strip()
    if title and len(title) > 3:
        if title.lower() in resp_lower:
            signals.append(f"title_match:{title}")
            score += 0.45
        else:
            title_words = [w for w in title.lower().split() if len(w) > 3]
            matched = sum(1 for w in title_words if w in resp_lower)
            if matched >= 2 or (len(title_words) == 1 and matched == 1):
                signals.append(f"partial_title:{matched}/{len(title_words)}")
                score += 0.3

    # 2. Author mentioned ---------------------------------------
    author = book_metadata.get("author", "").strip()
    if author and len(author) > 3:
        if author.lower() in resp_lower:
            signals.append(f"author_match:{author}")
            score += 0.45
        else:
            for part in author.split():
                if len(part) > 3 and part.lower() in resp_lower:
                    signals.append(f"author_partial:{part}")
                    score += 0.3
                    break

    # 3. Refusal-with-knowledge pattern -------------------------
    if _REFUSAL_PATTERNS.search(response):
        signals.append("refusal_with_knowledge")
        score += 0.35

    # 4. Copyright / source awareness ---------------------------
    if _SOURCE_AWARENESS.search(response):
        signals.append("source_awareness")
        score += 0.25

    # 5. Character / proper-name mentions -----------------------
    names_found = [n for n in notable_names if n.lower() in resp_lower]
    if names_found:
        signals.append(f"character_names:{','.join(names_found[:3])}")
        score += min(0.35, 0.15 * len(names_found))

    # 6. Abnormally long response (explaining instead of continuing)
    word_count = len(response.split())
    if word_count > 40:
        signals.append(f"verbose:{word_count}w")
        score += 0.15

    recognised = score >= 0.3
    confidence = min(1.0, score)
    return recognised, confidence, signals


def run(book: BookRecord, model_name: str, sample_count: int) -> AgentResponse:
    prompt_len = settings.rare_phrase_prompt_words
    expect_len = settings.rare_phrase_expected_words
    prompts, expected = _build_prompt_expected_pairs(book, sample_count, prompt_len, expect_len)

    logger.info("Rare phrase agent: %d prompts prepared (prompt=%d tok, expect=%d tok)",
                len(prompts), prompt_len, expect_len)

    t0 = time.perf_counter()
    n = len(prompts)
    max_attempts = max(1, settings.rare_phrase_max_attempts)
    raw_outputs = ["" for _ in range(n)]
    outputs_truncated = ["" for _ in range(n)]
    compliance_flags = np.zeros(n, dtype=int)
    compatibility_scores = np.zeros(n, dtype=float)
    attempts_used = np.zeros(n, dtype=int)

    for attempt in range(max_attempts):
        pending = [i for i in range(n) if compliance_flags[i] == 0]
        if not pending:
            break

        temperature = min(0.7, 0.2 * attempt)
        retry_outputs = generate_model_continuations_rare_phrase(
            model_name,
            [prompts[i] for i in pending],
            book.tokens,
            max_tokens=expect_len * 4,
            expected_words=expect_len,
            temperature=temperature,
        )

        for idx, out in zip(pending, retry_outputs):
            candidate, base_ok = _extract_continuation(out, prompts[idx], expect_len)
            score = _compatibility_score(candidate, prompts[idx], expected[idx])

            # ── Compliance = "sincere continuation attempt" ─────────
            # Match-independent: did the model genuinely try to continue
            # the text?  Not a refusal, not empty, at least N words.
            sincere = (
                base_ok
                and len(candidate.split()) >= settings.rare_phrase_min_sincere_words
            )

            if sincere or score > compatibility_scores[idx] or not raw_outputs[idx]:
                raw_outputs[idx] = out
                outputs_truncated[idx] = candidate
                compatibility_scores[idx] = score

            compliance_flags[idx] = 1 if sincere else 0
            attempts_used[idx] = attempt + 1

        logger.info(
            "Rare phrase generation attempt %d/%d: sincere=%d/%d",
            attempt + 1,
            max_attempts,
            int(compliance_flags.sum()),
            n,
        )

    logger.info("Model outputs finalised in %.2fs", time.perf_counter() - t0)

    # ── Matching metrics ────────────────────────────────────────
    exact = np.array(
        [1 if o.strip().lower() == e.strip().lower() else 0 for o, e in zip(outputs_truncated, expected)],
        dtype=int,
    )
    # Partial: Jaccard-style word overlap (case-insensitive)
    partial = np.array([
        len(set(o.lower().split()) & set(e.lower().split())) / max(1, len(set(e.lower().split())))
        for o, e in zip(outputs_truncated, expected)
    ])
    # Token-level sequential match (how many leading words match)
    sequential = np.array([_leading_match_ratio(o, e) for o, e in zip(outputs_truncated, expected)])

    # Zero metrics for non-sincere outputs (refusals/empty)
    partial = partial * compliance_flags
    sequential = sequential * compliance_flags

    # Use sincere-attempt count as the proper trial denominator
    n_sincere = max(1, int(compliance_flags.sum()))

    exact_rate = float(exact.sum()) / n_sincere if n_sincere else 0.0
    partial_rate = float(partial.sum()) / n_sincere if n_sincere else 0.0
    sequential_rate = float(sequential.sum()) / n_sincere if n_sincere else 0.0

    # ── Recognition detection ───────────────────────────────────
    # Check ALL sincere outputs (not just match-compliant ones) for
    # recognition signals – the model may name characters/book/author
    # even when it doesn't reproduce the exact text.
    notable_names = _extract_notable_names(book.tokens)
    logger.info("Extracted %d notable names from book tokens", len(notable_names))

    recognition = []  # per-phrase: (bool, confidence, signals)
    for i in range(len(prompts)):
        if exact[i]:  # already matched verbatim — no need to check
            recognition.append((True, 1.0, ["verbatim_match"]))
        elif not compliance_flags[i]:
            recognition.append((False, 0.0, ["non_sincere_output"]))
        else:
            # Use the FULL (un-truncated) model response for recognition
            rec = _detect_recognition(raw_outputs[i], book.metadata, notable_names)
            recognition.append(rec)

    recognition_flags = np.array([r[0] for r in recognition], dtype=int)
    recognition_confidences = np.array([r[1] for r in recognition])
    recognition_count = int(recognition_flags.sum())
    recognition_rate = float(recognition_flags.mean()) if len(recognition_flags) else 0.0
    compliance_rate = float(compliance_flags.mean()) if len(compliance_flags) else 0.0

    logger.info(
        "Recognition detection: %d/%d recognised (%.1f%%)  avg confidence=%.2f",
        recognition_count, len(recognition_flags),
        recognition_rate * 100,
        float(recognition_confidences.mean()) if len(recognition_confidences) else 0.0,
    )
    logger.info(
        "Continuation compliance: %d/%d (%.1f%%)",
        int(compliance_flags.sum()),
        len(compliance_flags),
        compliance_rate * 100,
    )

    # ── Statistical tests ───────────────────────────────────────
    # Use sincere-attempt count as denominator for honest hypothesis
    # testing – refusals/empty outputs should not dilute the signal.
    n_trials = n_sincere

    # Soft match: partial overlap ≥ 50%
    soft_matches = int(np.sum(partial >= 0.5))

    p_value_exact = float(binomtest(
        int(exact.sum()), n_trials, p=0.01, alternative="greater"
    ).pvalue)
    p_value_soft = float(binomtest(
        soft_matches, n_trials, p=0.05, alternative="greater"
    ).pvalue)
    # Recognition: baseline expectation ~5% (random false positives)
    p_value_recognition = float(binomtest(
        recognition_count, n_trials, p=0.05, alternative="greater"
    ).pvalue)

    # ── Permutation test (non-parametric) ───────────────────────
    # Shuffle expected↔output associations.  If the model has
    # memorised the text, the TRUE pairings should yield higher
    # overlap than random pairings.  This captures continuous
    # signal missed by the binary soft-match threshold.
    rng = np.random.default_rng(settings.random_seed)
    sincere_idx = np.where(compliance_flags == 1)[0]
    observed_partial_sum = float(partial[sincere_idx].sum())
    n_perm = settings.permutation_iterations
    perm_beats = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(sincere_idx)
        perm_sum = sum(
            len(set(outputs_truncated[i].lower().split())
                & set(expected[j].lower().split()))
            / max(1, len(set(expected[j].lower().split())))
            for i, j in zip(sincere_idx, shuffled)
        )
        if perm_sum >= observed_partial_sum:
            perm_beats += 1
    p_value_permutation = (perm_beats + 1) / (n_perm + 1)
    logger.info("Permutation test: observed_sum=%.3f  p=%.4f  (%d/%d beats)",
                observed_partial_sum, p_value_permutation, perm_beats, n_perm)

    p_value = min(p_value_exact, p_value_soft, p_value_recognition, p_value_permutation)

    # Bootstrap CI on partial match rate
    boot = []
    for _ in range(settings.bootstrap_iterations):
        sampled = rng.choice(partial, size=len(partial), replace=True) if len(partial) else np.array([0.0])
        boot.append(float(np.mean(sampled)))
    ci = bootstrap_ci(np.array(boot))

    # LR from exact, soft, recognition, and permutation independently
    lr_exact = beta_binomial_lr(int(exact.sum()), n_trials, p0=0.01)
    lr_soft = beta_binomial_lr(soft_matches, n_trials, p0=0.05)
    lr_recognition = beta_binomial_lr(recognition_count, n_trials, p0=0.05)

    # Derive LR from permutation test.
    # The naïve (1-p)/p formula is capped at ~n_perm which makes it
    # unrealistically constant.  Instead use a calibrated log-linear
    # mapping: -log10(p) → LR, capped at a sensible evidence ceiling.
    if p_value_permutation < 1.0:
        neg_log_p = -math.log10(max(p_value_permutation, 1e-12))
        # Scale: -log10(0.05)=1.3→LR~2, -log10(0.003)=2.5→LR~15,
        #        -log10(0.0003)=3.5→LR~60
        lr_permutation = float(10 ** (neg_log_p * 0.75))
        # Cap: permutation evidence alone shouldn't exceed ~80
        lr_permutation = min(lr_permutation, 80.0)
    else:
        lr_permutation = 1.0

    # Combine: weighted geometric mean of the top signals rather than
    # a raw max, so a single lucky permutation doesn't dominate.
    lr_candidates = [lr_exact, lr_soft, lr_recognition, lr_permutation]
    lr_max_val = max(lr_candidates)
    lr_second = sorted(lr_candidates, reverse=True)[1] if len(lr_candidates) > 1 else 1.0
    # Blend: 70% best signal + 30% second-best (in log-space)
    lr = float(np.exp(0.7 * np.log(max(lr_max_val, 1e-12)) + 0.3 * np.log(max(lr_second, 1e-12))))

    if exact.sum() == 0 and soft_matches == 0 and recognition_count == 0 and p_value_permutation > 0.1:
        lr *= 0.5
    lr = clip_lr(lr, settings.lr_min, settings.lr_max)

    # ── Build comparison table ──────────────────────────────────
    comparisons = []
    for i in range(len(prompts)):
        rec_recognised, rec_confidence, rec_signals = recognition[i]
        comparisons.append({
            "prompt": prompts[i],
            "expected": expected[i],
            "model_response": outputs_truncated[i],
            "model_truncated": outputs_truncated[i],
            "full_response": raw_outputs[i],
            "exact_match": bool(exact[i]),
            "continuation_compliant": bool(compliance_flags[i]),
            "compatibility_score": round(float(compatibility_scores[i]), 4),
            "attempts_used": int(attempts_used[i]),
            "partial_overlap": round(float(partial[i]), 4),
            "sequential_overlap": round(float(sequential[i]), 4),
            "recognised": rec_recognised,
            "recognition_confidence": round(rec_confidence, 3),
            "recognition_signals": rec_signals,
        })

    logger.info(
        "Rare phrase result: exact=%d/%d (%.1f%%)  soft=%d/%d  partial=%.1f%%  "
        "sequential=%.1f%%  recognised=%d/%d (%.1f%%)  p=%.4f  LR=%.4f",
        int(exact.sum()), len(exact), exact_rate * 100,
        soft_matches, len(partial), partial_rate * 100,
        sequential_rate * 100,
        recognition_count, len(recognition_flags), recognition_rate * 100,
        p_value, lr,
    )

    return AgentResponse(
        agent_name="rare_phrase",
        hypothesis_test={
            "H0": "Model was NOT trained on this book",
            "H1": "Model WAS trained on this book",
        },
        metrics={
            "exact_match_rate": exact_rate,
            "partial_match_rate": partial_rate,
            "sequential_match_rate": sequential_rate,
            "recognition_rate": recognition_rate,
            "recognition_count": recognition_count,
            "continuation_compliance_rate": compliance_rate,
            "continuation_compliance_count": int(compliance_flags.sum()),
            "trial_count_used": int(len(prompts)),
            "generation_attempts_max": max_attempts,
            "generation_attempts_mean": float(np.mean(attempts_used)) if len(attempts_used) else 0.0,
            "soft_match_count": soft_matches,
            "binomial_test": {
                "successes_exact": int(exact.sum()),
                "successes_soft": soft_matches,
                "successes_recognition": recognition_count,
                "trials": n_trials,
                "trials_sincere": n_sincere,
                "trials_total": int(len(exact)),
                "p_value_exact": p_value_exact,
                "p_value_soft": p_value_soft,
                "p_value_recognition": p_value_recognition,
                "p_value_permutation": p_value_permutation,
            },
            "bootstrap_ci": {"lower": ci[0], "upper": ci[1]},
            "comparisons": comparisons,
        },
        p_value=p_value,
        likelihood_ratio=lr,
        log_likelihood_ratio=safe_log_lr(lr),
        evidence_direction="supports_H1" if lr > 1 else "supports_H0",
    )
