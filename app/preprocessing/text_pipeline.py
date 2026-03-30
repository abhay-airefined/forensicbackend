from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[.,!?;:\-()\[\]\"`]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Matches "word-\n  word" line-break hyphenation from PDFs.
# The newline after the hyphen is the reliable indicator that it's a
# line-break split, NOT a real compound like "well-known".
_LINEBREAK_HYPHEN_RE = re.compile(r"(\w{2,})-\s*\n\s*(\w{2,})")

# Page headers / footers: short lines with a title + optional page number.
# Matches patterns like "David Copperfield 1148" or "CHAPTER XII" sitting
# alone on a line (common in PDF-extracted text).
_PAGE_HEADER_RE = re.compile(
    r"^\s*(?:[A-Z][a-zA-Z]+(?:\s+[A-Za-z]+){0,4})\s+\d{1,5}\s*$",
    re.MULTILINE,
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # ── PDF artefact cleanup (must run before whitespace collapse) ──
    # 1. Rejoin line-break hyphens: "immedi-\nately" → "immediately"
    text = _LINEBREAK_HYPHEN_RE.sub(r"\1\2", text)
    # 2. Strip page headers/footers ("David Copperfield 1148")
    text = _PAGE_HEADER_RE.sub("", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_tokenize(text: str) -> list[str]:
    if not text:
        return []
    pieces = SENTENCE_RE.split(text)
    return [s.strip() for s in pieces if s.strip()]


def word_tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def split_segments(tokens: list[str], k: int, min_segment_tokens: int) -> list[list[str]]:
    if not tokens:
        return []
    k = max(1, min(k, len(tokens) // max(min_segment_tokens, 1) or 1))
    k = max(1, k)
    seg_len = max(min_segment_tokens, len(tokens) // k)
    segments: list[list[str]] = []
    for i in range(0, len(tokens), seg_len):
        chunk = tokens[i : i + seg_len]
        if chunk:
            segments.append(chunk)
    return segments
