from __future__ import annotations

import logging
import re
import time

from openai import AzureOpenAI, BadRequestError

from app.config import settings

logger = logging.getLogger("sfas.gateway")

_client: AzureOpenAI | None = None

# Minimal sanitiser to reduce content-filter rejections
_VIOLENCE_RE = re.compile(
    r"\b(kill(?:ed|ing|s)?|murder(?:ed|ing|s)?|stab(?:bed|bing|s)?|die(?:d|s)?|death|dead|blood|suicide|rape[ds]?|assault(?:ed)?|behead|slaughter|shoot|shot|gun|weapon|bomb|attack|torture)\b",
    re.IGNORECASE,
)


def _sanitise_prompt(text: str) -> str:
    """Replace violent/sensitive tokens with neutral placeholders to avoid
    Azure content-filter rejections.  This is a best-effort measure."""
    return _VIOLENCE_RE.sub("[...]", text)


def _get_client() -> AzureOpenAI:
    """Lazily create a single Azure OpenAI client reused across calls."""
    global _client
    if _client is None:
        if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
            raise RuntimeError(
                "SFAS_AZURE_OPENAI_ENDPOINT and SFAS_AZURE_OPENAI_API_KEY "
                "environment variables are required. See .env.example."
            )
        _client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        logger.info(
            "Azure OpenAI client created  endpoint=%s  api_version=%s",
            settings.azure_openai_endpoint,
            settings.azure_openai_api_version,
        )
    return _client


def generate_model_continuations(
    model_name: str,
    prompts: list[str],
    corpus_tokens: list[str],  # kept for interface compatibility; unused
    max_tokens: int = 20,
) -> list[str]:
    """Call Azure OpenAI to generate a text continuation for every prompt.

    *model_name* is treated as the Azure OpenAI **deployment name**.
    If empty, the configured default deployment is used.
    """
    client = _get_client()
    deployment = model_name or settings.azure_openai_deployment
    if not deployment:
        raise RuntimeError(
            "No Azure OpenAI deployment specified. Pass model_name in the "
            "request or set SFAS_AZURE_OPENAI_DEPLOYMENT."
        )

    logger.info(
        "Starting %d Azure OpenAI calls  deployment=%s  max_tokens=%d",
        len(prompts),
        deployment,
        max_tokens,
    )
    total_start = time.perf_counter()
    outputs: list[str] = []
    failed = 0
    filtered = 0

    for i, prompt in enumerate(prompts):
        call_start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a text-continuation engine. "
                            "Continue the following text passage naturally. "
                            "Output ONLY the continuation text, nothing else."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            text = response.choices[0].message.content or ""
            call_elapsed = time.perf_counter() - call_start
            usage = response.usage
            tokens_used = f"in={usage.prompt_tokens}/out={usage.completion_tokens}" if usage else "n/a"
            logger.debug(
                "  [%d/%d] %.2fs  tokens(%s)  output=%d chars",
                i + 1,
                len(prompts),
                call_elapsed,
                tokens_used,
                len(text),
            )
            outputs.append(text.strip())
        except BadRequestError as exc:
            # ── Content filter hit → retry with sanitised prompt ─────
            if "content_filter" in str(exc) or "content management policy" in str(exc):
                filtered += 1
                logger.warning(
                    "  [%d/%d] Content filter hit — retrying with sanitised prompt",
                    i + 1, len(prompts),
                )
                try:
                    safe_prompt = _sanitise_prompt(prompt)
                    response = client.chat.completions.create(
                        model=deployment,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a text-continuation engine. "
                                    "Continue the following text passage naturally. "
                                    "Output ONLY the continuation text, nothing else."
                                ),
                            },
                            {"role": "user", "content": safe_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.7,
                    )
                    text = response.choices[0].message.content or ""
                    outputs.append(text.strip())
                except Exception:
                    logger.warning(
                        "  [%d/%d] Sanitised retry also failed; returning empty",
                        i + 1, len(prompts),
                    )
                    failed += 1
                    outputs.append("")
            else:
                call_elapsed = time.perf_counter() - call_start
                failed += 1
                logger.exception(
                    "  [%d/%d] FAILED after %.2fs  prompt=%.80s",
                    i + 1, len(prompts), call_elapsed, prompt,
                )
                outputs.append("")
        except Exception:
            call_elapsed = time.perf_counter() - call_start
            failed += 1
            logger.exception(
                "  [%d/%d] FAILED after %.2fs  prompt=%.80s",
                i + 1,
                len(prompts),
                call_elapsed,
                prompt,
            )
            outputs.append("")

    total_elapsed = time.perf_counter() - total_start
    logger.info(
        "Azure OpenAI batch done  %d/%d ok  %d filtered  %.2fs total  (avg %.2fs/call)",
        len(prompts) - failed,
        len(prompts),
        filtered,
        total_elapsed,
        total_elapsed / max(1, len(prompts)),
    )
    return outputs


# ── Rare-phrase specific continuation ────────────────────────────────

_RARE_PHRASE_SYSTEM = (
    "You are a strict continuation engine. "
    "Continue the given prefix with exactly the requested number of next words. "
    "Never explain, refuse, apologize, or add commentary. "
    "Return only one XML tag in this exact format: <CONT>...</CONT>. "
    "Inside <CONT>, output only continuation words and punctuation. "
    "Do not repeat the prefix."
)


def generate_model_continuations_rare_phrase(
    model_name: str,
    prompts: list[str],
    corpus_tokens: list[str],
    max_tokens: int = 24,
    expected_words: int = 8,
    temperature: float = 0.0,
) -> list[str]:
    """Specialised continuation for rare-phrase detection.

    Uses a stricter system prompt that enforces verbatim completion and
    lower temperature to maximise memorisation signal.
    """
    client = _get_client()
    deployment = model_name or settings.azure_openai_deployment
    if not deployment:
        raise RuntimeError(
            "No Azure OpenAI deployment specified. Pass model_name in the "
            "request or set SFAS_AZURE_OPENAI_DEPLOYMENT."
        )

    logger.info(
        "Starting %d rare-phrase calls  deployment=%s  max_tokens=%d  temp=%.2f",
        len(prompts), deployment, max_tokens, temperature,
    )
    total_start = time.perf_counter()
    outputs: list[str] = []
    failed = 0
    filtered = 0

    for i, prompt in enumerate(prompts):
        call_start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": _RARE_PHRASE_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Prefix:\n{prompt}\n\n"
                            f"Task: output exactly {expected_words} next words as continuation.\n"
                            "Format: <CONT>your continuation words here</CONT>"
                        ),
                    },
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content or ""
            call_elapsed = time.perf_counter() - call_start
            usage = response.usage
            tokens_used = f"in={usage.prompt_tokens}/out={usage.completion_tokens}" if usage else "n/a"
            logger.debug(
                "  [%d/%d] %.2fs  tokens(%s)  output=%d chars",
                i + 1, len(prompts), call_elapsed, tokens_used, len(text),
            )
            outputs.append(text.strip())
        except BadRequestError as exc:
            if "content_filter" in str(exc) or "content management policy" in str(exc):
                filtered += 1
                logger.warning(
                    "  [%d/%d] Content filter hit — retrying with sanitised prompt",
                    i + 1, len(prompts),
                )
                try:
                    safe_prompt = _sanitise_prompt(prompt)
                    response = client.chat.completions.create(
                        model=deployment,
                        messages=[
                            {"role": "system", "content": _RARE_PHRASE_SYSTEM},
                            {
                                "role": "user",
                                "content": (
                                    f"Prefix:\n{safe_prompt}\n\n"
                                    f"Task: output exactly {expected_words} next words as continuation.\n"
                                    "Format: <CONT>your continuation words here</CONT>"
                                ),
                            },
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    text = response.choices[0].message.content or ""
                    outputs.append(text.strip())
                except Exception:
                    logger.warning("  [%d/%d] Sanitised retry also failed", i + 1, len(prompts))
                    failed += 1
                    outputs.append("")
            else:
                failed += 1
                logger.exception("  [%d/%d] FAILED  prompt=%.80s", i + 1, len(prompts), prompt)
                outputs.append("")
        except Exception:
            failed += 1
            logger.exception("  [%d/%d] FAILED  prompt=%.80s", i + 1, len(prompts), prompt)
            outputs.append("")

    total_elapsed = time.perf_counter() - total_start
    logger.info(
        "Rare-phrase batch done  %d/%d ok  %d filtered  %.2fs total",
        len(prompts) - failed, len(prompts), filtered, total_elapsed,
    )
    return outputs
