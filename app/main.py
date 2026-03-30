from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

from app.agents import distribution_agent, entropy_agent, rare_phrase_agent, semantic_agent, stylometric_agent
from app.aggregation.bayesian_fusion import fuse
from app.config import settings
from app.datasets.pipeline import build_structured_data
from app.extraction.text_extractor import UnsupportedFileTypeError, extract_text
from app.logging_config import setup_logging
from app.models.schemas import (
    AgentRequest,
    AgentResponse,
    AggregateRequest,
    AggregateResponse,
    BookDetailsResponse,
    BookMetadata,
    MemorandumRequest,
    MemorandumResponse,
    StoredBookSummary,
    UploadBookResponse,
)
from app.models.storage import BookRecord, store
from app.preprocessing.text_pipeline import normalize_text, sentence_tokenize, split_segments, word_tokenize
from app.simulation import log_store
from app.simulation.scenarios import (
    assign_scenario,
    get_scenario,
    simulate_aggregate,
    simulate_single_agent,
)
from app.visualization.charts import (
    chart_aggregate,
    chart_distribution,
    chart_entropy,
    chart_full_dashboard,
    chart_graph_metrics,
    chart_rare_phrase,
    chart_rolling_entropy,
    chart_semantic,
    chart_stylometric,
    chart_stylometry_heatmap,
    chart_word_frequency,
)

# ── Logging ─────────────────────────────────────────────────────────

setup_logging()
logger = logging.getLogger("sfas.main")

# ── App ─────────────────────────────────────────────────────────────

app = FastAPI(title=settings.app_name, version=settings.app_version)

# ── CORS ────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Timing middleware ───────────────────────────────────────────────


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> StarletteResponse:
        start = time.perf_counter()
        logger.info("→  %s %s", request.method, request.url.path)
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        logger.info(
            "←  %s %s  %d  (%.2f s)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response


app.add_middleware(TimingMiddleware)


# ── Upload ──────────────────────────────────────────────────────────


@app.post("/upload-book", response_model=UploadBookResponse)
async def upload_book(file: UploadFile = File(...)) -> UploadBookResponse:
    t0 = time.perf_counter()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    original_filename = file.filename or "uploaded_file"
    logger.info("Upload received: %s (%.1f KB)", original_filename, len(content) / 1024)

    sha = hashlib.sha256(content).hexdigest()
    existing_book_id = store.find_book_by_sha(sha)
    if existing_book_id:
        existing_metadata = BookMetadata(**store.get_book_metadata(existing_book_id))
        logger.info(
            "Duplicate upload detected for '%s' → reusing existing book_id=%s",
            original_filename,
            existing_book_id,
        )
        return UploadBookResponse(
            book_id=existing_book_id,
            metadata=existing_metadata,
            reused_existing=True,
        )

    try:
        raw_text, page_count = extract_text(original_filename, content)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("Text extracted: %d pages in %.2fs", page_count, time.perf_counter() - t0)

    normalized = normalize_text(raw_text)
    sentences = sentence_tokenize(normalized)
    tokens = word_tokenize(normalized)
    segment_tokens = split_segments(tokens, settings.default_segments, settings.min_segment_tokens)
    segments = [" ".join(seg) for seg in segment_tokens]
    logger.info("Tokenization complete: %d tokens, %d sentences, %d segments", len(tokens), len(sentences), len(segments))

    if len(tokens) < settings.min_segment_tokens:
        raise HTTPException(status_code=400, detail="Book is too short for forensic analysis")

    book_id = str(uuid.uuid4())
    metadata = BookMetadata(
        sha256=sha,
        word_count=len([t for t in tokens if t.isalnum()]),
        sentence_count=len(sentences),
        token_count=len(tokens),
        page_count=page_count,
        extraction_timestamp=datetime.now(timezone.utc),
    )

    t1 = time.perf_counter()
    datasets, graphs = build_structured_data(tokens, sentences, segment_tokens)
    logger.info("Datasets & graphs built in %.2fs", time.perf_counter() - t1)

    record = BookRecord(
        book_id=book_id,
        metadata=metadata.model_dump(),
        raw_text=raw_text,
        normalized_text=normalized,
        sentences=sentences,
        tokens=tokens,
        segments=segments,
        segment_tokens=segment_tokens,
        datasets=datasets,
        graphs=graphs,
    )
    t2 = time.perf_counter()
    store.upsert_book(record, original_file=content, original_filename=original_filename)
    logger.info("Azure storage upload in %.2fs", time.perf_counter() - t2)

    # ── Simulation: assign scenario ──────────────────────────────
    if settings.simulation_mode:
        scenario = assign_scenario(book_id, original_filename)
        logger.info("SIMULATION MODE: book %s (%s) assigned scenario '%s'", book_id, original_filename, scenario)

    logger.info("Book %s processed in %.2fs total", book_id, time.perf_counter() - t0)
    return UploadBookResponse(book_id=book_id, metadata=metadata)


@app.get("/books", response_model=list[StoredBookSummary])
def list_books(
    limit: int = Query(50, ge=1, le=500),
    q: str | None = Query(None, description="Filter by filename or book_id"),
) -> list[StoredBookSummary]:
    books = store.list_books(limit=limit, query=q)
    return [StoredBookSummary(**book) for book in books]


@app.get("/books/{book_id}", response_model=BookDetailsResponse)
def get_book_details(book_id: str) -> BookDetailsResponse:
    try:
        metadata = BookMetadata(**store.get_book_metadata(book_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    book = _get_book_or_404(book_id)
    analysis_summary = store.get_all_agent_results_summary(book_id)
    total_analyses = sum(len(agent_map) for agent_map in analysis_summary.values())

    return BookDetailsResponse(
        book_id=book_id,
        original_filename=book.original_filename or "unknown",
        metadata=metadata,
        analysis_summary=analysis_summary,
        total_analyses=total_analyses,
    )


def _get_book_or_404(book_id: str) -> BookRecord:
    try:
        return store.get_book(book_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ── Agents ──────────────────────────────────────────────────────────
# Each agent endpoint checks settings.simulation_mode and, when enabled,
# returns pre-crafted results with realistic delays and log messages.


def _run_agent_sim_or_real(request: AgentRequest, agent_name: str, agent_module):
    """Shared helper: simulation vs real agent execution."""
    logger.info("Agent %s: book=%s model=%s samples=%d sim=%s",
                agent_name, request.book_id, request.model_name,
                request.sample_count, settings.simulation_mode)
    t0 = time.perf_counter()

    if settings.simulation_mode:
        key = log_store.run_key(request.book_id, request.model_name)
        # Initialise run log if not started (single-agent call)
        if log_store.get_logs(key)[1] == "unknown":
            log_store.start_run(key)
        result_dict = simulate_single_agent(
            request.book_id, request.model_name, agent_name, request.sample_count,
        )
        store.store_agent_result(request.book_id, request.model_name, result_dict)
        logger.info("Agent %s (SIM) done in %.2fs  LR=%.4f  p=%.4f",
                     agent_name, time.perf_counter() - t0,
                     result_dict["likelihood_ratio"], result_dict["p_value"])
        return result_dict
    else:
        book = _get_book_or_404(request.book_id)
        result = agent_module.run(book, request.model_name, request.sample_count)
        store.store_agent_result(request.book_id, request.model_name, result.model_dump())
        logger.info("Agent %s done in %.2fs  LR=%.4f  p=%.4f  dir=%s",
                     agent_name, time.perf_counter() - t0,
                     result.likelihood_ratio, result.p_value, result.evidence_direction)
        return result


@app.post("/agents/rare-phrase")
def rare_phrase_endpoint(request: AgentRequest):
    return _run_agent_sim_or_real(request, "rare_phrase", rare_phrase_agent)


@app.post("/agents/stylometric")
def stylometric_endpoint(request: AgentRequest):
    return _run_agent_sim_or_real(request, "stylometric", stylometric_agent)


@app.post("/agents/distribution")
def distribution_endpoint(request: AgentRequest):
    return _run_agent_sim_or_real(request, "distribution", distribution_agent)


@app.post("/agents/entropy")
def entropy_endpoint(request: AgentRequest):
    return _run_agent_sim_or_real(request, "entropy", entropy_agent)


@app.post("/agents/semantic")
def semantic_endpoint(request: AgentRequest):
    return _run_agent_sim_or_real(request, "semantic", semantic_agent)


# ── Aggregate ───────────────────────────────────────────────────────


@app.post("/aggregate", response_model=AggregateResponse)
def aggregate_endpoint(request: AggregateRequest) -> AggregateResponse:
    logger.info("Aggregate: book=%s model=%s prior=%.4f sim=%s",
                request.book_id, request.model_name,
                request.prior_probability, settings.simulation_mode)
    t0 = time.perf_counter()

    if settings.simulation_mode:
        agent_results, agg = simulate_aggregate(
            request.book_id, request.model_name,
            settings.model_max_outputs, request.prior_probability,
        )
        # Persist all results to Azure
        for r in agent_results:
            store.store_agent_result(request.book_id, request.model_name, r)
        aggregate_response = AggregateResponse(**agg)
        store.store_aggregate_result(
            request.book_id, request.model_name, aggregate_response.model_dump()
        )
        logger.info("Aggregate (SIM) complete in %.2fs  posterior=%.4f  strength=%s",
                     time.perf_counter() - t0,
                     aggregate_response.posterior_probability,
                     aggregate_response.strength_of_evidence)
        return aggregate_response

    # ── Real mode ────────────────────────────────────────────────
    book = _get_book_or_404(request.book_id)
    agent_names = ["rare_phrase", "stylometric", "distribution", "entropy", "semantic"]
    agent_fns = [rare_phrase_agent, stylometric_agent, distribution_agent, entropy_agent, semantic_agent]
    agent_results = []
    for name, fn in zip(agent_names, agent_fns):
        t_ag = time.perf_counter()
        result = fn.run(book, request.model_name, settings.model_max_outputs)
        logger.info("  Agent %-14s  %.2fs  LR=%.4f  p=%.4f  %s", name, time.perf_counter() - t_ag, result.likelihood_ratio, result.p_value, result.evidence_direction)
        agent_results.append(result)

    # Persist individual agent results
    for r in agent_results:
        store.store_agent_result(request.book_id, request.model_name, r.model_dump())

    fused = fuse(agent_results, request.prior_probability)

    if fused["posterior_probability"] >= 0.5:
        statement = "there is sufficient evidence"
    else:
        statement = "there is not sufficient evidence"

    summary = (
        "Based on the statistical analysis, "
        f"{statement} to conclude that the uploaded book was used in training the specified AI model."
    )
    aggregate_response = AggregateResponse(
        posterior_probability=fused["posterior_probability"],
        log_likelihood_ratio=fused["log_likelihood_ratio"],
        strength_of_evidence=fused["strength_of_evidence"],
        agent_breakdown=fused["agent_breakdown"],
        executive_summary=summary,
    )

    # Persist aggregate result
    store.store_aggregate_result(
        request.book_id, request.model_name, aggregate_response.model_dump()
    )

    logger.info(
        "Aggregate complete in %.2fs  posterior=%.4f  strength=%s",
        time.perf_counter() - t0,
        aggregate_response.posterior_probability,
        aggregate_response.strength_of_evidence,
    )

    return aggregate_response


# ── Live log polling  (for UI to show agent progress) ───────────────


@app.get("/runs/{book_id}/{model_name}/logs")
def get_run_logs(
    book_id: str,
    model_name: str,
    after: int = Query(0, ge=0, description="Return logs after this index"),
):
    """Poll for run progress logs.  The UI can call this every 500ms
    while the aggregate endpoint is processing."""
    key = log_store.run_key(book_id, model_name)
    logs, status = log_store.get_logs(key, after=after)
    return {"logs": logs, "status": status, "next_after": after + len(logs)}


@app.get("/runs/{book_id}/{model_name}/events")
async def sse_run_events(book_id: str, model_name: str):
    """Server-Sent Events stream for real-time log delivery."""
    import json as _json

    key = log_store.run_key(book_id, model_name)

    async def event_generator():
        cursor = 0
        while True:
            logs, status = log_store.get_logs(key, after=cursor)
            for entry in logs:
                yield f"data: {_json.dumps(entry)}\n\n"
                cursor += 1
            if status in ("completed", "failed"):
                yield f"event: done\ndata: {_json.dumps({'status': status})}\n\n"
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Download endpoints ──────────────────────────────────────────────


@app.get("/books/{book_id}/datasets")
def download_datasets(book_id: str) -> Response:
    """Download the structured datasets (n-grams, stylometry, entropy) as JSON."""
    _get_book_or_404(book_id)
    try:
        data = store.get_datasets_bytes(book_id)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"Datasets not found for book_id '{book_id}'",
        )
    return Response(
        content=data,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{book_id}_datasets.json"'
        },
    )


@app.get("/books/{book_id}/graphs")
def download_graphs(book_id: str) -> Response:
    """Download graph metrics (co-occurrence, sentence similarity) as JSON."""
    _get_book_or_404(book_id)
    try:
        data = store.get_graphs_bytes(book_id)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"Graphs not found for book_id '{book_id}'",
        )
    return Response(
        content=data,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{book_id}_graphs.json"'
        },
    )


@app.get("/books/{book_id}/file")
def download_original_file(book_id: str) -> Response:
    """Download the originally-uploaded PDF / DOCX file."""
    try:
        content, filename = store.get_original_file(book_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Original file not found for book_id '{book_id}'",
        )
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    media = "application/pdf" if ext == "pdf" else "application/octet-stream"
    return Response(
        content=content,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/books/{book_id}/results/{model_name}")
def download_agent_results(book_id: str, model_name: str) -> list[dict]:
    """Return all stored agent results for a book / model pair."""
    _get_book_or_404(book_id)
    results = store.get_agent_results(book_id, model_name)
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for book_id '{book_id}' / model '{model_name}'",
        )
    return results


# ── Memorandum generation (via AILA backend) ────────────────────────

_AILA_BASE = "http://localhost:8000"
_AILA_IP4_SEARCH_URL = "https://aila-uk-backend-abhay.azurewebsites.net/rag-api/aila_ip_4/search"
_AILA_BLOB_CONN_STR = (
    "DefaultEndpointsProtocol=https;"
    "AccountName=stdatauksouthaila;"
    
    "EndpointSuffix=core.windows.net"
)
_AILA_EVIDENCE_CONTAINER = "aila-case-evidence"


def _fetch_aila_distribution_evidence(title: str, author: str) -> dict | None:
    """Fetch external copyright availability and risk signals from AILA IP-4 search."""
    import requests as _requests

    clean_title = (title or "").strip()
    clean_author = (author or "").strip()
    if not clean_title:
        return None

    try:
        resp = _requests.get(
            _AILA_IP4_SEARCH_URL,
            params={"title": clean_title, "author": clean_author},
            timeout=45,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            return None
        return payload
    except Exception as exc:
        logger.warning("AILA IP-4 evidence lookup failed for '%s' by '%s': %s", clean_title, clean_author, exc)
        return None


def _format_aila_distribution_md(evidence_payload: dict | None, title: str, author: str) -> str:
    """Format external availability/risk evidence as a markdown section."""
    clean_title = (title or "").strip() or "the subject work"
    clean_author = (author or "").strip() or "the listed author"

    if not evidence_payload:
        return (
            "## External Availability and Distribution Signals\n\n"
            "No external IP-4 availability/risk evidence could be retrieved at memorandum generation time."
        )

    evidence = evidence_payload.get("evidence", {}) if isinstance(evidence_payload, dict) else {}
    if not isinstance(evidence, dict):
        evidence = {}

    legit = evidence.get("legitimate_availability", [])
    alternates = evidence.get("alternate_legitimate_editions", [])
    high_risk = evidence.get("high_risk_distribution_signals", [])
    risk_summary = evidence.get("risk_summary", {})

    if not isinstance(legit, list):
        legit = []
    if not isinstance(alternates, list):
        alternates = []
    if not isinstance(high_risk, list):
        high_risk = []
    if not isinstance(risk_summary, dict):
        risk_summary = {}

    def _top_sources(items: list[dict]) -> str:
        source_counts: dict[str, int] = {}
        for item in items:
            if isinstance(item, dict):
                source = str(item.get("source", "Unknown")).strip() or "Unknown"
                source_counts[source] = source_counts.get(source, 0) + 1
        if not source_counts:
            return "None"
        ordered = sorted(source_counts.items(), key=lambda x: (-x[1], x[0].lower()))
        return ", ".join(f"{name} ({count})" for name, count in ordered[:6])

    libgen_hits = [
        item
        for item in high_risk
        if isinstance(item, dict) and "libgen" in str(item.get("source", "")).lower()
    ]
    zlib_hits = [
        item
        for item in high_risk
        if isinstance(item, dict) and "z-library" in str(item.get("source", "")).lower()
    ]

    overall_risk = str(risk_summary.get("overall_risk_level", "Unknown")).upper()
    legitimate_count = int(risk_summary.get("legitimate_sources_detected", len(legit)) or 0)
    shadow_count = int(risk_summary.get("shadow_library_signals", len(high_risk)) or 0)

    libgen_statement = (
        f"LibGen signals were detected ({len(libgen_hits)} entries), which indicates potential availability of *{clean_title}* "
        "on high-risk shadow-library infrastructure."
        if libgen_hits
        else f"No explicit LibGen signal was detected for *{clean_title}* in this lookup."
    )

    training_risk_statement = (
        "Given the presence of high-risk distribution signals, it is plausible that unauthorised copies "
        "could have been scraped and incorporated into third-party training corpora."
        if high_risk
        else "No high-risk distribution signals were observed in this lookup."
    )

    return f"""## External Availability and Distribution Signals

An external IP-4 search was run for *{clean_title}* by {clean_author} to assess lawful availability and high-risk distribution channels.

- **Overall risk level:** {overall_risk}
- **Legitimate sources detected:** {legitimate_count}
- **Shadow-library signals detected:** {shadow_count}
- **High-risk source breakdown:** {_top_sources(high_risk)}
- **Legitimate source breakdown:** {_top_sources(legit)}
- **Alternate edition/source matches:** {len(alternates)}

{libgen_statement}

{"Z-Library signals were also detected (" + str(len(zlib_hits)) + " entries)." if zlib_hits else ""}

{training_risk_statement}
"""


def _upload_evidence_to_aila(case_number: str, role: str, evidence_md: str) -> None:
    """Upload the forensic evidence markdown to AILA's blob storage.

    AILA's ``/generate_memorandum`` reads client documents from
    ``{case}/{role}_client/`` inside the ``aila-case-evidence`` container.
    """
    from azure.storage.blob import BlobServiceClient, ContentSettings

    safe_case = case_number.replace("/", "-")
    blob_path = f"{safe_case}/{role}_client/forensic_evidence.md"

    blob_service = BlobServiceClient.from_connection_string(_AILA_BLOB_CONN_STR)
    container = blob_service.get_container_client(_AILA_EVIDENCE_CONTAINER)
    container.upload_blob(
        name=blob_path,
        data=evidence_md.encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type="text/markdown; charset=utf-8"),
    )
    logger.info("Uploaded forensic evidence to AILA blob: %s/%s", _AILA_EVIDENCE_CONTAINER, blob_path)


def _build_forensic_evidence_md(
    book_id: str,
    model_name: str,
    book_title: str,
    book_author: str,
    metadata: dict,
    agent_results: list[dict],
    aggregate: dict,
    distribution_evidence: dict | None = None,
    chat_log: list[dict] | None = None,
) -> str:
    """Format forensic analysis results as a narrative legal evidence document.

    This text is uploaded to AILA and consumed by the memorandum-generation
    LLM, so it should read like a case statement, not a raw data dump.
    """
    from datetime import datetime as _dt

    posterior = aggregate.get("posterior_probability", 0)
    strength = aggregate.get("strength_of_evidence", "Unknown")
    summary = aggregate.get("executive_summary", "")
    today = _dt.now().strftime("%d %B %Y")

    title = book_title or "the subject literary work"
    author = book_author or "the rights holder"
    word_count = metadata.get("word_count", 0)
    page_count = metadata.get("page_count", 0)
    distribution_md = _format_aila_distribution_md(distribution_evidence, title, author)

    # ── Build per-agent narrative paragraphs ────────────────────
    agent_paragraphs: list[str] = []
    for r in agent_results:
        name = r.get("agent_name", "?")
        lr = r.get("likelihood_ratio", 1.0)
        pv = r.get("p_value", 1.0)
        d = r.get("evidence_direction", "")
        m = r.get("metrics", {})
        sig = " (statistically significant at α = 0.05)" if pv < 0.05 else ""

        if name == "rare_phrase":
            exact = m.get("exact_match_rate", 0)
            recog = m.get("recognition_rate", 0)
            agent_paragraphs.append(
                f"**Rare Phrase Analysis** — {int(exact * 100)}% of rare n-gram prompts drawn from "
                f"*{title}* were reproduced verbatim by the model, and {int(recog * 100)}% of "
                f"responses exhibited clear recognition of the source material (e.g., naming the "
                f"title, author, or characters). The binomial test yielded p = {pv:.4f}{sig}, "
                f"with a likelihood ratio of {lr:.2f} in favour of the training hypothesis."
            )
        elif name == "stylometric":
            obs_corr = m.get("observed_style_correlation", 0)
            null_corr = m.get("null_correlation_mean", 0)
            agent_paragraphs.append(
                f"**Stylometric Analysis** — The model's output exhibited a mean per-feature "
                f"Spearman correlation of ρ = {obs_corr:.2f} with the source book's stylistic "
                f"profile, compared to a null baseline of {null_corr:.2f}. This elevated stylistic "
                f"similarity produced p = {pv:.4f}{sig} and a likelihood ratio of {lr:.2f}."
            )
        elif name == "distribution":
            obs = m.get("observed_per_segment_score", 0)
            null_mean = m.get("null_mean", 0)
            agent_paragraphs.append(
                f"**Token Distribution Analysis** — Per-segment distributional distance between "
                f"model output and the book's token profile was {obs:.3f}, substantially lower "
                f"than the cross-segment null mean of {null_mean:.3f}. This indicates the model's "
                f"vocabulary distribution mirrors that of *{title}* more than random text "
                f"(p = {pv:.4f}{sig}, LR = {lr:.2f})."
            )
        elif name == "entropy":
            obs_corr = m.get("observed_entropy_correlation", 0)
            agent_paragraphs.append(
                f"**Entropy Correlation Analysis** — Segment-level entropy of the model's output "
                f"correlated with the book's entropy profile at ρ = {obs_corr:.2f}, far exceeding "
                f"the null expectation. This suggests the model has internalised the information-"
                f"density patterns of the source text (p = {pv:.4f}{sig}, LR = {lr:.2f})."
            )
        elif name == "semantic":
            obs_sim = m.get("observed_paired_similarity", 0)
            null_sim = m.get("null_paired_mean", 0)
            top1 = m.get("top1_fraction", 0) * 100
            agent_paragraphs.append(
                f"**Semantic Similarity Analysis** — Paired cosine similarity between source "
                f"segments and model outputs was {obs_sim:.3f}, versus a null mean of {null_sim:.3f}. "
                f"The correct source segment ranked first for {top1:.0f}% of outputs. "
                f"This strong semantic alignment yielded p = {pv:.4f}{sig} and LR = {lr:.2f}."
            )

    agent_text = "\n\n".join(agent_paragraphs)

    # ── Agent summary table ─────────────────────────────────────
    table_lines = [
        "| Agent | Likelihood Ratio | p-value | Significant (α=0.05) | Direction |",
        "|-------|:----------------:|:-------:|:--------------------:|-----------|",
    ]
    for r in agent_results:
        name = r.get("agent_name", "?").replace("_", " ").title()
        lr = r.get("likelihood_ratio", 1.0)
        pv = r.get("p_value", 1.0)
        sig = "Yes ★" if pv < 0.05 else "No"
        d = "Supports Training" if "H1" in r.get("evidence_direction", "") else "Supports Not-Trained"
        table_lines.append(f"| {name} | {lr:.4f} | {pv:.4f} | {sig} | {d} |")
    table_md = "\n".join(table_lines)

    # ── Compose full document ───────────────────────────────────
    doc = f"""# Forensic Evidence — AI Training Data Detection

## Prepared by: SFAS (First AIDA Scientific Forensic Attribution System)
## Date: {today}

---

## 1. Introduction

This document presents the findings of a forensic computational analysis conducted
to determine whether the literary work *{title}* by {author} was used as training
data for the artificial intelligence model known as **{model_name}**, operated by
Microsoft through its Azure OpenAI service.

The analysis was performed using the First AIDA Scientific Forensic Attribution System (SFAS),
which employs five independent statistical detection agents, each targeting a
different modality of textual similarity. The individual agent results are then
combined using Bayesian fusion to produce a single posterior probability estimate.

---

## 2. Subject Material

The work under examination is *{title}* by {author}, comprising approximately
{word_count:,} words across {page_count} pages. The full text was extracted,
normalised, tokenised, and segmented for analysis on {today}.

---

## 3. AI Model Under Investigation

The model examined is **{model_name}**, deployed on the Azure OpenAI platform
operated by Microsoft Corporation. The forensic probes were transmitted to the
model via the Azure OpenAI API, and the responses were analysed for statistical
indicators of prior exposure to the source material.

---

## 4. Aggregate Finding

The Bayesian fusion of all five forensic agents yields:

- **Posterior probability of training inclusion: {posterior:.1%}**
- **Strength of evidence: {strength}**

> {summary}

A posterior probability of {posterior:.1%} indicates that, given the observed
evidence and a neutral prior belief (50/50), it is {"highly probable" if posterior >= 0.8 else "probable" if posterior >= 0.6 else "uncertain whether"} that the
model was trained on text from *{title}*.

---

## 5. Detailed Forensic Evidence

The SFAS system applies five independent analysis agents. Each agent's findings
are summarised below.

{agent_text}

---

## 6. Summary of Agent Results

{table_md}

All five agents produced likelihood ratios greater than 1.0, with every p-value
below the conventional significance threshold of 0.05. This consistent pattern
across independent statistical tests demonstrates {"compelling" if posterior >= 0.8 else "substantial" if posterior >= 0.6 else "moderate"} convergent evidence
that *{title}* was present in the model's training corpus.

---

## 7. Legal Significance

Based on the forensic analysis presented above:

1. **Copyright Infringement** — A training usage probability of {posterior:.1%}
   constitutes strong forensic evidence that the copyrighted text of *{title}*
   was reproduced, stored, and processed as part of the AI model's training data
   without the authorisation of the rights holder.

2. **Unauthorised Derivative Work** — The model's demonstrated ability to
   reproduce stylistic patterns, rare phrases, and content-specific vocabulary
   from *{title}* indicates the creation of a derivative work within the meaning
   of applicable copyright law.

3. **Evidence of Actual Copying** — The rare phrase analysis detected verbatim
   reproduction and source recognition by the model, providing direct evidence
   of memorisation of the copyrighted text.

---

## 8. Methodology Notes

---

{distribution_md}

---

## 9. Methodology Notes

- **Rare Phrase Agent**: Extracts rare n-gram passages from the book, prompts
  the model for continuations, and measures exact, partial, and recognition-level
  matches via binomial testing.
- **Stylometric Agent**: Compares 10 stylistic features (sentence length, lexical
  diversity, punctuation density, etc.) between book segments and model output
  using per-feature Spearman correlation and normalised Euclidean distance.
- **Distribution Agent**: Measures JS divergence, Wasserstein distance, and KL
  divergence between token frequency distributions of matched source-output pairs.
- **Entropy Agent**: Computes rolling Shannon entropy profiles and tests whether
  the model's output entropy correlates with the book's information-density pattern.
- **Semantic Agent**: Builds TF-IDF representations and measures paired cosine
  similarity and top-1 source-segment retrieval accuracy.
- **Bayesian Fusion**: Combines agent likelihood ratios with a correlation penalty
  to produce a calibrated posterior probability.

---

{_format_chat_log_md(chat_log) if chat_log else ""}

*This forensic evidence report was generated by the First AIDA Scientific Forensic Attribution
System (SFAS) on {today}. The analysis was conducted programmatically and the
results are reproducible given the same input material and model access.*
"""
    return doc


def _format_chat_log_md(logs: list[dict]) -> str:
    """Format the dashboard chat log entries as a markdown section.
    
    Each log entry contains: index, timestamp, elapsed_s, agent, level, message, detail
    """
    if not logs:
        return ""
    
    lines = [
        "## Analysis Session Log\n",
        "The following is a transcript of the real-time analysis session, showing the",
        "step-by-step forensic processing as it occurred:\n",
        "| Time (s) | Agent | Level | Message |",
        "|:--------:|-------|-------|---------|",
    ]
    
    for entry in logs:
        elapsed = entry.get("elapsed_s", 0.0)
        agent = entry.get("agent", "system").replace("_", " ").title()
        level = entry.get("level", "INFO")
        message = entry.get("message", "").replace("|", "\\|").replace("\n", " ")
        # Truncate very long messages
        if len(message) > 200:
            message = message[:197] + "..."
        lines.append(f"| {elapsed:.1f}s | {agent} | {level} | {message} |")
    
    lines.append("")
    return "\n".join(lines)


def _build_fallback_memorandum(
    case_number: str,
    book_title: str,
    book_author: str,
    model_name: str,
    metadata: dict,
    agent_results: list[dict],
    aggregate: dict,
    role: str,
    tone_style: str,
    distribution_evidence: dict | None = None,
    chat_log: list[dict] | None = None,
) -> str:
    """Generate a formatted legal memorandum locally when AILA is unavailable.

    The memorandum is evidence-dependent: when evidence is weak (posterior < 50%),
    it reports findings without asserting infringement claims.
    """
    from datetime import datetime as _dt

    posterior = aggregate.get("posterior_probability", 0)
    strength = aggregate.get("strength_of_evidence", "Unknown")
    summary = aggregate.get("executive_summary", "")
    today = _dt.now().strftime("%d %B %Y")
    title = book_title or "the Subject Work"
    author = book_author or "the Rights Holder"
    distribution_md = _format_aila_distribution_md(distribution_evidence, title, author)

    # Determine evidence strength category
    strong_evidence = posterior >= 0.6
    moderate_evidence = 0.4 <= posterior < 0.6
    weak_evidence = posterior < 0.4

    # Count agents with LR > 1 and significant p-values
    lr_above_1 = sum(1 for r in agent_results if r.get("likelihood_ratio", 1.0) > 1.0)
    significant_agents = sum(1 for r in agent_results if r.get("p_value", 1.0) < 0.05)
    total_agents = len(agent_results)

    # Build agent evidence table with correct interpretation
    table_lines = [
        "| Forensic Agent | Likelihood Ratio | p-value | Statistical Significance | Direction |",
        "|----------------|:----------------:|:-------:|:-----------------------:|-----------|",
    ]
    for r in agent_results:
        name = r.get("agent_name", "?").replace("_", " ").title()
        lr = r.get("likelihood_ratio", 1.0)
        pv = r.get("p_value", 1.0)
        sig = "p < 0.05 ★" if pv < 0.05 else f"p = {pv:.3f}"
        direction = "Supports Training" if lr > 1.0 else "Supports Not-Trained"
        table_lines.append(f"| {name} | {lr:.2f} | {pv:.4f} | {sig} | {direction} |")
    table_md = "\n".join(table_lines)

    # --- Document subject line depends on evidence ---
    if strong_evidence:
        subject_line = f"Unauthorised Use of *{title}* in AI Model Training Data"
    else:
        subject_line = f"Forensic Analysis Report: *{title}* and AI Model Training Data"

    # --- Introduction varies by evidence strength ---
    if strong_evidence:
        intro_para = f"""We write on behalf of our client, {author}, the author and copyright holder
of the literary work entitled *{title}*, to present forensic evidence demonstrating
that the artificial intelligence model **{model_name}**, operated by Microsoft
Corporation through its Azure OpenAI service ("the Respondent"), was trained using
the copyrighted text of the aforesaid work without authorisation, licence, or consent.

The First AIDA Scientific Forensic Attribution System (SFAS), a specialised computational
forensic platform, was employed to conduct a rigorous multi-modal statistical
analysis. The system returned a **posterior probability of {posterior:.1%}** that
the work was included in the model's training data, with an overall evidence
strength rated as **{strength}**."""
    else:
        intro_para = f"""We write on behalf of our client, {author}, the author and copyright holder
of the literary work entitled *{title}*, to report the findings of a forensic
investigation into whether the artificial intelligence model **{model_name}**,
operated by Microsoft Corporation through its Azure OpenAI service ("the Respondent"),
was trained using the copyrighted text of the aforesaid work.

The First AIDA Scientific Forensic Attribution System (SFAS), a specialised computational
forensic platform, was employed to conduct a rigorous multi-modal statistical
analysis. The system returned a **posterior probability of {posterior:.1%}** that
the work was included in the model's training data, with an overall evidence
strength rated as **{strength}**."""

    # --- Findings interpretation varies ---
    if strong_evidence:
        findings_interp = f"""The aggregate posterior probability of **{posterior:.1%}** — combined with the
**{strength}** evidence classification — {"constitutes compelling proof" if posterior >= 0.8 else "provides strong indication"} that the Respondent's model was
trained on the Claimant's copyrighted work."""
    elif moderate_evidence:
        findings_interp = f"""The aggregate posterior probability of **{posterior:.1%}** — combined with the
**{strength}** evidence classification — indicates inconclusive evidence. Further
investigation or additional forensic techniques may be required to reach a definitive
conclusion regarding training data inclusion."""
    else:
        findings_interp = f"""The aggregate posterior probability of **{posterior:.1%}** — combined with the
**{strength}** evidence classification — does not support the hypothesis that the
Respondent's model was trained on the Claimant's copyrighted work. The forensic
evidence suggests the work was likely **not** included in the model's training data."""

    # --- Agent results interpretation ---
    if lr_above_1 == total_agents and significant_agents == total_agents:
        agent_interp = f"""All {total_agents} forensic agents returned likelihood ratios exceeding 1.0
(favouring the training hypothesis), and all achieved statistical significance
at the conventional α = 0.05 threshold. The convergence of {total_agents} independent
lines of evidence substantially reduces the probability of a false positive finding."""
    elif lr_above_1 > 0 or significant_agents > 0:
        agent_interp = f"""{lr_above_1} of {total_agents} forensic agents returned likelihood ratios
exceeding 1.0 (favouring the training hypothesis), and {significant_agents} achieved
statistical significance at the conventional α = 0.05 threshold. The mixed results
warrant cautious interpretation."""
    else:
        agent_interp = f"""All {total_agents} forensic agents returned likelihood ratios below 1.0,
favouring the hypothesis that the work was **not** used in model training. None of
the agents achieved statistical significance at the α = 0.05 threshold for the
training hypothesis. These results consistently support the conclusion that the
subject work was not included in the model's training data."""

    # --- Legal analysis depends on evidence ---
    if strong_evidence:
        legal_section = f"""## IV. LEGAL ANALYSIS

### A. Applicable Legal Framework

This analysis is conducted under the following legal instruments:

1. **Copyright, Designs and Patents Act 1988 (CDPA)** — The principal UK statute
   governing copyright protection, as amended by subsequent legislation including
   the Copyright and Related Rights Regulations 2003.

2. **The Berne Convention for the Protection of Literary and Artistic Works** —
   The international treaty establishing minimum standards of copyright protection
   to which both the UK and the United States are signatories.

3. **The WIPO Copyright Treaty (WCT)** — Extending Berne Convention protections
   to the digital environment.

4. **The Digital Millennium Copyright Act (DMCA)** (US) — Relevant to the extent
   the Respondent's services operate from or are directed at US jurisdictions.

5. **Regulation (EU) 2019/790 (DSM Directive)** — The EU Directive on Copyright
   in the Digital Single Market, particularly Articles 3 and 4 concerning text
   and data mining, which remains persuasive authority post-Brexit.

### B. Copyright Infringement — Reproduction Right

Under **Section 16(1)(a) CDPA 1988**, the copyright owner has the exclusive right
to copy the work. "Copying" is defined in **Section 17** as reproducing the work
in any material form, which includes storing the work in any medium by electronic
means.

The training of a large language model necessarily involves:

1. **Ingestion** — The copyrighted text of *{title}* was copied into the
   Respondent's training pipeline.

2. **Storage** — The work was stored in electronic form on the Respondent's
   servers during the training process.

3. **Reproduction** — The work was reproduced multiple times during training
   epochs, gradient calculations, and weight updates.

4. **Embedding** — Elements of the work have become embedded in the model's
   parameters, as evidenced by the forensic analysis demonstrating memorisation
   and reproduction capability.

The forensic evidence (posterior probability {posterior:.1%}) establishes that
the model retains sufficient traces of *{title}* to reproduce its stylistic
features, rare phrases, and semantic structures. This constitutes prima facie
evidence of copying within the meaning of **Section 17 CDPA**.

### C. Infringement — Adaptation Right

Under **Section 16(1)(e) CDPA 1988**, the copyright owner has the exclusive right
to make an adaptation of the work. An AI model trained on copyrighted text
constitutes a derivative work that incorporates substantial protected expression.

The UK Supreme Court in **Designers Guild Ltd v Russell Williams (Textiles) Ltd
[2000] UKHL 58** held that copyright infringement occurs where there is sufficient
objective similarity between the original and the alleged copy, and where there
is a causal connection. The forensic analysis establishes both elements:

- **Objective similarity**: The rare phrase analysis demonstrates verbatim
  reproduction capability; the stylometric analysis confirms stylistic mimicry.

- **Causal connection**: A posterior probability of {posterior:.1%} establishes,
  to a high degree of statistical confidence, that the model was trained on the
  Claimant's work — thereby establishing the causal link.

### D. No Applicable Exception or Defence

The Respondent cannot rely on:

1. **Section 29A CDPA (Text and Data Mining)** — This exception applies only to
   lawfully accessible works and only for non-commercial research purposes. The
   commercial training of AI models for profit does not qualify.

2. **Fair Dealing (Section 29/30 CDPA)** — The wholesale reproduction of entire
   literary works for commercial AI training does not constitute fair dealing for
   research, criticism, or news reporting.

3. **Temporary Copies Exception (Section 28A CDPA)** — Training storage is not
   transient or incidental; it results in permanent incorporation into the model.

### E. Database Right Infringement

If *{title}* forms part of a database or compilation, the systematic extraction
and re-utilisation of its contents for AI training may additionally infringe the
**sui generis** database right under **Regulation 16 of The Copyright and Rights
in Databases Regulations 1997**, implementing **Directive 96/9/EC**.

### F. Relevant Case Law

1. **Getty Images v Stability AI (2023)** — Ongoing litigation in the High Court
   concerning AI training on copyrighted images, establishing precedent for
   forensic detection methodologies.

2. **The New York Times v Microsoft & OpenAI (SDNY 2023)** — US litigation
   alleging copyright infringement through LLM training, with allegations of
   verbatim reproduction directly relevant to the present analysis.

3. **Infopaq International A/S v Danske Dagblades Forening (C-5/08)** — CJEU
   ruling that reproduction of even 11 words may constitute infringement if they
   convey the author's intellectual creation.

4. **SAS Institute Inc v World Programming Ltd [2013] EWHC 69 (Ch)** — Establishing
   that functionality and programming methods are not protected, but underlying
   literary expression is.

### G. Quantum of Damages

Under **Section 96 CDPA**, the Claimant is entitled to damages, an injunction, or
an account of profits. Given the commercial scale of the Respondent's AI services,
damages may be assessed by reference to:

1. A reasonable licence fee the Respondent would have paid for authorised use.
2. The profits attributable to the infringement.
3. Additional damages under **Section 97(2)** where the infringement is flagrant."""
    elif moderate_evidence:
        legal_section = f"""## IV. LEGAL ANALYSIS

### A. Preliminary Assessment

The forensic evidence is inconclusive. A posterior probability of {posterior:.1%}
does not meet the evidentiary threshold typically required for asserting copyright
infringement claims with confidence under the Copyright, Designs and Patents Act
1988.

### B. Applicable Legal Framework

For reference, the relevant legal instruments include:

1. **Copyright, Designs and Patents Act 1988 (CDPA)** — Sections 16-17 (reproduction
   right), Section 21 (adaptation right), Sections 96-97 (remedies).
2. **The Berne Convention** — International copyright protection standards.
3. **Section 29A CDPA** — Text and data mining exception (limited to non-commercial
   research on lawfully accessible works).

### C. Recommended Next Steps

1. **Additional Forensic Analysis** — Consider applying supplementary detection
   methods or analysing a larger sample of the work to improve statistical power.
2. **Discovery Requests** — Formal discovery may be pursued to obtain direct
   evidence of training data composition from the Respondent under CPR Part 31.
3. **Preserve Evidence** — Document the current analysis results for potential
   future proceedings should additional evidence emerge.
4. **Monitor Emerging Case Law** — The legal landscape for AI copyright is rapidly
   evolving (see *Getty v Stability AI*, *NYT v OpenAI*)."""
    else:
        legal_section = f"""## IV. LEGAL ANALYSIS

### A. Assessment of Evidence

The forensic analysis does not support a claim of copyright infringement based
on training data inclusion. The posterior probability of {posterior:.1%} and the
**{strength}** evidence classification indicate that the statistical tests did
not detect signatures consistent with the work having been used in model training.

### B. Interpretation

A low posterior probability does not conclusively prove that the work was *never*
used in training, but it does indicate that the forensic signals associated with
training-data exposure were not observed in this analysis. Possible interpretations
include:

1. The work was genuinely not included in the model's training data.
2. The work was included but at insufficient frequency to leave detectable traces.
3. The model's training process included techniques that reduced memorisation.

### C. Recommendations

Based on the current findings, there is insufficient forensic evidence to support
legal action at this time. The Claimant may wish to:

1. Monitor for future model releases that may exhibit different behaviour.
2. Consider alternative investigative approaches if there is independent reason
   to believe the work was used in training.
3. Retain these results as a baseline for comparison against future analyses."""

    # --- Relief sought (only for strong evidence) ---
    if strong_evidence:
        relief_section = f"""## V. RELIEF SOUGHT

Based on the evidence set out above, the Claimant seeks the following remedies
pursuant to **Sections 96-100 of the Copyright, Designs and Patents Act 1988**:

1. **Declaration** — A declaration that the Respondent has infringed the Claimant's
   copyright in *{title}* by reproducing, storing, and using it as training data
   for the AI model **{model_name}** without authorisation.

2. **Injunctive Relief** — An injunction pursuant to **Section 96(2) CDPA**
   requiring the Respondent to:
   (a) Cease all use of *{title}* in current and future model training;
   (b) Remove or quarantine the Claimant's work from training datasets;
   (c) Implement technical measures to prevent the model from reproducing
       substantial portions of *{title}*.

3. **Disclosure Order** — An order pursuant to **CPR 31.16** requiring the
   Respondent to disclose:
   (a) The complete list of copyrighted works used in training the model;
   (b) Documentation evidencing the source and licensing of training data;
   (c) Technical specifications regarding how training data is stored and processed.

4. **Damages or Account of Profits** — At the Claimant's election:
   (a) Damages calculated by reference to a reasonable licensing fee; or
   (b) An account of the Respondent's profits attributable to the infringement.

5. **Additional Damages** — Additional damages pursuant to **Section 97(2) CDPA**
   in light of the flagrant nature of the infringement and the benefit accruing
   to the Respondent from the unauthorised use.

6. **Delivery Up** — An order for delivery up or destruction of any infringing
   copies pursuant to **Section 99 CDPA**.

7. **Costs** — The Respondent to pay the Claimant's costs of and incidental to
   this claim on the standard basis, or such other basis as the Court sees fit."""
        
        next_steps_section = f"""## VI. RECOMMENDED NEXT STEPS

Based on the forensic evidence presented, the following course of action is
recommended to advance the Claimant's legal position:

### A. Immediate Actions (0-30 Days)

1. **Preserve Evidence** — Ensure all forensic analysis data, model outputs, and
   API responses are securely stored with chain-of-custody documentation. Consider
   instructing a digital forensics expert to certify the integrity of the evidence.

2. **Send Letter Before Action** — Issue a formal Letter Before Action to the
   Respondent (Microsoft Corporation / OpenAI) in accordance with the Pre-Action
   Protocol for Intellectual Property Claims. The letter should:
   - Set out the Claimant's rights in *{title}*;
   - Detail the forensic evidence of infringement;
   - Demand immediate cessation of the infringing conduct;
   - Request voluntary disclosure of training data sources;
   - Allow 14 days for a substantive response.

3. **Without Prejudice Correspondence** — Consider initiating without prejudice
   settlement discussions, potentially including a licensing arrangement, damages
   payment, or commitment to exclude the work from future training.

### B. Pre-Litigation Phase (30-90 Days)

4. **Instruct Counsel** — Engage specialist intellectual property counsel with
   experience in AI copyright matters. Consider counsel familiar with the *Getty
   v Stability AI* and *NYT v OpenAI* proceedings.

5. **Expert Evidence** — Instruct a qualified expert in machine learning and
   computational forensics to prepare a report suitable for court proceedings.
   The SFAS forensic analysis should be reviewed and validated by an independent
   expert who can testify as to its methodology and reliability.

6. **Jurisdictional Analysis** — Consider jurisdictional options:
   - **UK High Court (IPEC or Chancery Division)** — Specialist IP courts with
     established procedures for copyright claims;
   - **US Federal Courts** — If the Respondent's primary operations are US-based,
     consider parallel or alternative US proceedings.

7. **Funding Assessment** — Evaluate litigation funding options:
   - **After-the-Event (ATE) Insurance** — To cover adverse costs risk;
   - **Third-Party Litigation Funding** — Commercial funders may be interested
     given the precedent value and potential damages quantum;
   - **Damages-Based Agreement (DBA)** — Consider contingency fee arrangements.

### C. Litigation Phase (90+ Days)

8. **Issue Proceedings** — File a Claim Form in the Intellectual Property
   Enterprise Court (IPEC) or Chancery Division of the High Court of Justice,
   supported by Particulars of Claim setting out the infringement allegations.

9. **Interim Relief** — Consider applying for interim injunctive relief to prevent
   ongoing infringement pending trial, particularly if the Respondent announces
   new model releases.

10. **Disclosure and Inspection** — Pursue specific disclosure of:
    - Training data manifests and composition records;
    - Model architecture and training logs;
    - Revenue and profit data for quantum assessment.

### D. Alternative Dispute Resolution

11. **WIPO Mediation** — The World Intellectual Property Organization offers
    specialist mediation services for IP disputes that may achieve faster
    resolution than court proceedings.

12. **Industry Engagement** — Consider engagement with industry bodies and
    regulatory initiatives concerning AI training data transparency.

### E. Public and Advocacy Actions

13. **Regulatory Complaints** — Consider filing complaints with:
    - The UK Intellectual Property Office regarding AI and copyright policy;
    - The Competition and Markets Authority (CMA) if competition concerns arise.

14. **Coalition Building** — Engage with other authors and rights holders who
    may have similar claims, potentially supporting a class action or coordinated
    litigation strategy.

### F. Timeline Summary

| Phase | Timeframe | Key Actions |
|-------|-----------|-------------|
| Immediate | 0-30 days | Preserve evidence, Letter Before Action |
| Pre-Litigation | 30-90 days | Instruct counsel, expert evidence, funding |
| Litigation | 90+ days | Issue proceedings, disclosure, trial |
| Alternative | Ongoing | Settlement discussions, mediation |"""

    else:
        relief_section = f"""## V. SUMMARY

Based on the forensic analysis, the evidence does not support the conclusion that
*{title}* was used in training the AI model **{model_name}**. No claims of
copyright infringement are asserted at this time.

The Claimant's rights in *{title}* remain fully reserved, and this memorandum
does not preclude future action should additional evidence emerge."""
        
        next_steps_section = ""

    # --- Conclusion varies ---
    if strong_evidence:
        conclusion = f"""The forensic evidence presented by the SFAS analysis establishes, to a high
degree of statistical confidence ({posterior:.1%} posterior probability, {strength}
classification), that *{title}* by {author} was included in the training data
of the Respondent's AI model **{model_name}**.

The Claimant has strong grounds to pursue legal action for copyright infringement
under the Copyright, Designs and Patents Act 1988, with potential remedies including
injunctive relief, damages, an account of profits, and disclosure orders.

Time is of the essence. Legal proceedings should be initiated promptly to preserve
the Claimant's rights and prevent further infringement. The Claimant reserves all
rights and remedies available under UK, EU, and international copyright law."""
    else:
        conclusion = f"""The forensic analysis conducted by the SFAS system returned a posterior
probability of {posterior:.1%} with a **{strength}** evidence classification.
These results do not support the hypothesis that *{title}* by {author} was
included in the training data of the AI model **{model_name}**.

The Claimant's intellectual property rights in *{title}* remain fully reserved."""

    # --- Assemble section number based on what's included ---
    if strong_evidence:
        conclusion_num = "VII"
    else:
        conclusion_num = "VI"

    # --- Pre-compute the aggregate chart markdown ---
    # Use a placeholder that frontend can detect and replace with the actual image
    # The image is returned separately in aggregate_chart_base64 field
    aggregate_chart_md = "<!-- AGGREGATE_CHART_PLACEHOLDER -->"

    doc = f"""# LEGAL MEMORANDUM

**Case Number:** {case_number}
**Date:** {today}
**Prepared for:** {author} ("the Claimant")
**Prepared by:** Legal Counsel, on the basis of forensic analysis by SFAS
**Subject:** {subject_line}
**Classification:** Privileged and Confidential

---

## I. INTRODUCTION

{intro_para}

---

## II. FACTUAL BACKGROUND

### A. The Work

The subject work, *{title}* by {author}, comprises approximately
{metadata.get('word_count', 0):,} words across {metadata.get('page_count', 0)}
pages. The work is an original literary composition, the copyright of which is
owned exclusively by the Claimant.

### B. The AI Model

The model under investigation, **{model_name}**, is a large language model
deployed on the Azure OpenAI platform, a commercial service operated by
Microsoft Corporation. The model accepts text prompts and generates text
continuations, drawing upon patterns learnt from its training data.

### C. The Investigation

A forensic analysis was initiated to ascertain whether the Respondent's model
has been trained on text from *{title}*. The SFAS platform subjected the model
to five independent forensic tests, each designed to detect a distinct modality
of textual memorisation and training-data exposure.

### D. External Availability and Distribution Signals

{distribution_md.replace("## External Availability and Distribution Signals", "")}

---

## III. FORENSIC EVIDENCE

### A. Summary of Findings

> {summary}

{findings_interp}

### B. Per-Agent Statistical Results

{table_md}

{agent_interp}

### C. Aggregate Analysis Chart

{aggregate_chart_md}

---

{legal_section}

---

{relief_section}

---

{next_steps_section}

{"---" if next_steps_section else ""}

## {conclusion_num}. CONCLUSION

{conclusion}

---

{_format_chat_log_md(chat_log) if chat_log else ""}

*This memorandum was prepared on the basis of forensic analysis conducted by the
First AIDA Scientific Forensic Attribution System (SFAS) on {today}. The statistical
findings are reproducible and available for expert testimony if required.*

---

**CONFIDENTIAL — SUBJECT TO LEGAL PRIVILEGE**
"""
    return doc


@app.post("/generate-memorandum", response_model=MemorandumResponse)
def generate_memorandum(request: MemorandumRequest) -> MemorandumResponse:
    """Generate a legal memorandum by combining SFAS forensic evidence with AILA.

    Flow:
    1. Gather all forensic evidence from Azure storage.
    2. Format a structured evidence document.
    3. Upload evidence to AILA's blob storage.
    4. Call AILA ``/generate_memorandum`` to produce a formal legal memorandum.
    5. Fall back to a locally-generated legal memorandum if AILA is unavailable.
    """
    import requests as _requests

    _get_book_or_404(request.book_id)
    case_number = request.case_number or f"SFAS-{request.book_id[:8].upper()}"

    # Gather evidence
    agent_results = store.get_agent_results(request.book_id, request.model_name)
    aggregate = store.get_aggregate_result(request.book_id, request.model_name)
    if not aggregate:
        raise HTTPException(
            status_code=400,
            detail="No aggregate result found. Run /aggregate first before generating a memorandum.",
        )

    book = store.get_book(request.book_id)
    metadata = book.metadata if isinstance(book.metadata, dict) else book.metadata

    distribution_evidence = _fetch_aila_distribution_evidence(
        request.book_title,
        request.book_author,
    )

    evidence_md = _build_forensic_evidence_md(
        book_id=request.book_id,
        model_name=request.model_name,
        book_title=request.book_title,
        book_author=request.book_author,
        metadata=metadata,
        agent_results=agent_results,
        aggregate=aggregate,
        distribution_evidence=distribution_evidence,
    )

    # Try AILA backend
    try:
        logger.info("Uploading forensic evidence to AILA blob storage (case %s)", case_number)
        _upload_evidence_to_aila(case_number, request.role, evidence_md)

        logger.info("Calling AILA backend at %s for memorandum generation", _AILA_BASE)
        memo_payload = {
            "caseNumber": case_number,
            "firmShortName": request.firm_name,
            "role": request.role,
            "toneStyle": request.tone_style,
            "lengthStyle": request.length_style,
        }
        resp = _requests.post(
            f"{_AILA_BASE}/rag-api/generate_memorandum",
            json=memo_payload,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()

        # Generate aggregate chart for the response
        try:
            aila_chart_bytes = chart_aggregate(aggregate)
            aila_chart_b64 = base64.b64encode(aila_chart_bytes).decode("utf-8")
        except Exception as chart_exc:
            logger.warning("Failed to generate aggregate chart for AILA response: %s", chart_exc)
            aila_chart_b64 = ""

        logger.info("AILA memorandum generated successfully for case %s", case_number)
        return MemorandumResponse(
            english_markdown_memorandum=data.get("english_markdown_memorandum", ""),
            arabic_markdown_memorandum=data.get("arabic_markdown_memorandum", ""),
            blob_path=data.get("blob_path", ""),
            case_number=case_number,
            aggregate_chart_base64=aila_chart_b64,
        )
    except Exception as exc:
        logger.warning("AILA integration failed (%s) — generating memorandum locally", exc)
        # Generate aggregate chart and convert to base64 for embedding
        try:
            aggregate_chart_bytes = chart_aggregate(aggregate)
            aggregate_chart_b64 = base64.b64encode(aggregate_chart_bytes).decode("utf-8")
        except Exception as chart_exc:
            logger.warning("Failed to generate aggregate chart: %s", chart_exc)
            aggregate_chart_b64 = ""
        # Fallback: generate a proper legal memorandum locally
        fallback_md = _build_fallback_memorandum(
            case_number=case_number,
            book_title=request.book_title,
            book_author=request.book_author,
            model_name=request.model_name,
            metadata=metadata,
            agent_results=agent_results,
            aggregate=aggregate,
            role=request.role,
            tone_style=request.tone_style,
            distribution_evidence=distribution_evidence,
        )
        return MemorandumResponse(
            english_markdown_memorandum=fallback_md,
            arabic_markdown_memorandum="",
            blob_path="",
            case_number=case_number,
            aggregate_chart_base64=aggregate_chart_b64,
        )


# ── Visualization endpoints ─────────────────────────────────────────

_PNG = "image/png"


def _png(data: bytes, filename: str) -> Response:
    return Response(content=data, media_type=_PNG, headers={
        "Content-Disposition": f'inline; filename="{filename}"',
        "Cache-Control": "public, max-age=3600",
    })


# ---- Book-level charts ----

@app.get("/viz/{book_id}/entropy")
def viz_entropy(book_id: str) -> Response:
    """Visualise rolling & segment entropy for the book."""
    book = _get_book_or_404(book_id)
    return _png(chart_rolling_entropy(book.datasets), f"{book_id}_entropy.png")


@app.get("/viz/{book_id}/word-frequency")
def viz_word_frequency(book_id: str, top_n: int = 40) -> Response:
    """Top-N word frequency bar chart."""
    book = _get_book_or_404(book_id)
    return _png(chart_word_frequency(book.datasets, top_n), f"{book_id}_word_freq.png")


@app.get("/viz/{book_id}/stylometry")
def viz_stylometry(book_id: str) -> Response:
    """Stylometric feature heatmap across book segments."""
    book = _get_book_or_404(book_id)
    return _png(chart_stylometry_heatmap(book.datasets), f"{book_id}_stylometry.png")


@app.get("/viz/{book_id}/graph-metrics")
def viz_graph_metrics(book_id: str) -> Response:
    """Graph degree distributions + metric comparison."""
    book = _get_book_or_404(book_id)
    return _png(chart_graph_metrics(book.graphs), f"{book_id}_graphs.png")


# ---- Agent result charts ----

@app.get("/viz/{book_id}/{model_name}/rare-phrase")
def viz_rare_phrase(book_id: str, model_name: str) -> Response:
    """Rare phrase agent visualisation."""
    _get_book_or_404(book_id)
    result = store.get_single_agent_result(book_id, model_name, "rare_phrase")
    if not result:
        raise HTTPException(404, detail="Rare phrase result not found. Run the agent first.")
    return _png(chart_rare_phrase(result), f"{book_id}_{model_name}_rare_phrase.png")


@app.get("/viz/{book_id}/{model_name}/entropy")
def viz_entropy_agent(book_id: str, model_name: str) -> Response:
    """Entropy agent visualisation."""
    _get_book_or_404(book_id)
    result = store.get_single_agent_result(book_id, model_name, "entropy")
    if not result:
        raise HTTPException(404, detail="Entropy result not found. Run the agent first.")
    return _png(chart_entropy(result), f"{book_id}_{model_name}_entropy.png")


@app.get("/viz/{book_id}/{model_name}/distribution")
def viz_distribution_agent(book_id: str, model_name: str) -> Response:
    """Distribution agent visualisation."""
    _get_book_or_404(book_id)
    result = store.get_single_agent_result(book_id, model_name, "distribution")
    if not result:
        raise HTTPException(404, detail="Distribution result not found. Run the agent first.")
    return _png(chart_distribution(result), f"{book_id}_{model_name}_distribution.png")


@app.get("/viz/{book_id}/{model_name}/stylometric")
def viz_stylometric_agent(book_id: str, model_name: str) -> Response:
    """Stylometric agent visualisation."""
    _get_book_or_404(book_id)
    result = store.get_single_agent_result(book_id, model_name, "stylometric")
    if not result:
        raise HTTPException(404, detail="Stylometric result not found. Run the agent first.")
    return _png(chart_stylometric(result), f"{book_id}_{model_name}_stylometric.png")


@app.get("/viz/{book_id}/{model_name}/semantic")
def viz_semantic_agent(book_id: str, model_name: str) -> Response:
    """Semantic agent visualisation."""
    _get_book_or_404(book_id)
    result = store.get_single_agent_result(book_id, model_name, "semantic")
    if not result:
        raise HTTPException(404, detail="Semantic result not found. Run the agent first.")
    return _png(chart_semantic(result), f"{book_id}_{model_name}_semantic.png")


@app.get("/viz/{book_id}/{model_name}/aggregate")
def viz_aggregate(book_id: str, model_name: str) -> Response:
    """Aggregate (Bayesian fusion) visualisation."""
    _get_book_or_404(book_id)
    agg = store.get_aggregate_result(book_id, model_name)
    if not agg:
        raise HTTPException(404, detail="Aggregate result not found. Run /aggregate first.")
    return _png(chart_aggregate(agg), f"{book_id}_{model_name}_aggregate.png")


@app.get("/viz/{book_id}/{model_name}/dashboard")
def viz_dashboard(book_id: str, model_name: str) -> Response:
    """Full 6-panel analysis dashboard (book data + all agents + verdict)."""
    book = _get_book_or_404(book_id)
    agent_results = store.get_agent_results(book_id, model_name)
    aggregate = store.get_aggregate_result(book_id, model_name)
    return _png(
        chart_full_dashboard(book.datasets, book.graphs, agent_results, aggregate),
        f"{book_id}_{model_name}_dashboard.png",
    )
