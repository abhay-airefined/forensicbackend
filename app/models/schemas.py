from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BookMetadata(BaseModel):
    sha256: str
    word_count: int
    sentence_count: int
    token_count: int
    page_count: int
    extraction_timestamp: datetime


class UploadBookResponse(BaseModel):
    book_id: str
    metadata: BookMetadata
    reused_existing: bool = False


class StoredBookSummary(BaseModel):
    book_id: str
    original_filename: str = ""
    metadata: BookMetadata


class BookDetailsResponse(BaseModel):
    book_id: str
    original_filename: str = ""
    metadata: BookMetadata
    analysis_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of past analyses: {model_name: {agent_name: {p_value, likelihood_ratio, evidence_direction}}}",
    )
    total_analyses: int = 0


class AgentRequest(BaseModel):
    book_id: str
    model_name: str
    sample_count: int = Field(default=20, ge=5, le=200)


class AgentResponse(BaseModel):
    agent_name: str
    hypothesis_test: dict[str, Any]
    metrics: dict[str, Any]
    p_value: float
    likelihood_ratio: float
    log_likelihood_ratio: float
    evidence_direction: str


class AggregateRequest(BaseModel):
    book_id: str
    model_name: str
    prior_probability: float = Field(default=0.5, ge=1e-6, le=1 - 1e-6)


class AggregateResponse(BaseModel):
    posterior_probability: float
    log_likelihood_ratio: float
    strength_of_evidence: str
    agent_breakdown: list[dict[str, Any]]
    executive_summary: str


class MemorandumRequest(BaseModel):
    book_id: str
    model_name: str
    case_number: str = Field(default="", description="Unique case identifier (auto-generated if blank)")
    firm_name: str = Field(default="forensic-legal", description="Organisation / client identifier")
    role: str = Field(default="plaintiff", pattern="^(plaintiff|defendant)$")
    tone_style: str = Field(default="assertive", pattern="^(assertive|conciliatory)$")
    length_style: str = Field(default="detailed", pattern="^(detailed|concise)$")
    book_title: str = Field(default="", description="Title of the analysed book")
    book_author: str = Field(default="", description="Author of the analysed book")
    include_chat_log: bool = Field(default=True, description="Include dashboard chat log in the memorandum")


class MemorandumResponse(BaseModel):
    english_markdown_memorandum: str
    arabic_markdown_memorandum: str = ""
    blob_path: str = ""
    case_number: str = ""
    aggregate_chart_base64: str = Field(default="", description="Base64-encoded PNG of the aggregate analysis chart")
