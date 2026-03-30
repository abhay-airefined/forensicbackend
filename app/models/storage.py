from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.data.tables import TableServiceClient
from azure.storage.blob import BlobServiceClient

from app.config import settings

logger = logging.getLogger(__name__)

# ── Upload configuration ─────────────────────────────────────────
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB chunks
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, will exponential backoff
UPLOAD_TIMEOUT = 300  # 5 minutes for overall upload


# ── JSON helper ──────────────────────────────────────────────────
def _json_default(obj: object) -> object:
    """Handle numpy / datetime types during JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


# ── BookRecord (unchanged interface) ─────────────────────────────
@dataclass
class BookRecord:
    book_id: str
    metadata: dict[str, Any]
    raw_text: str
    normalized_text: str
    sentences: list[str]
    tokens: list[str]
    segments: list[str]
    segment_tokens: list[list[str]]
    datasets: dict[str, Any] = field(default_factory=dict)
    graphs: dict[str, Any] = field(default_factory=dict)
    original_filename: str = ""


# ── Azure-backed store ───────────────────────────────────────────
class AzureStore:
    """Persists book data in Azure Blob Storage and metadata /
    results in Azure Table Storage."""

    def __init__(self) -> None:
        conn = settings.azure_storage_connection_string
        if not conn:
            raise RuntimeError(
                "SFAS_AZURE_STORAGE_CONNECTION_STRING environment variable "
                "is required.  See .env.example for details."
            )
        self._blob_svc = BlobServiceClient.from_connection_string(conn)
        self._table_svc = TableServiceClient.from_connection_string(conn)
        self._container = settings.azure_blob_container
        self._books_tbl = settings.azure_table_books
        self._results_tbl = settings.azure_table_results
        self._ensure_resources()
        self._cache: dict[str, BookRecord] = {}

    # ---- bootstrap -------------------------------------------------
    def _ensure_resources(self) -> None:
        try:
            self._blob_svc.create_container(self._container)
            logger.info("Created blob container '%s'", self._container)
        except ResourceExistsError:
            pass
        for name in (self._books_tbl, self._results_tbl):
            try:
                self._table_svc.create_table(name)
                logger.info("Created table '%s'", name)
            except ResourceExistsError:
                pass

    # ---- low-level helpers -----------------------------------------
    def _upload(self, blob_path: str, data: str | bytes) -> None:
        """Upload data to blob storage with retry logic and chunking for large payloads."""
        payload = data.encode("utf-8") if isinstance(data, str) else data
        payload_size_mb = len(payload) / (1024 * 1024)
        
        logger.info(
            "Starting upload: %s (size: %.2f MB, chunks: %d)",
            blob_path,
            payload_size_mb,
            (len(payload) + CHUNK_SIZE - 1) // CHUNK_SIZE,
        )
        
        client = self._blob_svc.get_blob_client(self._container, blob_path)
        
        # For small payloads, use direct upload
        if len(payload) <= CHUNK_SIZE:
            self._upload_with_retry(client, payload, blob_path)
            logger.info("Upload completed: %s", blob_path)
            return
        
        # For large payloads, use chunked upload
        logger.info("Large payload detected, using chunked upload for %s", blob_path)
        self._upload_chunked(client, payload, blob_path)
        logger.info("Chunked upload completed: %s", blob_path)

    def _upload_with_retry(self, client, payload: bytes, blob_path: str) -> None:
        """Upload payload with exponential backoff retry logic."""
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(
                    "Upload attempt %d/%d for %s (size: %d bytes)",
                    attempt + 1,
                    MAX_RETRIES,
                    blob_path,
                    len(payload),
                )
                client.upload_blob(payload, overwrite=True, timeout=UPLOAD_TIMEOUT)
                logger.debug("Upload attempt %d succeeded for %s", attempt + 1, blob_path)
                return
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "Upload attempt %d failed for %s: %s. Retrying in %ds...",
                        attempt + 1,
                        blob_path,
                        str(e)[:100],
                        wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        "Upload failed after %d attempts for %s: %s",
                        MAX_RETRIES,
                        blob_path,
                        str(e),
                    )
        
        raise last_error

    def _upload_chunked(self, client, payload: bytes, blob_path: str) -> None:
        """Upload large payload in chunks to avoid timeouts."""
        num_chunks = (len(payload) + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        logger.info(
            "Uploading %s in %d chunks of %d bytes each",
            blob_path,
            num_chunks,
            CHUNK_SIZE,
        )
        
        # First, initialize staged blocks upload
        block_ids = []
        
        for chunk_num in range(num_chunks):
            start_idx = chunk_num * CHUNK_SIZE
            end_idx = min((chunk_num + 1) * CHUNK_SIZE, len(payload))
            chunk = payload[start_idx:end_idx]
            
            chunk_num_padded = str(chunk_num).zfill(6)
            block_id = f"block_{chunk_num_padded}_{int(time.time())}"
            
            logger.debug(
                "Uploading chunk %d/%d for %s (bytes %d-%d, block_id: %s)",
                chunk_num + 1,
                num_chunks,
                blob_path,
                start_idx,
                end_idx,
                block_id,
            )
            
            try:
                # Stage the block with retry
                for attempt in range(MAX_RETRIES):
                    try:
                        client.stage_block(block_id, chunk)
                        logger.debug(
                            "Successfully staged block %d/%d for %s",
                            chunk_num + 1,
                            num_chunks,
                            blob_path,
                        )
                        block_ids.append(block_id)
                        break
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:
                            wait_time = RETRY_DELAY * (2 ** attempt)
                            logger.warning(
                                "Chunk %d staging failed (attempt %d/%d): %s. Retrying in %ds...",
                                chunk_num + 1,
                                attempt + 1,
                                MAX_RETRIES,
                                str(e)[:100],
                                wait_time,
                            )
                            time.sleep(wait_time)
                        else:
                            logger.error(
                                "Chunk %d staging failed after %d attempts: %s",
                                chunk_num + 1,
                                MAX_RETRIES,
                                str(e),
                            )
                            raise
            except Exception as e:
                logger.error(
                    "Failed to upload chunk %d/%d for %s: %s",
                    chunk_num + 1,
                    num_chunks,
                    blob_path,
                    str(e),
                )
                raise
        
        # Commit all blocks
        logger.info(
            "Committing %d blocks for %s",
            len(block_ids),
            blob_path,
        )
        
        try:
            client.commit_block_list(block_ids)
            logger.info("All blocks committed successfully for %s", blob_path)
        except Exception as e:
            logger.error("Failed to commit blocks for %s: %s", blob_path, str(e))
            raise


    def _download(self, blob_path: str) -> bytes:
        client = self._blob_svc.get_blob_client(self._container, blob_path)
        return client.download_blob().readall()

    def _download_text(self, blob_path: str) -> str:
        return self._download(blob_path).decode("utf-8")

    def _download_json(self, blob_path: str) -> Any:
        return json.loads(self._download_text(blob_path))

    def _dumps(self, obj: Any) -> str:
        return json.dumps(obj, default=_json_default)

    # ---- books -----------------------------------------------------
    def upsert_book(
        self,
        record: BookRecord,
        original_file: bytes | None = None,
        original_filename: str = "",
    ) -> None:
        bid = record.book_id
        record.original_filename = original_filename
        
        logger.info("=" * 80)
        logger.info("Starting upsert_book for: %s", bid)
        logger.info("=" * 80)

        # Large payloads → Blob Storage
        try:
            logger.info("Uploading raw_text.txt...")
            self._upload(f"{bid}/raw_text.txt", record.raw_text)
            
            logger.info("Uploading normalized_text.txt...")
            self._upload(f"{bid}/normalized_text.txt", record.normalized_text)
            
            logger.info("Uploading tokens.json...")
            self._upload(f"{bid}/tokens.json", self._dumps(record.tokens))
            
            logger.info("Uploading sentences.json...")
            self._upload(f"{bid}/sentences.json", self._dumps(record.sentences))
            
            logger.info("Uploading segments.json...")
            self._upload(f"{bid}/segments.json", self._dumps(record.segments))
            
            logger.info("Uploading segment_tokens.json...")
            self._upload(f"{bid}/segment_tokens.json", self._dumps(record.segment_tokens))
            
            logger.info("Uploading datasets.json (size: %.2f MB)...", 
                       len(self._dumps(record.datasets)) / (1024 * 1024))
            self._upload(f"{bid}/datasets.json", self._dumps(record.datasets))
            
            logger.info("Uploading graphs.json (size: %.2f MB)...", 
                       len(self._dumps(record.graphs)) / (1024 * 1024))
            self._upload(f"{bid}/graphs.json", self._dumps(record.graphs))

            if original_file and original_filename:
                logger.info(
                    "Uploading original file: %s (size: %.2f MB)",
                    original_filename,
                    len(original_file) / (1024 * 1024),
                )
                self._upload(f"{bid}/original/{original_filename}", original_file)
        except Exception as e:
            logger.error("Failed to upload book data for %s: %s", bid, str(e))
            raise

        # Metadata → Azure Table Storage
        logger.info("Uploading metadata to table storage...")
        table = self._table_svc.get_table_client(self._books_tbl)
        entity = {
            "PartitionKey": "book",
            "RowKey": bid,
            "sha256": record.metadata.get("sha256", ""),
            "word_count": record.metadata.get("word_count", 0),
            "sentence_count": record.metadata.get("sentence_count", 0),
            "token_count": record.metadata.get("token_count", 0),
            "page_count": record.metadata.get("page_count", 0),
            "extraction_timestamp": str(
                record.metadata.get("extraction_timestamp", "")
            ),
            "original_filename": original_filename,
        }
        table.upsert_entity(entity)
        self._cache[bid] = record
        logger.info("=" * 80)
        logger.info("Book %s stored successfully", bid)
        logger.info("=" * 80)

    def get_book(self, book_id: str) -> BookRecord:
        if book_id in self._cache:
            return self._cache[book_id]

        table = self._table_svc.get_table_client(self._books_tbl)
        try:
            entity = table.get_entity("book", book_id)
        except ResourceNotFoundError:
            raise KeyError(f"book_id '{book_id}' not found")

        record = BookRecord(
            book_id=book_id,
            metadata={
                "sha256": entity.get("sha256", ""),
                "word_count": entity.get("word_count", 0),
                "sentence_count": entity.get("sentence_count", 0),
                "token_count": entity.get("token_count", 0),
                "page_count": entity.get("page_count", 0),
                "extraction_timestamp": entity.get("extraction_timestamp", ""),
            },
            raw_text=self._download_text(f"{book_id}/raw_text.txt"),
            normalized_text=self._download_text(f"{book_id}/normalized_text.txt"),
            tokens=self._download_json(f"{book_id}/tokens.json"),
            sentences=self._download_json(f"{book_id}/sentences.json"),
            segments=self._download_json(f"{book_id}/segments.json"),
            segment_tokens=self._download_json(f"{book_id}/segment_tokens.json"),
            datasets=self._download_json(f"{book_id}/datasets.json"),
            graphs=self._download_json(f"{book_id}/graphs.json"),
            original_filename=entity.get("original_filename", ""),
        )
        self._cache[book_id] = record
        return record

    def has_book(self, book_id: str) -> bool:
        if book_id in self._cache:
            return True
        table = self._table_svc.get_table_client(self._books_tbl)
        try:
            table.get_entity("book", book_id)
            return True
        except ResourceNotFoundError:
            return False

    def get_book_metadata(self, book_id: str) -> dict[str, Any]:
        table = self._table_svc.get_table_client(self._books_tbl)
        try:
            entity = table.get_entity("book", book_id)
        except ResourceNotFoundError:
            raise KeyError(f"book_id '{book_id}' not found")
        return {
            "sha256": entity.get("sha256", ""),
            "word_count": int(entity.get("word_count", 0)),
            "sentence_count": int(entity.get("sentence_count", 0)),
            "token_count": int(entity.get("token_count", 0)),
            "page_count": int(entity.get("page_count", 0)),
            "extraction_timestamp": entity.get("extraction_timestamp", ""),
        }

    def find_book_by_sha(self, sha256: str) -> str | None:
        table = self._table_svc.get_table_client(self._books_tbl)
        entities = table.query_entities(
            query_filter=f"PartitionKey eq 'book' and sha256 eq '{sha256}'"
        )
        for ent in entities:
            row = ent.get("RowKey")
            if row:
                return str(row)
        return None

    def list_books(self, limit: int = 100, query: str | None = None) -> list[dict[str, Any]]:
        table = self._table_svc.get_table_client(self._books_tbl)
        entities = list(table.query_entities("PartitionKey eq 'book'"))

        q = (query or "").strip().lower()
        rows: list[dict[str, Any]] = []
        for ent in entities:
            row_key = str(ent.get("RowKey", ""))
            filename = str(ent.get("original_filename", ""))
            if q and q not in row_key.lower() and q not in filename.lower():
                continue
            rows.append(
                {
                    "book_id": row_key,
                    "original_filename": filename,
                    "metadata": {
                        "sha256": ent.get("sha256", ""),
                        "word_count": int(ent.get("word_count", 0)),
                        "sentence_count": int(ent.get("sentence_count", 0)),
                        "token_count": int(ent.get("token_count", 0)),
                        "page_count": int(ent.get("page_count", 0)),
                        "extraction_timestamp": ent.get("extraction_timestamp", ""),
                    },
                }
            )

        def _parse_ts(item: dict[str, Any]) -> datetime:
            ts = str(item["metadata"].get("extraction_timestamp", ""))
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                return datetime.min

        rows.sort(key=_parse_ts, reverse=True)
        return rows[: max(1, limit)]

    # ---- agent / aggregate results ---------------------------------
    def store_agent_result(
        self, book_id: str, model_name: str, result: dict[str, Any]
    ) -> str:
        agent_name = result.get("agent_name", "unknown")
        blob_path = f"{book_id}/results/{model_name}/{agent_name}.json"
        self._upload(blob_path, self._dumps(result))

        table = self._table_svc.get_table_client(self._results_tbl)
        entity = {
            "PartitionKey": book_id,
            "RowKey": f"{model_name}__{agent_name}",
            "model_name": model_name,
            "agent_name": agent_name,
            "blob_path": blob_path,
            "p_value": float(result.get("p_value", 0.0)),
            "likelihood_ratio": float(result.get("likelihood_ratio", 0.0)),
            "evidence_direction": str(result.get("evidence_direction", "")),
        }
        table.upsert_entity(entity)
        return blob_path

    def store_aggregate_result(
        self, book_id: str, model_name: str, result: dict[str, Any]
    ) -> str:
        blob_path = f"{book_id}/results/{model_name}/aggregate.json"
        self._upload(blob_path, self._dumps(result))

        table = self._table_svc.get_table_client(self._results_tbl)
        entity = {
            "PartitionKey": book_id,
            "RowKey": f"{model_name}__aggregate",
            "model_name": model_name,
            "agent_name": "aggregate",
            "blob_path": blob_path,
            "posterior_probability": float(
                result.get("posterior_probability", 0.0)
            ),
            "strength_of_evidence": str(
                result.get("strength_of_evidence", "")
            ),
        }
        table.upsert_entity(entity)
        return blob_path

    def get_agent_results(
        self, book_id: str, model_name: str
    ) -> list[dict[str, Any]]:
        """Return all stored agent results for a book / model pair."""
        table = self._table_svc.get_table_client(self._results_tbl)
        entities = list(
            table.query_entities(f"PartitionKey eq '{book_id}'")
        )
        results: list[dict[str, Any]] = []
        for ent in entities:
            if (
                ent.get("model_name") == model_name
                and ent.get("agent_name") != "aggregate"
            ):
                try:
                    results.append(self._download_json(ent["blob_path"]))
                except Exception:
                    logger.warning(
                        "Could not load result blob: %s", ent.get("blob_path")
                    )
        return results

    def get_all_agent_results_summary(self, book_id: str) -> dict[str, dict[str, dict[str, Any]]]:
        """Return lightweight summary of all stored non-aggregate agent results for a book.

        Shape:
        {
            model_name: {
                agent_name: {
                    p_value: float,
                    likelihood_ratio: float,
                    evidence_direction: str,
                }
            }
        }
        """
        table = self._table_svc.get_table_client(self._results_tbl)
        entities = list(table.query_entities(f"PartitionKey eq '{book_id}'"))

        summary: dict[str, dict[str, dict[str, Any]]] = {}
        for ent in entities:
            model_name = str(ent.get("model_name", "")).strip()
            agent_name = str(ent.get("agent_name", "")).strip()
            if not model_name or not agent_name or agent_name == "aggregate":
                continue

            model_bucket = summary.setdefault(model_name, {})
            model_bucket[agent_name] = {
                "p_value": float(ent.get("p_value", 0.0)),
                "likelihood_ratio": float(ent.get("likelihood_ratio", 0.0)),
                "evidence_direction": str(ent.get("evidence_direction", "")),
            }

        return summary

    # ---- download helpers ------------------------------------------
    def get_datasets_bytes(self, book_id: str) -> bytes:
        return self._download(f"{book_id}/datasets.json")

    def get_graphs_bytes(self, book_id: str) -> bytes:
        return self._download(f"{book_id}/graphs.json")

    def get_original_file(self, book_id: str) -> tuple[bytes, str]:
        table = self._table_svc.get_table_client(self._books_tbl)
        try:
            entity = table.get_entity("book", book_id)
        except ResourceNotFoundError:
            raise KeyError(f"book_id '{book_id}' not found")
        filename = entity.get("original_filename", "uploaded_file")
        return self._download(f"{book_id}/original/{filename}"), filename

    def get_single_agent_result(
        self, book_id: str, model_name: str, agent_name: str
    ) -> dict[str, Any] | None:
        """Return a single agent result dict, or None if it doesn't exist."""
        blob_path = f"{book_id}/results/{model_name}/{agent_name}.json"
        try:
            return self._download_json(blob_path)
        except Exception:
            return None

    def get_aggregate_result(
        self, book_id: str, model_name: str
    ) -> dict[str, Any] | None:
        """Return the aggregate result dict, or None."""
        blob_path = f"{book_id}/results/{model_name}/aggregate.json"
        try:
            return self._download_json(blob_path)
        except Exception:
            return None


store = AzureStore()
