from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Scientific Forensic Attribution System"
    app_version: str = "2.0.0"
    default_segments: int = 50
    min_segment_tokens: int = 150
    rare_ngram_percentile: float = 10.0
    lr_min: float = 1e-6
    lr_max: float = 1e6
    bootstrap_iterations: int = 400
    permutation_iterations: int = 300
    random_seed: int = 42
    model_max_outputs: int = 40
    rare_phrase_max_attempts: int = 5
    rare_phrase_prompt_words: int = 16
    rare_phrase_expected_words: int = 4
    rare_phrase_min_sincere_words: int = 3
    rare_phrase_min_expected_alpha_tokens: int = 3
    rare_phrase_min_prompt_jaccard_distance: float = 0.35

    # ── Simulation mode (for demos) ──────────────────────────────
    # Set SFAS_SIMULATION_MODE=true to use pre-crafted results
    # instead of real model calls.  Easily removable post-demo.
    simulation_mode: bool = False

    # ── Azure OpenAI ──────────────────────────────────────────────
    azure_openai_endpoint: str =
    azure_openai_api_key: str = 
    azure_openai_api_version: str = "2025-01-01-preview"
    azure_openai_deployment: str = "gpt-4.1-test"  # default deployment name

    # ── Azure Storage (Blob + Table) ─────────────────────────────
    azure_storage_connection_string: str =
    azure_blob_container: str = "sfas-data"
    azure_table_books: str = "sfasbooks"
    azure_table_results: str = "sfasresults"

    model_config = SettingsConfigDict(
        env_prefix="SFAS_",
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
