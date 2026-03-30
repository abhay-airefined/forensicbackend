# Scientific Forensic Attribution System (SFAS)

Scientific Forensic Attribution System (SFAS) is a FastAPI backend for forensic training-data attribution. It estimates whether an uploaded book was used to train a specified AI model using **internal-control methodology only**.

## Features

- Upload PDF or DOCX books.
- Internal baseline generation by segmenting the same book (no external control book).
- Automatic dataset construction:
  - Unigrams, bigrams, 5-grams, 20-grams.
  - Stylometric metrics.
  - Entropy profiles.
  - Word co-occurrence + sentence similarity graph metrics.
- Five independent forensic agents with hypothesis testing:
  - Rare phrase regeneration.
  - Stylometric similarity.
  - Distribution distance.
  - Entropy drift.
  - Semantic similarity.
- Bayesian fusion endpoint with correlation penalty and legal/statistical guardrails.
- **Azure OpenAI** integration for real model-output generation.
- **Azure Blob Storage** for persisting uploaded files, datasets, graphs, and agent results.
- **Azure Table Storage** for book metadata and result indices.
- Download endpoints for datasets, graphs, and original files.

## Prerequisites

| Resource | Purpose |
|---|---|
| Azure OpenAI Service | Generate model text continuations |
| Azure Storage Account | Blob containers + Table Storage |

You will need:

1. An **Azure OpenAI** resource with at least one chat-completion deployment (e.g. `gpt-4o`).
2. An **Azure Storage Account** with Blob and Table services enabled.

## Installation

```bash
cd sfas
python3.11 -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Copy the example environment file and fill in your Azure credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# Azure OpenAI
SFAS_AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
SFAS_AZURE_OPENAI_API_KEY=<your-api-key>
SFAS_AZURE_OPENAI_API_VERSION=2024-12-01-preview
SFAS_AZURE_OPENAI_DEPLOYMENT=<deployment-name>

# Azure Storage (Blob + Table)
SFAS_AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net
SFAS_AZURE_BLOB_CONTAINER=sfas-data
SFAS_AZURE_TABLE_BOOKS=sfasbooks
SFAS_AZURE_TABLE_RESULTS=sfasresults
```

On first start the application automatically creates the blob container and tables if they do not exist.

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI:

- `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## API Usage

### 1) Upload book

```bash
curl -X POST "http://localhost:8000/upload-book" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/book.pdf"
```

Returns:

```json
{
  "book_id": "...",
  "metadata": {
    "sha256": "...",
    "word_count": 120000,
    "sentence_count": 6500,
    "token_count": 145000,
    "page_count": 390,
    "extraction_timestamp": "2026-01-01T00:00:00Z"
  }
}
```

The original file, extracted datasets, and graph metrics are automatically stored in **Azure Blob Storage**. Book metadata is indexed in **Azure Table Storage**.

### 2) Run an agent

```bash
curl -X POST "http://localhost:8000/agents/rare-phrase" \
  -H "Content-Type: application/json" \
  -d '{"book_id":"<BOOK_ID>","model_name":"<DEPLOYMENT_NAME>","sample_count":20}'
```

> **Note:** `model_name` must match an Azure OpenAI deployment name (e.g. `gpt-4o`).

Available agent endpoints:

- `/agents/rare-phrase`
- `/agents/stylometric`
- `/agents/distribution`
- `/agents/entropy`
- `/agents/semantic`

Each agent result is persisted to Azure Blob Storage and indexed in Azure Table Storage.

### 3) Aggregate forensic conclusion

```bash
curl -X POST "http://localhost:8000/aggregate" \
  -H "Content-Type: application/json" \
  -d '{"book_id":"<BOOK_ID>","model_name":"<DEPLOYMENT_NAME>","prior_probability":0.5}'
```

Returns:

```json
{
  "posterior_probability": 0.71,
  "log_likelihood_ratio": 1.42,
  "strength_of_evidence": "Strong",
  "agent_breakdown": [...],
  "executive_summary": "Based on the statistical analysis, there is sufficient evidence to conclude that the uploaded book was used in training the specified AI model."
}
```

### 4) Download datasets, graphs, and files

| Endpoint | Description |
|---|---|
| `GET /books/{book_id}/datasets` | Download n-gram, stylometric, and entropy datasets (JSON) |
| `GET /books/{book_id}/graphs` | Download co-occurrence and similarity graph metrics (JSON) |
| `GET /books/{book_id}/file` | Download the originally uploaded PDF / DOCX |
| `GET /books/{book_id}/results/{model_name}` | Retrieve all stored agent results for a model |

Example:

```bash
curl -o datasets.json "http://localhost:8000/books/<BOOK_ID>/datasets"
curl -o graphs.json  "http://localhost:8000/books/<BOOK_ID>/graphs"
curl -o book.pdf     "http://localhost:8000/books/<BOOK_ID>/file"
```

## Architecture

```
┌────────────┐  PDF/DOCX   ┌──────────────────────┐
│   Client   │────────────▶│  FastAPI  /upload-book│
└────────────┘             └──────┬───────────────┘
                                  │
               ┌──────────────────┼──────────────────┐
               ▼                  ▼                   ▼
        Azure Blob Storage  Azure Table Storage  In-memory cache
        (files, datasets,   (book metadata,      (per-process)
         graphs, results)    result index)
                                  │
               ┌──────────────────┼──────────────────┐
               ▼                  ▼                   ▼
        /agents/*            /aggregate          /books/*/...
        (calls Azure         (fuses 5 agents,    (download datasets,
         OpenAI per prompt)   stores result)      graphs, files)
```

## Statistical Methodology

### Hypothesis framework

Each agent tests:

- **H0**: The AI model was **not** trained on this book.
- **H1**: The AI model **was** trained on this book.

### Internal-control baseline (no external book)

The uploaded book is split into equal-length token segments. Null behavior under H0 is estimated from:

- Segment-to-segment variability.
- Cross-segment distributional distances.
- Bootstrapped / permutation-derived null distributions.

### Agent calibration

- Rare phrase agent uses binomial test + beta-binomial calibrated LR.
- Stylometric and semantic agents use permutation + density-ratio LR.
- Distribution and entropy agents use permutation-calibrated distance scores and bounded LR mappings.
- LRs are clipped to safe finite bounds to avoid pathological infinities.

### Bayesian fusion logic

Aggregate endpoint:

1. Computes per-agent LR and p-value.
2. Applies correlation penalty over log-LR dependence.
3. Combines evidence in log space.
4. Computes posterior probability:

\[
posterior = \frac{LR \cdot prior}{LR \cdot prior + (1-prior)}
\]

### Guardrails

- If all p-values > 0.05, posterior is capped at 0.65.
- If rare phrase exact matches are zero, negative evidentiary weight is applied.
- No infinite LR values; variance-collapse protections are enabled.
- Final executive summary is aligned with statistical conclusion language.

## Azure Storage Layout

```
sfas-data (Blob Container)
└── {book_id}/
    ├── original/{filename}       # uploaded PDF/DOCX
    ├── raw_text.txt              # extracted raw text
    ├── normalized_text.txt       # normalized text
    ├── tokens.json               # word tokens
    ├── sentences.json            # sentence list
    ├── segments.json             # text segments
    ├── segment_tokens.json       # tokenized segments
    ├── datasets.json             # n-grams, stylometry, entropy
    ├── graphs.json               # co-occurrence & similarity metrics
    └── results/{model_name}/
        ├── rare_phrase.json      # agent result
        ├── stylometric.json
        ├── distribution.json
        ├── entropy.json
        ├── semantic.json
        └── aggregate.json        # fused result
```

**Table: sfasbooks**

| PartitionKey | RowKey | Fields |
|---|---|---|
| `book` | `{book_id}` | sha256, word_count, sentence_count, token_count, page_count, extraction_timestamp, original_filename |

**Table: sfasresults**

| PartitionKey | RowKey | Fields |
|---|---|---|
| `{book_id}` | `{model}__{agent}` | model_name, agent_name, blob_path, p_value, likelihood_ratio, evidence_direction |
