# Frontend Integration Prompt — SFAS Simulation + Live Logs

You are integrating the frontend (React + TypeScript) with the SFAS (Scientific Forensic Attribution System) backend. The backend runs at `http://localhost:8001`. CORS is already enabled for all origins.

---

## 1. BACKEND API REFERENCE

### 1.1 Upload Book
```
POST /upload-book
Content-Type: multipart/form-data
Body: file (PDF or DOCX)

Response 200:
{
  "book_id": "fb7ceaf3-a88c-4f0f-ac54-d95937664d7c",
  "metadata": {
    "sha256": "abc123...",
    "word_count": 85432,
    "sentence_count": 4210,
    "token_count": 92100,
    "page_count": 312,
    "extraction_timestamp": "2026-02-27T10:30:00Z"
  }
}
```
- The backend automatically assigns a simulation scenario based on the filename.
- If the filename contains any of: `tr`, `harry`, `yogi`, `dinner`, `train` → **trained scenario (95%)**.
- All other filenames → **not trained scenario (20%)**.
- Book processing (text extraction, tokenization, dataset building) is real — takes 5-30 seconds depending on book size.

### 1.2 Run Individual Agent
```
POST /agents/{agent-name}
Content-Type: application/json
Body:
{
  "book_id": "fb7ceaf3-...",
  "model_name": "gpt-4.1-test",
  "sample_count": 40
}

Response 200: AgentResponse (see schema below)
```
Available agent endpoints:
- `POST /agents/rare-phrase`
- `POST /agents/stylometric`
- `POST /agents/distribution`
- `POST /agents/entropy`
- `POST /agents/semantic`

**IMPORTANT:** In simulation mode, each agent call takes 2-10 seconds (realistic delays). The response is NOT instant — this is intentional so the UI feels realistic.

### 1.3 Run Aggregate (All 5 Agents + Bayesian Fusion)
```
POST /aggregate
Content-Type: application/json
Body:
{
  "book_id": "fb7ceaf3-...",
  "model_name": "gpt-4.1-test",
  "prior_probability": 0.5
}

Response 200: AggregateResponse (see schema below)
```
- This runs ALL 5 agents sequentially, then fuses results.
- Takes 20-40 seconds total (simulation mode has realistic delays per agent).
- While this is processing, the frontend should poll the log endpoint (see 1.4).

### 1.4 Poll Run Logs (NEW — for live progress)
```
GET /runs/{book_id}/{model_name}/logs?after=0

Response 200:
{
  "logs": [
    {
      "index": 0,
      "timestamp": 1740000000.0,
      "elapsed_s": 0.0,
      "agent": "system",
      "level": "INFO",
      "message": "Starting forensic analysis — scenario='trained'  model='gpt-4.1-test'",
      "detail": ""
    },
    {
      "index": 1,
      "timestamp": 1740000000.5,
      "elapsed_s": 0.5,
      "agent": "rare_phrase",
      "level": "INFO",
      "message": "Preparing 40 rare 20-gram prompts from book corpus",
      "detail": ""
    },
    ...
  ],
  "status": "running",       // "running" | "completed" | "failed" | "unknown"
  "next_after": 15           // use this as ?after= in the next poll
}
```
**Polling pattern:**
1. When the user clicks "Run Analysis" (aggregate), fire the `POST /aggregate` request (it will block for 20-40s).
2. Immediately start polling `GET /runs/{book_id}/{model_name}/logs?after=0` every 500ms.
3. Each poll returns only NEW log entries (pass `next_after` from previous response as `?after=`).
4. Keep polling until `status` is `"completed"` or `"failed"`.
5. When `status === "completed"`, the `POST /aggregate` response will also have arrived — display the final results.

### 1.5 SSE Stream (Alternative to polling)
```
GET /runs/{book_id}/{model_name}/events
Content-Type: text/event-stream

Events:
data: {"index": 0, "timestamp": ..., "agent": "system", "message": "Starting..."}
data: {"index": 1, "timestamp": ..., "agent": "rare_phrase", "message": "Preparing 40 prompts..."}
...
event: done
data: {"status": "completed"}
```
You can use either polling (1.4) or SSE (1.5) — SSE is more efficient but polling is simpler to implement.

### 1.6 Visualization Charts (PNG images)

**Book-level charts** (available immediately after upload):
- `GET /viz/{book_id}/entropy` — Rolling entropy plot
- `GET /viz/{book_id}/word-frequency?top_n=40` — Top word frequencies (stopwords removed)
- `GET /viz/{book_id}/stylometry` — Stylometric feature heatmap
- `GET /viz/{book_id}/graph-metrics` — Graph degree distributions

**Agent-level charts** (available after running the agent):
- `GET /viz/{book_id}/{model_name}/rare-phrase` — Match rates + comparison table
- `GET /viz/{book_id}/{model_name}/entropy` — Entropy correlation vs null
- `GET /viz/{book_id}/{model_name}/distribution` — Distribution distance vs null
- `GET /viz/{book_id}/{model_name}/stylometric` — Feature correlations + distance
- `GET /viz/{book_id}/{model_name}/semantic` — Paired similarity + rank metrics

**Aggregate charts** (available after running aggregate):
- `GET /viz/{book_id}/{model_name}/aggregate` — Posterior gauge + per-agent breakdown
- `GET /viz/{book_id}/{model_name}/dashboard` — Full 6-panel dashboard

All chart endpoints return `image/png`. Display them as `<img src="..." />`.

### 1.7 Download Endpoints
- `GET /books/{book_id}/datasets` — JSON download of structured datasets
- `GET /books/{book_id}/graphs` — JSON download of graph metrics
- `GET /books/{book_id}/file` — Original uploaded PDF/DOCX
- `GET /books/{book_id}/results/{model_name}` — All agent results as JSON array

### 1.8 Generate Legal Memorandum (NEW)
```
POST /generate-memorandum
Content-Type: application/json
Body:
{
  "book_id": "fb7ceaf3-...",
  "model_name": "gpt-4.1-test",
  "case_number": "SFAS-FB7CEAF3",
  "firm_name": "forensic-legal",
  "role": "plaintiff",
  "tone_style": "assertive",
  "length_style": "detailed",
  "book_title": "Harry Potter and the Philosopher's Stone",
  "book_author": "J.K. Rowling"
}

Response 200:
{
  "english_markdown_memorandum": "# AI Training Detection — Forensic Analysis Report\n\n...",
  "arabic_markdown_memorandum": "...",
  "blob_path": "...",
  "case_number": "SFAS-FB7CEAF3"
}
```
- **Requires**: The aggregate endpoint must have been run first (`POST /aggregate`).
- `case_number` is auto-generated from book_id if left empty.
- `role`: `"plaintiff"` (copyright holder) or `"defendant"` (AI company).
- `tone_style`: `"assertive"` (strong, accusatory) or `"conciliatory"` (diplomatic).
- `length_style`: `"detailed"` (comprehensive) or `"concise"` (summary).
- `book_title` and `book_author` are optional — improve the memorandum quality.
- The `english_markdown_memorandum` contains a full Markdown document suitable for rendering or PDF export.
- Calls the AILA legal backend (localhost:8000) if available; otherwise returns a locally-generated forensic evidence report.

---

## 2. RESPONSE SCHEMAS

### AgentResponse
```typescript
interface AgentResponse {
  agent_name: string;          // "rare_phrase" | "stylometric" | "distribution" | "entropy" | "semantic"
  hypothesis_test: {
    H0: string;                // "Model was NOT trained on this book"
    H1: string;                // "Model WAS trained on this book"
  };
  metrics: Record<string, any>; // Agent-specific metrics (varies per agent)
  p_value: number;              // 0.0 - 1.0  (lower = more evidence for training)
  likelihood_ratio: number;     // >1 supports H1 (trained), <1 supports H0
  log_likelihood_ratio: number;
  evidence_direction: string;   // "supports_H1" | "supports_H0"
}
```

### AggregateResponse
```typescript
interface AggregateResponse {
  posterior_probability: number;    // 0.0 - 1.0 (the main result: probability book was used in training)
  log_likelihood_ratio: number;
  strength_of_evidence: string;    // "No Evidence" | "Weak" | "Moderate" | "Strong" | "Very Strong" | "Decisive"
  agent_breakdown: AgentResponse[];  // Array of 5 agent results
  executive_summary: string;       // Human-readable conclusion
}
```

### MemorandumRequest
```typescript
interface MemorandumRequest {
  book_id: string;
  model_name: string;
  case_number?: string;            // Auto-generated if empty
  firm_name?: string;              // Default: "forensic-legal"
  role?: "plaintiff" | "defendant"; // Default: "plaintiff"
  tone_style?: "assertive" | "conciliatory"; // Default: "assertive"
  length_style?: "detailed" | "concise";     // Default: "detailed"
  book_title?: string;             // Optional, improves memorandum
  book_author?: string;            // Optional, improves memorandum
}
```

### MemorandumResponse
```typescript
interface MemorandumResponse {
  english_markdown_memorandum: string;  // Full Markdown legal document
  arabic_markdown_memorandum: string;   // Arabic translation (may be empty)
  blob_path: string;                    // Storage path (may be empty)
  case_number: string;                  // The case number used
  aggregate_chart_base64: string;       // Base64-encoded PNG of the aggregate analysis chart
}
```

**Frontend Usage for Chart:**
The `aggregate_chart_base64` field contains the base64-encoded PNG image of the aggregate analysis chart. 
Render it directly in an `<img>` tag:

```tsx
{response.aggregate_chart_base64 && (
  <img 
    src={`data:image/png;base64,${response.aggregate_chart_base64}`}
    alt="Aggregate Forensic Analysis"
  />
)}
```

The markdown contains a placeholder `<!-- AGGREGATE_CHART_PLACEHOLDER -->` where the chart should be inserted. 
Replace this placeholder with the rendered image or inject the image at that location in your markdown viewer.

### LogEntry
```typescript
interface LogEntry {
  index: number;
  timestamp: number;       // Unix timestamp
  elapsed_s: number;       // Seconds since run started
  agent: string;           // "system" | "rare_phrase" | "stylometric" | "distribution" | "entropy" | "semantic" | "fusion"
  level: string;           // "INFO" | "WARNING" | "ERROR"
  message: string;         // Human-readable log message
  detail: string;          // Optional extra detail
}
```

### LogPollResponse
```typescript
interface LogPollResponse {
  logs: LogEntry[];
  status: "running" | "completed" | "failed" | "unknown";
  next_after: number;
}
```

---

## 3. UI REQUIREMENTS

### 3.1 Live Log Panel During Analysis

When the user triggers the aggregate analysis (POST /aggregate), show a **live log panel** that displays real-time progress. This is the most important new UI feature.

**Design:**
- Show a collapsible/expandable log panel below or beside the analysis results area.
- Each log entry should show: `[elapsed time] [agent name] message`
- Color-code by agent:
  - `system` → gray
  - `rare_phrase` → blue
  - `stylometric` → purple
  - `distribution` → orange
  - `entropy` → green
  - `semantic` → teal
  - `fusion` → gold/yellow
- Auto-scroll to the latest log entry.
- Show a progress indicator: "Agent 2/5: stylometric" based on which agent's logs are currently appearing.

**Implementation (polling approach):**
```typescript
// Start when user clicks "Run Analysis"
const runAnalysis = async (bookId: string, modelName: string) => {
  // 1. Fire aggregate request (will resolve in 20-40s)
  const aggregatePromise = fetch(`http://localhost:8001/aggregate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      book_id: bookId,
      model_name: modelName,
      prior_probability: 0.5,
    }),
  });

  // 2. Start polling logs immediately
  let nextAfter = 0;
  let status = 'running';
  const allLogs: LogEntry[] = [];

  const pollInterval = setInterval(async () => {
    try {
      const res = await fetch(
        `http://localhost:8001/runs/${bookId}/${modelName}/logs?after=${nextAfter}`
      );
      const data: LogPollResponse = await res.json();
      allLogs.push(...data.logs);
      nextAfter = data.next_after;
      status = data.status;
      
      // Update UI with new logs
      setLogs([...allLogs]);
      
      if (status === 'completed' || status === 'failed') {
        clearInterval(pollInterval);
      }
    } catch (e) {
      // Ignore poll errors during processing
    }
  }, 500);

  // 3. Await final result
  const response = await aggregatePromise;
  const result: AggregateResponse = await response.json();
  clearInterval(pollInterval);
  
  // 4. Final poll to get any remaining logs
  const finalRes = await fetch(
    `http://localhost:8001/runs/${bookId}/${modelName}/logs?after=${nextAfter}`
  );
  const finalData = await finalRes.json();
  allLogs.push(...finalData.logs);
  setLogs([...allLogs]);
  
  return result;
};
```

**Alternative (SSE approach):**
```typescript
const eventSource = new EventSource(
  `http://localhost:8001/runs/${bookId}/${modelName}/events`
);
eventSource.onmessage = (event) => {
  const logEntry: LogEntry = JSON.parse(event.data);
  setLogs(prev => [...prev, logEntry]);
};
eventSource.addEventListener('done', (event) => {
  eventSource.close();
});
```

### 3.2 Per-Agent Section with Logs

Under each agent section in the analysis results panel, show:

1. **Agent status indicator**: 
   - ⏳ Waiting (gray) — before the agent starts
   - 🔄 Running (blue spinner) — while logs for that agent are appearing
   - ✅ Complete (green) — after the agent's final log entry
   - ❌ Failed (red) — if an error occurred

2. **Mini log view**: A collapsible section showing only that agent's log messages (filter by `log.agent === agentName`).

3. **Results summary** (after completion):
   - Likelihood Ratio with color coding (green if >1, red if <1)
   - p-value with significance indicator (★ if p < 0.05)
   - Evidence direction badge: "Supports Training" (H1) or "Supports Not Trained" (H0)

4. **Chart image**: Load from `GET /viz/{book_id}/{model_name}/{agent-slug}` after the agent completes.

### 3.3 Agent Execution Order

The agents run in this fixed order during aggregate:
1. `rare_phrase` (6-10s)
2. `stylometric` (3-5s)
3. `distribution` (2.5-4.5s)
4. `entropy` (2-4s)
5. `semantic` (4-7s)

You can detect which agent is currently running by watching which `agent` name appears in the newest log entries.

### 3.4 Aggregate Results Display

After all agents complete, show:

1. **Posterior Probability** — the main result. Display as a large gauge/meter:
   - 0-30%: Green "Not Trained" 
   - 30-70%: Yellow "Inconclusive"
   - 70-100%: Red "Likely Trained"

2. **Strength of Evidence** badge: "Weak" / "Moderate" / "Strong" / "Very Strong" / "Decisive"

3. **Executive Summary** — the `executive_summary` string from the response.

4. **Per-Agent Breakdown** table:
   | Agent | LR | p-value | Direction |
   |-------|----|---------|-----------|
   | Rare Phrase | 11.73 | 0.0012 ★ | Supports Training |
   | Stylometric | 4.83 | 0.0137 ★ | Supports Training |
   | Distribution | 3.47 | 0.0194 ★ | Supports Training |
   | Entropy | 2.81 | 0.0241 ★ | Supports Training |
   | Semantic | 8.47 | 0.0038 ★ | Supports Training |
6. **Full dashboard**: `<img src="http://localhost:8001/viz/{bookId}/{modelName}/dashboard" />`

### 3.5 Legal Memorandum Generation (NEW)

After the aggregate analysis is complete, show a **"Generate Legal Memorandum"** button. When clicked:

1. Show a brief form (pre-filled defaults are fine):
   - **Book Title** (text input) — optional
   - **Book Author** (text input) — optional
   - **Role**: dropdown → `plaintiff` (default) | `defendant`
   - **Tone**: dropdown → `assertive` (default) | `conciliatory`
   - **Length**: dropdown → `detailed` (default) | `concise`

2. Call `POST /generate-memorandum` with the form data + `book_id` + `model_name`.

3. While generating, show a loading spinner (takes 5-30s if AILA is available, instant fallback).

4. On response, render `english_markdown_memorandum` as formatted Markdown in a document viewer panel:
   - Use a Markdown renderer (e.g., `react-markdown` or similar)
   - Show the document in a scrollable panel with professional styling
   - Add a **"Download as Markdown"** button that exports the content as a `.md` file
   - Add a **"Print / Export PDF"** button using `window.print()` with print-friendly CSS

5. The memorandum contains:
   - Case information and metadata
   - Forensic detection results (posterior probability, per-agent evidence)
   - Legal implications and recommended claims
   - Structured as a formal legal evidence document

### 3.6 Book-Level Charts (After Upload)

After a book is uploaded successfully, immediately show these 4 book-level charts:
```html
<img src="http://localhost:8001/viz/{bookId}/entropy" />
<img src="http://localhost:8001/viz/{bookId}/word-frequency?top_n=40" />
<img src="http://localhost:8001/viz/{bookId}/stylometry" />
<img src="http://localhost:8001/viz/{bookId}/graph-metrics" />
```

---

## 4. COMPLETE USER FLOW

```
1. User uploads a book (PDF/DOCX)
   → POST /upload-book
   → Show upload progress
   → On success: display book metadata + 4 book-level charts

2. User clicks "Run Forensic Analysis"
   → Fire POST /aggregate (async, takes 20-40s)
   → Immediately start polling GET /runs/{bookId}/{modelName}/logs?after=0 every 500ms
   → Show live log panel with agent progress
   → Show per-agent status indicators (waiting → running → complete)
   
3. As each agent completes:
   → Update agent status to ✅
   → Show agent's LR / p-value / direction summary
   → Load agent's chart image

4. When POST /aggregate resolves:
   → Stop polling logs
   → Show final aggregate results (posterior, strength, summary)
   → Load aggregate + dashboard charts

5. User clicks "Generate Legal Memorandum"
   → Fire POST /generate-memorandum with book_id, model_name, book title/author
   → Show loading spinner while generating
   → On success: render the english_markdown_memorandum as formatted Markdown
   → Provide a "Download" button that exports the memorandum as .md or prints as PDF
```

---

## 5. MODEL NAME

The default model name to use in all requests is: `"gpt-4.1-test"`

This should be pre-filled in any model name input field, or hardcoded if there's no user-facing selector.

---

## 6. EXAMPLE LOG SEQUENCE

Here's what the log stream looks like for a typical aggregate run (trained scenario):

```
[0.00s] [system]       Starting forensic analysis — scenario='trained'  model='gpt-4.1-test'
[0.01s] [system]       Book fb7ceaf3… loaded — running 5 forensic agents sequentially
[0.02s] [rare_phrase]  ▶ Starting agent 1/5: rare_phrase
[0.02s] [rare_phrase]  Preparing 40 rare 20-gram prompts from book corpus
[0.32s] [rare_phrase]  Prompt/expected pairs constructed (prompt=12 words, expect=8 words)
[0.92s] [rare_phrase]  Sending 40 prompts to Azure OpenAI deployment 'gpt-4.1-test'…
[1.47s] [rare_phrase]    Batch progress: 25% complete (10/40 prompts)
[1.97s] [rare_phrase]    Batch progress: 50% complete (20/40 prompts)
[2.52s] [rare_phrase]    Batch progress: 75% complete (30/40 prompts)
[2.97s] [rare_phrase]  Model outputs received — cleaning continuations
[3.27s] [rare_phrase]  Computing match metrics: exact / partial (Jaccard) / sequential
[3.47s] [rare_phrase]  Running recognition detection on 40 responses
[3.77s] [rare_phrase]  Recognition scan complete — 31 of 40 responses show source awareness
[3.97s] [rare_phrase]  Binomial tests: p(exact)=0.0003  p(soft)=0.0008  p(recognition)=0.0001
[4.12s] [rare_phrase]  Bootstrap CI computed (400 iterations)
[4.22s] [rare_phrase]  Agent complete — LR=11.7300  p=0.0012  direction=supports_H1
[4.23s] [rare_phrase]  ✓ Agent rare_phrase complete — LR=11.7300  p=0.0012  [supports_H1]
[4.24s] [stylometric]  ▶ Starting agent 2/5: stylometric
...
[28.5s] [fusion]       Starting Bayesian fusion of 5 agent results
[29.0s] [fusion]       Fusion complete — posterior=0.9200  strength=Strong
[29.1s] [system]       ✓ All agents complete — posterior probability: 92.0%
```

---

## 7. NOTES

- The backend URL is `http://localhost:8001`. No authentication required.
- All chart endpoints return PNG images — use `<img>` tags or fetch as blob.
- The `model_name` in URLs uses the raw string (e.g., `gpt-4.1-test`). No URL encoding needed since it only contains alphanumeric, hyphens, and dots.
- The log `agent` field values are exactly: `"system"`, `"rare_phrase"`, `"stylometric"`, `"distribution"`, `"entropy"`, `"semantic"`, `"fusion"`.
- If the user runs individual agents (not aggregate), logs are also captured but only for that single agent.
- The `POST /aggregate` request will block until all 5 agents finish — the log polling happens in parallel while waiting for this response.
