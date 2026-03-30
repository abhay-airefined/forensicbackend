# First AIDA — AI Forensic Attribution System
## Technical Architecture & Product Roadmap

**Document Classification:** Confidential — Executive & Technical Review  
**Version:** 1.0  
**Date:** March 6, 2026  
**Prepared for:** Ingenta Group Strategic Partnership Review

---

## Executive Summary

**First AIDA** (AI Data Attribution) is an enterprise-grade forensic attribution platform designed to scientifically determine whether machine learning models have been trained on, influenced by, or have memorized specific copyrighted content. The system leverages a **multi-agent forensic architecture** comprising **30+ specialized detection agents**, each performing independent evidentiary tests that are aggregated through rigorous Bayesian statistical fusion to produce court-admissible attribution scores.

The platform directly addresses the publishing industry's critical need for **intellectual property protection** in the age of generative AI, enabling publishers to:
- Detect unauthorized training data usage
- Generate legal-grade forensic reports
- Protect author rights and royalty streams
- Ensure compliance with emerging AI regulations

### Strategic Alignment with Ingenta

First AIDA integrates seamlessly with Ingenta's existing publishing ecosystem:

| Ingenta Product | First AIDA Integration Point |
|-----------------|------------------------------|
| **Ingenta Connect** (5M+ articles, 16K publications) | Bulk forensic scanning of hosted content against AI models |
| **Ingenta Edify** | Embedded attribution layer for digital library platforms |
| **Vista author2reader** | Rights & royalties protection through proactive detection |
| **PCG Services** | AI attribution consulting & market research offerings |

---

## 1. System Philosophy

The First AIDA architecture follows established forensic science principles:

> **No single test can prove training inclusion.** Instead, the system implements a **multi-instrument forensic methodology** where independent evidence sources are statistically fused to produce legally defensible confidence scores.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FORENSIC EVIDENCE CHAIN                             │
│                                                                             │
│   Independent Evidence Sources → Statistical Fusion → Legal-Grade Score     │
│                                                                             │
│   Each agent produces:                                                      │
│   • Test statistic with confidence interval                                 │
│   • Likelihood ratio (P(Evidence|Trained) / P(Evidence|Not Trained))       │
│   • Uncertainty estimate                                                    │
│   • Evidence direction (supports/contradicts/inconclusive)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. High-Level System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          FIRST AIDA PLATFORM ARCHITECTURE                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │                    LAYER 1: DATA INGESTION                           │    ║
║  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │    ║
║  │  │ PDF/DOCX    │ │ EPUB/HTML   │ │ Model API   │ │ Reference   │    │    ║
║  │  │ Upload      │ │ Ingestion   │ │ Access      │ │ Corpora     │    │    ║
║  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘    │    ║
║  │         │               │               │               │           │    ║
║  │         └───────────────┴───────┬───────┴───────────────┘           │    ║
║  │                                 ▼                                    │    ║
║  │  ┌─────────────────────────────────────────────────────────────┐    │    ║
║  │  │ Text Extraction → Normalization → Tokenization → Segmentation │   │    ║
║  │  │ N-gram Generation → Stylometric Feature Extraction             │   │    ║
║  │  │ Entropy Profiling → Graph Metric Computation                   │   │    ║
║  │  └─────────────────────────────────────────────────────────────┘    │    ║
║  └──────────────────────────────────────────────────────────────────────┘    ║
║                                    │                                         ║
║                                    ▼                                         ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │                    LAYER 2: FORENSIC AGENT LAYER                     │    ║
║  │                        (30+ Parallel Agents)                          │    ║
║  │                                                                       │    ║
║  │  ┌─────────────────────────────────────────────────────────────┐     │    ║
║  │  │ CATEGORY I: DIRECT MEMBERSHIP AGENTS (4 Agents)             │     │    ║
║  │  │ • Confidence-Based Membership Test                          │     │    ║
║  │  │ • Loss Differential Test                                    │     │    ║
║  │  │ • Entropy Leakage Test                                      │     │    ║
║  │  │ • Shadow Model Simulation                                   │     │    ║
║  │  └─────────────────────────────────────────────────────────────┘     │    ║
║  │                                                                       │    ║
║  │  ┌─────────────────────────────────────────────────────────────┐     │    ║
║  │  │ CATEGORY II: BEHAVIORAL SIGNATURE AGENTS (4 Agents)         │     │    ║
║  │  │ • Stylometric Fingerprint Analysis                          │     │    ║
║  │  │ • Rare Phrase Regeneration Test                             │     │    ║
║  │  │ • Semantic Bias Alignment Test                              │     │    ║
║  │  │ • Latent Embedding Similarity                               │     │    ║
║  │  └─────────────────────────────────────────────────────────────┘     │    ║
║  │                                                                       │    ║
║  │  ┌─────────────────────────────────────────────────────────────┐     │    ║
║  │  │ CATEGORY III: DISTRIBUTIONAL STATISTICAL AGENTS (5 Agents)  │     │    ║
║  │  │ • KL Divergence Test                                        │     │    ║
║  │  │ • Jensen-Shannon Distance                                   │     │    ║
║  │  │ • Wasserstein Distance                                      │     │    ║
║  │  │ • Mutual Information Leakage                                │     │    ║
║  │  │ • Variance Collapse Test                                    │     │    ║
║  │  └─────────────────────────────────────────────────────────────┘     │    ║
║  │                                                                       │    ║
║  │  ┌─────────────────────────────────────────────────────────────┐     │    ║
║  │  │ CATEGORY IV: GRADIENT & PARAMETER INFLUENCE (4 Agents)      │     │    ║
║  │  │ • Influence Function Estimation                             │     │    ║
║  │  │ • Gradient Alignment (TracIn-style)                         │     │    ║
║  │  │ • Hessian Sensitivity Analysis                              │     │    ║
║  │  │ • Fisher Information Contribution                           │     │    ║
║  │  └─────────────────────────────────────────────────────────────┘     │    ║
║  │                                                                       │    ║
║  │  ┌─────────────────────────────────────────────────────────────┐     │    ║
║  │  │ CATEGORY V: RECONSTRUCTION & EXTRACTION (3 Agents)          │     │    ║
║  │  │ • Prompt-Based Reconstruction                               │     │    ║
║  │  │ • Likelihood Maximization Reconstruction                    │     │    ║
║  │  │ • Latent Nearest Neighbor Search                            │     │    ║
║  │  └─────────────────────────────────────────────────────────────┘     │    ║
║  │                                                                       │    ║
║  │  ┌─────────────────────────────────────────────────────────────┐     │    ║
║  │  │ CATEGORY VI: DATA PROVENANCE & LINEAGE (3 Agents)           │     │    ║
║  │  │ • Token Frequency Signature Comparison                      │     │    ║
║  │  │ • Dataset Watermark Detection                               │     │    ║
║  │  │ • Metadata & Licensing Audit                                │     │    ║
║  │  └─────────────────────────────────────────────────────────────┘     │    ║
║  │                                                                       │    ║
║  │  ┌─────────────────────────────────────────────────────────────┐     │    ║
║  │  │ CATEGORY VII: ADVERSARIAL STRESS TEST (3 Agents)            │     │    ║
║  │  │ • Paraphrase Robustness Test                                │     │    ║
║  │  │ • Noise Injection Sensitivity                               │     │    ║
║  │  │ • Temperature Sensitivity Analysis                          │     │    ║
║  │  └─────────────────────────────────────────────────────────────┘     │    ║
║  │                                                                       │    ║
║  │  ┌─────────────────────────────────────────────────────────────┐     │    ║
║  │  │ CATEGORY VIII: META-STATISTICAL VALIDATION (4 Agents)       │     │    ║
║  │  │ • False Positive Rate Estimator                             │     │    ║
║  │  │ • Bootstrap Confidence Interval Generator                   │     │    ║
║  │  │ • Likelihood Ratio Calculator                               │     │    ║
║  │  │ • Bayesian Aggregation Engine                               │     │    ║
║  │  └─────────────────────────────────────────────────────────────┘     │    ║
║  └──────────────────────────────────────────────────────────────────────┘    ║
║                                    │                                         ║
║                                    ▼                                         ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │                LAYER 3: STATISTICAL AGGREGATION ENGINE               │    ║
║  │                                                                       │    ║
║  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐             │    ║
║  │  │ Likelihood    │→ │ Correlation   │→ │ Bayesian      │             │    ║
║  │  │ Ratio Norm.   │  │ Adjustment    │  │ Fusion        │             │    ║
║  │  └───────────────┘  └───────────────┘  └───────────────┘             │    ║
║  │           │                                      │                    │    ║
║  │           ▼                                      ▼                    │    ║
║  │  ┌───────────────┐                    ┌───────────────┐              │    ║
║  │  │ False Positive│                    │ Confidence    │              │    ║
║  │  │ Calibration   │                    │ Estimation    │              │    ║
║  │  └───────────────┘                    └───────────────┘              │    ║
║  └──────────────────────────────────────────────────────────────────────┘    ║
║                                    │                                         ║
║                                    ▼                                         ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │                      LAYER 4: DECISION & REPORTING                    │    ║
║  │                                                                       │    ║
║  │  • Posterior Probability of Training Influence                        │    ║
║  │  • Strength-of-Evidence Classification                                │    ║
║  │  • Court-Ready Report Generation (Multi-language)                     │    ║
║  │  • Executive Summary & Technical Appendix                             │    ║
║  │  • Reproducibility Documentation                                      │    ║
║  └──────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 3. Complete Agent Specification

### 3.1 Category I: Direct Membership Agents

| Agent # | Name | Methodology | Output |
|---------|------|-------------|--------|
| **1** | Confidence-Based Membership | Measures model probability P_model(x) assigned to suspected samples against reference baseline | Test statistic, p-value |
| **2** | Loss Differential | Computes ΔL = L_reference(x) - L_model(x) to detect lower loss on suspected training samples | Loss delta, significance |
| **3** | Entropy Leakage | Measures output entropy; lower entropy may indicate memorization | Entropy score, deviation |
| **4** | Shadow Model Simulation | Trains surrogate models to estimate membership decision boundary | Membership probability |

### 3.2 Category II: Behavioral Signature Agents

| Agent # | Name | Methodology | Output |
|---------|------|-------------|--------|
| **5** | Stylometric Fingerprint | Extracts sentence length distribution, rare n-grams, function word frequency, structural motifs. Computes cosine/Euclidean distances | Similarity scores |
| **6** | Rare Phrase Regeneration | Tests reproduction of statistically rare sequences (20-grams) via controlled prompting | Regeneration rate, binomial test |
| **7** | Semantic Bias Alignment | Measures bias vector similarity between dataset and model outputs | Bias correlation coefficient |
| **8** | Latent Embedding Similarity | Computes cosine(embedding_dataset, embedding_output) in hidden-layer representation space | Embedding similarity |

### 3.3 Category III: Distributional Statistical Agents

| Agent # | Name | Methodology | Output |
|---------|------|-------------|--------|
| **9** | KL Divergence | D_KL(P_model || P_dataset) — asymmetric divergence | KL divergence, p-value |
| **10** | Jensen-Shannon Distance | Symmetric distribution similarity √(JSD) | JS distance, confidence |
| **11** | Wasserstein Distance | Earth-mover's distance (transport cost) between token distributions | Wasserstein metric |
| **12** | Mutual Information Leakage | I(D; f(x)) — information theoretic leakage measure | MI estimate |
| **13** | Variance Collapse | Compares prediction variance on suspected vs control inputs | Variance ratio |

### 3.4 Category IV: Gradient & Parameter Influence Agents (White-Box)

| Agent # | Name | Methodology | Output |
|---------|------|-------------|--------|
| **14** | Influence Function Estimation | I(z) = -∇θ L_test^T H^{-1} ∇θ L_train | Influence score |
| **15** | Gradient Alignment (TracIn) | Cosine similarity between training and test gradients | Gradient alignment |
| **16** | Hessian Sensitivity | Evaluates curvature impact of removing data | Sensitivity score |
| **17** | Fisher Information Contribution | Quantifies parameter sensitivity due to specific data | Fisher contribution |

### 3.5 Category V: Reconstruction & Extraction Agents

| Agent # | Name | Methodology | Output |
|---------|------|-------------|--------|
| **18** | Prompt-Based Reconstruction | Attempts controlled extraction of verbatim sequences | Extraction success rate |
| **19** | Likelihood Maximization | Searches for high-probability reconstructions via beam search | Max likelihood matches |
| **20** | Latent Nearest Neighbor | Searches embedding space for closest training sample | NN distance, matches |

### 3.6 Category VI: Data Provenance & Lineage Agents

| Agent # | Name | Methodology | Output |
|---------|------|-------------|--------|
| **21** | Token Frequency Signature | Compares dataset token histograms with model output distributions | Signature similarity |
| **22** | Dataset Watermark Detection | Searches for watermark patterns or hidden markers | Watermark detection |
| **23** | Metadata & Licensing Audit | Cross-references documented dataset sources | Provenance report |

### 3.7 Category VII: Adversarial Stress Test Agents

| Agent # | Name | Methodology | Output |
|---------|------|-------------|--------|
| **24** | Paraphrase Robustness | Tests if semantic similarity persists across paraphrased inputs | Robustness score |
| **25** | Noise Injection Sensitivity | Tests stability under small perturbations | Sensitivity metric |
| **26** | Temperature Sensitivity | Detects memorized content under low-temperature sampling | Temperature correlation |

### 3.8 Category VIII: Meta-Statistical Validation Agents

| Agent # | Name | Methodology | Output |
|---------|------|-------------|--------|
| **27** | False Positive Rate Estimator | Estimates baseline error rate using control corpus | FPR estimate |
| **28** | Bootstrap Confidence Generator | Generates uncertainty bounds via resampling (400 iterations) | CI bounds |
| **29** | Likelihood Ratio Calculator | LR = P(E|H1) / P(E|H0) — formal evidential weight | Log LR |
| **30** | Bayesian Aggregation | P(H1|E1..En) ∝ Π_i P(Ei|H1) P(H1) — posterior computation | Posterior probability |

---

## 4. Evidence Aggregation Engine

The Evidence Aggregation Engine combines outputs from all 30+ agents using weighted Bayesian fusion with correlation adjustment to prevent double-counting similar evidence.

### 4.1 Aggregation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EVIDENCE AGGREGATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Likelihood ratios (LR_1, LR_2, ..., LR_30)                         │
│         Uncertainty estimates (σ_1, σ_2, ..., σ_30)                        │
│                                                                             │
│  STEP 1: Normalize statistics                                               │
│          log_LR_i = log(max(LR_min, min(LR_max, LR_i)))                    │
│                                                                             │
│  STEP 2: Estimate dependency matrix (Σ)                                     │
│          Compute correlation between agent outputs                          │
│          Σ = covariance matrix of log-likelihood ratios                    │
│                                                                             │
│  STEP 3: Correlation penalty                                                │
│          penalty = clip(1 - 0.35 × mean_correlation, 0.4, 1.0)             │
│                                                                             │
│  STEP 4: Weighted Bayesian fusion                                           │
│          combined_log_LR = Σ(log_LR_i) × penalty                           │
│          combined_LR = exp(combined_log_LR)                                │
│                                                                             │
│  STEP 5: Posterior probability                                              │
│          posterior = (combined_LR × prior) / (combined_LR × prior + 1-prior)│
│                                                                             │
│  STEP 6: Confidence interval estimation                                     │
│          Bootstrap 400 iterations for uncertainty bounds                    │
│                                                                             │
│  OUTPUT:                                                                    │
│  • Posterior Probability of Training Influence: 0.0 - 1.0                  │
│  • Strength-of-Evidence: Strong | Moderate | Weak | Inconclusive           │
│  • 95% Confidence Interval: [lower, upper]                                 │
│  • Risk Classification: High | Medium | Low                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Decision Framework

| Posterior Probability | Classification | Legal Interpretation |
|-----------------------|----------------|---------------------|
| > 0.95 | **Strong Evidence** | Sufficient for legal proceedings |
| 0.75 – 0.95 | **Moderate Evidence** | Supports but requires additional corroboration |
| 0.50 – 0.75 | **Weak Evidence** | Inconclusive, further investigation recommended |
| < 0.50 | **No Evidence** | Insufficient to support training inclusion claim |

---

## 5. Technical Implementation Stack

### 5.1 Core Platform

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Framework** | FastAPI | High-performance async REST API |
| **Runtime** | Python 3.11+ | Core processing engine |
| **ML Framework** | PyTorch + Transformers | Agent model inference |
| **Statistical Engine** | NumPy, SciPy, scikit-learn | Mathematical computations |
| **NLP Pipeline** | spaCy, NLTK | Text processing |

### 5.2 Azure Cloud Infrastructure

| Service | Usage |
|---------|-------|
| **Azure OpenAI Service** | Model API access for 25+ LLM deployments |
| **Azure Blob Storage** | Document storage, datasets, results (scalable to PB) |
| **Azure Table Storage** | Book metadata, result indices |
| **Azure Container Instances** | Agent parallel execution |
| **Azure API Management** | Rate limiting, authentication, analytics |

### 5.3 Scalability Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PRODUCTION DEPLOYMENT                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────────────────────┐                              │
│                    │   Azure API Management   │                              │
│                    │   (Rate Limiting, Auth)  │                              │
│                    └───────────┬─────────────┘                              │
│                                │                                            │
│         ┌──────────────────────┼──────────────────────┐                     │
│         ▼                      ▼                      ▼                     │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐               │
│  │ API Gateway │       │ API Gateway │       │ API Gateway │               │
│  │  Instance 1 │       │  Instance 2 │       │  Instance N │               │
│  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘               │
│         │                     │                     │                       │
│         └──────────────────┬──┴──────────────────┬──┘                       │
│                            │                     │                          │
│         ┌──────────────────▼──────────────────┐  │                          │
│         │         Agent Orchestrator          │◄─┘                          │
│         │   (Parallel Agent Scheduling)       │                             │
│         └──────────────────┬──────────────────┘                             │
│                            │                                                │
│    ┌───────┬───────┬───────┼───────┬───────┬───────┐                       │
│    ▼       ▼       ▼       ▼       ▼       ▼       ▼                       │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                   │
│ │Ag 1 │ │Ag 5 │ │Ag 9 │ │Ag14 │ │Ag18 │ │Ag24 │ │Ag30 │  (30 Agents)      │
│ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                   │
│                            │                                                │
│                            ▼                                                │
│         ┌──────────────────────────────────────┐                           │
│         │     Evidence Aggregation Engine      │                           │
│         │     (Bayesian Fusion + Reporting)    │                           │
│         └──────────────────────────────────────┘                           │
│                            │                                                │
│         ┌──────────────────┼──────────────────┐                            │
│         ▼                  ▼                  ▼                            │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                       │
│  │ Azure Blob  │   │ Azure Table │   │ Azure Queue │                       │
│  │  Storage    │   │  Storage    │   │  (Async)    │                       │
│  └─────────────┘   └─────────────┘   └─────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Ingenta Integration Architecture

### 6.1 Integration Points

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     INGENTA + FIRST AIDA INTEGRATION                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                      INGENTA CONNECT                                 │    ║
║  │                 (5M+ articles, 16K publications)                     │    ║
║  │  ┌─────────────┐                         ┌─────────────┐            │    ║
║  │  │ Content API │ ◄─────────────────────► │ First AIDA  │            │    ║
║  │  │             │  Batch Attribution      │ Batch Scan  │            │    ║
║  │  └─────────────┘  Webhook Notifications  └─────────────┘            │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                       INGENTA EDIFY                                  │    ║
║  │                 (Digital Library Platform)                           │    ║
║  │  ┌─────────────┐                         ┌─────────────┐            │    ║
║  │  │ Content     │ ◄─────────────────────► │ Embedded    │            │    ║
║  │  │ Management  │  Real-time Attribution  │ Attribution │            │    ║
║  │  │ System      │  API Integration        │ Widget      │            │    ║
║  │  └─────────────┘                         └─────────────┘            │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                    VISTA AUTHOR2READER                               │    ║
║  │              (Rights, Royalties, Fulfillment)                        │    ║
║  │  ┌─────────────┐                         ┌─────────────┐            │    ║
║  │  │ Rights &    │ ◄─────────────────────► │ Attribution │            │    ║
║  │  │ Royalties   │  Infringement Alerts    │ Monitoring  │            │    ║
║  │  │ Database    │  Royalty Impact Report  │ Service     │            │    ║
║  │  └─────────────┘                         └─────────────┘            │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                           PCG SERVICES                               │    ║
║  │              (Marketing, Sales, Consulting)                          │    ║
║  │  ┌─────────────┐                         ┌─────────────┐            │    ║
║  │  │ Client      │ ◄─────────────────────► │ White-label │            │    ║
║  │  │ Publishers  │  Attribution Reports    │ Forensic    │            │    ║
║  │  │             │  Market Intelligence    │ Services    │            │    ║
║  │  └─────────────┘                         └─────────────┘            │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 6.2 Integration API Specifications

#### Batch Attribution API (for Ingenta Connect)

```json
POST /v1/batch/submit
{
  "job_id": "batch-2026-03-06-001",
  "source_system": "ingenta_connect",
  "content_ids": ["10.1000/xyz123", "10.1000/abc456", ...],
  "target_models": ["gpt-4o", "claude-3", "gemini-pro"],
  "callback_url": "https://connect.ingenta.com/webhook/aida",
  "priority": "standard"
}

Response 202 Accepted:
{
  "job_id": "batch-2026-03-06-001",
  "estimated_completion": "2026-03-06T18:00:00Z",
  "status_url": "/v1/batch/batch-2026-03-06-001/status"
}
```

#### Real-time Attribution API (for Edify)

```json
POST /v1/realtime/analyze
{
  "content_type": "article",
  "content_id": "10.1000/xyz123",
  "content_text": "...",
  "target_model": "gpt-4o",
  "urgency": "high"
}

Response 200:
{
  "attribution_probability": 0.87,
  "strength": "Moderate",
  "confidence_interval": [0.81, 0.93],
  "agent_summary": {...},
  "report_url": "/reports/rpt-123456"
}
```

---

## 7. Product Roadmap

### Phase 1: Foundation (Q1 2026 — COMPLETED)

| Milestone | Status | Description |
|-----------|--------|-------------|
| ✅ Core Agent Framework | Complete | 30+ agents operational with parallel execution |
| ✅ Bayesian Fusion Engine | Complete | Correlation-adjusted evidence aggregation |
| ✅ Azure Integration | Complete | Blob/Table storage, OpenAI connectivity |
| ✅ API Layer | Complete | FastAPI with full REST endpoints |
| ✅ Report Generation | Complete | Multi-language (EN/AR) legal memoranda |

### Phase 2: Ingenta Integration (Q2 2026)

| Milestone | Target Date | Description |
|-----------|-------------|-------------|
| 🔄 Ingenta Connect API | Apr 2026 | Batch attribution endpoint for 5M+ articles |
| 🔄 Edify Widget | May 2026 | Embedded real-time attribution UI |
| 🔄 Vista Integration | Jun 2026 | Rights & royalties infringement alerting |
| 🔄 SSO Integration | Jun 2026 | Ingenta identity federation |

### Phase 3: Scale & Optimization (Q3 2026)

| Milestone | Target Date | Description |
|-----------|-------------|-------------|
| 🔲 Horizontal Scaling | Jul 2026 | Multi-region deployment (UK, US, EU) |
| 🔲 Agent Parallelization | Jul 2026 | 30 agents running concurrently |
| 🔲 Caching Layer | Aug 2026 | Redis-based result caching |
| 🔲 PCG White-label Portal | Sep 2026 | Publisher self-service dashboard |

### Phase 4: Enterprise Features (Q4 2026)

| Milestone | Target Date | Description |
|-----------|-------------|-------------|
| 🔲 Bulk Publisher Onboarding | Oct 2026 | Ingenta Connect 400+ publisher migration |
| 🔲 Advanced Reporting | Nov 2026 | Custom report templates, scheduling |
| 🔲 Compliance Dashboard | Nov 2026 | EU AI Act, DMCA, copyright compliance |
| 🔲 Model Marketplace | Dec 2026 | Support for 50+ LLM target models |

### Phase 5: Market Expansion (2027)

| Milestone | Target Date | Description |
|-----------|-------------|-------------|
| 🔲 Academic Licensing | Q1 2027 | University library partnerships |
| 🔲 Legal Services Integration | Q1 2027 | Direct law firm API access |
| 🔲 Government Contracts | Q2 2027 | National archives, regulatory bodies |
| 🔲 Global Publisher Network | Q2 2027 | 1000+ publisher coverage |

---

## 8. Deployment & Operations

### 8.1 Current Production Environment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PRODUCTION ENVIRONMENT (UK SOUTH)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Azure Resource Group: rg-first-aida-prod                                   │
│                                                                             │
│  ┌─────────────────────┐   ┌─────────────────────┐                         │
│  │ Azure App Service   │   │ Azure OpenAI        │                         │
│  │ Plan: P2v3          │   │ GPT-4, GPT-4o       │                         │
│  │ Instances: 3        │   │ Claude-3, Gemini    │                         │
│  └─────────────────────┘   └─────────────────────┘                         │
│                                                                             │
│  ┌─────────────────────┐   ┌─────────────────────┐                         │
│  │ Storage Account     │   │ Azure Table Storage │                         │
│  │ sfas-data (hot)     │   │ sfasbooks, results  │                         │
│  │ Capacity: 10 TB     │   │ Throughput: 20K RU  │                         │
│  └─────────────────────┘   └─────────────────────┘                         │
│                                                                             │
│  ┌─────────────────────┐   ┌─────────────────────┐                         │
│  │ Azure API Mgmt      │   │ Application Insights│                         │
│  │ Developer Tier      │   │ Full telemetry      │                         │
│  └─────────────────────┘   └─────────────────────┘                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 SLA Commitments

| Metric | Target |
|--------|--------|
| **API Availability** | 99.9% |
| **Single Document Analysis** | < 60 seconds |
| **Batch Processing Throughput** | 10,000 documents/hour |
| **Report Generation** | < 5 seconds |
| **Data Retention** | 7 years (legal compliance) |

---

## 9. Security & Compliance

### 9.1 Security Architecture

- **Encryption at Rest:** AES-256 (Azure Storage Service Encryption)
- **Encryption in Transit:** TLS 1.3
- **Authentication:** Azure AD B2C, API Key rotation
- **Authorization:** RBAC with principle of least privilege
- **Audit Logging:** Full chain-of-custody documentation

### 9.2 Compliance Certifications (In Progress)

| Standard | Status | Target |
|----------|--------|--------|
| **ISO 27001** | In progress | Q3 2026 |
| **SOC 2 Type II** | Planned | Q4 2026 |
| **GDPR** | Compliant | Current |
| **EU AI Act** | In progress | Q2 2026 |

---

## 10. Commercial Model

### 10.1 Pricing Tiers (Proposed)

| Tier | Volume | Price | Features |
|------|--------|-------|----------|
| **Starter** | 100 analyses/month | £500/month | 5 agents, basic reports |
| **Professional** | 1,000 analyses/month | £2,500/month | 30 agents, full reports |
| **Enterprise** | Unlimited | Custom | API access, SLA, support |
| **Ingenta Bundle** | Per-publisher | Revenue share | Full integration |

### 10.2 Revenue Projections (Ingenta Partnership)

| Year | Projected Publishers | ARR Contribution |
|------|---------------------|-----------------|
| 2026 | 50 | £625K |
| 2027 | 150 | £2.5M |
| 2028 | 400 | £7.5M |

---

## 11. Competitive Differentiation

| Capability | First AIDA | Competitors |
|------------|------------|-------------|
| **Agent Count** | 30+ | 3-5 typical |
| **Bayesian Fusion** | ✅ Correlation-adjusted | Basic averaging |
| **Legal-Grade Reports** | ✅ Multi-language | Single format |
| **Publishing Integration** | ✅ Ingenta native | Generic API |
| **White-Box Analysis** | ✅ Gradient/Influence | Black-box only |
| **Batch Processing** | ✅ 10K docs/hour | Manual upload |

---

## 12. Appendix: Mathematical Foundations

### A. Likelihood Ratio Computation

For each agent $i$:

$$LR_i = \frac{P(E_i | H_1)}{P(E_i | H_0)}$$

Where:
- $H_1$: Model was trained on the content
- $H_0$: Model was NOT trained on the content
- $E_i$: Evidence from agent $i$

### B. Bayesian Fusion

Combined posterior probability:

$$P(H_1 | E_1, E_2, ..., E_n) = \frac{LR_{combined} \cdot P(H_1)}{LR_{combined} \cdot P(H_1) + (1 - P(H_1))}$$

With correlation-adjusted combined LR:

$$\log(LR_{combined}) = \sum_{i=1}^{n} \log(LR_i) \times \text{penalty}$$

$$\text{penalty} = \text{clip}(1 - 0.35 \times \bar{\rho}, 0.4, 1.0)$$

Where $\bar{\rho}$ is the mean pairwise correlation between agent log-likelihood ratios.

### C. Confidence Interval (Bootstrap)

Using 400 bootstrap iterations:

$$CI_{95\%} = [P_{2.5\%}, P_{97.5\%}]$$

---

## 13. Contact & Next Steps

### Proposed Next Steps for Ingenta Partnership

1. **Technical Deep-Dive** (Week of March 10) — API integration review with Ingenta engineering
2. **Pilot Program Design** (March 2026) — Select 5-10 publishers for beta
3. **Contract Finalization** (April 2026) — Commercial terms, SLA agreement
4. **Phase 2 Kickoff** (April 2026) — Ingenta Connect integration development

---

**Document prepared by:** First AIDA Technical Architecture Team  
**Review status:** Ready for Executive Presentation

---

*© 2026 First AIDA. All rights reserved. Confidential and proprietary.*
