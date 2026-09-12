# Naukr.AI Retail Data & Conversational Intelligence Workbench

A production-minded retail data quality and conversational analytics workbench built for the Naukr.AI `DQ-AI-CHAT-03` technical assessment.

The application combines:

- CSV ingestion and artifact persistence
- Before/after data profiling
- Deterministic data-quality detection and cleaning
- Auditable cleaning plans
- Product/category normalization
- Safe retail joins with relationship and fan-out checks
- Retail analytics
- Structured LLM planning
- Allow-listed deterministic query execution
- Grounded evidence for numerical answers
- Multi-turn chat sessions
- Model failure and invalid-output handling
- Automated evaluation
- API endpoints for the main workflows

The central design principle is:

> The LLM plans the analysis; the application validates, executes, and verifies it.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Repository Structure](#2-repository-structure)
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Environment Configuration](#5-environment-configuration)
6. [Generate Sample Retail Data](#6-generate-sample-retail-data)
7. [Run the Cleaning Pipeline](#7-run-the-cleaning-pipeline)
8. [Build the Canonical Retail Dataset](#8-build-the-canonical-retail-dataset)
9. [Retail Capabilities](#9-retail-capabilities)
10. [Conversational Analytics](#10-conversational-analytics)
11. [AI vs Application Responsibility](#11-ai-vs-application-responsibility)
12. [Query Safety](#12-query-safety)
13. [Grounded Answers and Evidence](#13-grounded-answers-and-evidence)
14. [Multi-Turn Chat](#14-multi-turn-chat)
15. [Ambiguous Questions](#15-ambiguous-questions)
16. [Unsupported Questions](#16-unsupported-questions)
17. [Prompt Injection Protection](#17-prompt-injection-protection)
18. [Model Failure Handling](#18-model-failure-handling)
19. [API](#19-api)
20. [Automated Evaluation](#20-automated-evaluation)
21. [Test Suite](#21-test-suite)
22. [End-to-End Local Verification](#22-end-to-end-local-verification)
23. [Data and Artifact Handling](#23-data-and-artifact-handling)
24. [Security Decisions](#24-security-decisions)
25. [Known Limitations](#25-known-limitations)
26. [Design Tradeoffs](#26-design-tradeoffs)
27. [Submission Walkthrough](#27-submission-walkthrough)
28. [Current Verification Status](#28-current-verification-status)
29. [Final Submission Checklist](#29-final-submission-checklist)
30. [License / Assessment Context](#30-license--assessment-context)

---

## 1. Architecture

```text
                         ┌─────────────────────┐
                         │      CSV Upload      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Raw Artifact      │
                         │ data/raw/<run_id>/   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Data Profiler      │
                         │ schema/nulls/etc.    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Issue Detection      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Cleaning Plan       │
                         │ IDs/reason/risk       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Deterministic       │
                         │   Cleaning            │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Cleaned Artifact      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                 ┌───────────────────────────────────────┐
                 │         Canonical Retail Layer         │
                 │ products/customers/stores/orders       │
                 └──────────────────┬──────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    LLM Planner         │
                         │  Structured Plan       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Plan Validator       │
                         │ Allow-list/limits      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Deterministic        │
                         │   Query Executor        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Evidence Builder      │
                         │   Answer + Trace        │
                         └─────────────────────┘
```

## 2. Repository Structure

```text
naukr-ai-retail/
│
├── app/
│   ├── api/
│   │   └── main.py
│   │
│   ├── chat/
│   │   ├── evidence.py
│   │   ├── executor.py
│   │   ├── llm_planner.py
│   │   ├── models.py
│   │   ├── planner.py
│   │   ├── service.py
│   │   ├── session.py
│   │   ├── session_store.py
│   │   └── validator.py
│   │
│   ├── cleaning/
│   │   ├── cleaner.py
│   │   ├── detector.py
│   │   ├── planner.py
│   │   └── validator.py
│   │
│   ├── profiling/
│   │   └── profiler.py
│   │
│   ├── retail/
│   │   ├── canonical.py
│   │   ├── joins.py
│   │   ├── metrics.py
│   │   └── normalization.py
│   │
│   └── evaluate.py
│
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── sample/
│       ├── customers.csv
│       ├── inventory.csv
│       ├── orders.csv
│       ├── products.csv
│       ├── stores.csv
│       └── generate_data.py
│
├── evaluation/
│   ├── cases.json
│   └── results.json
│
├── evaluation_output/
│
├── tests/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 3. Requirements

Recommended environment:

- Python 3.11+
- macOS / Linux / Windows
- Git
- Ollama for live LLM planning

The project has been developed and tested in a Python virtual environment.

## 4. Installation

Clone the repository and enter the project directory:

```bash
git clone <YOUR_REPOSITORY_URL>
cd naukr-ai-retail
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 5. Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

The supported Ollama settings are:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT_SECONDS=30
```

`.env` should not be committed to Git. The application reads provider configuration from environment variables rather than hard-coding secrets or credentials.

## 6. Generate Sample Retail Data

The repository contains sample retail datasets for:

- orders
- products
- customers
- stores
- inventory

Generate/regenerate the sample datasets with:

```bash
python data/sample/generate_data.py
```

The generated datasets are located under `data/sample/`.

## 7. Run the Cleaning Pipeline

The current command-line cleaning workflow uses the sample orders dataset:

```bash
python app/cleaning/run_cleaning.py
```

The workflow performs:

1. Input loading
2. Data-quality issue detection
3. Cleaning-plan generation
4. Deterministic transformations
5. Cleaning-plan status updates
6. Cleaned CSV persistence

The sample cleaned output is `data/cleaned/orders_clean.csv`.

The cleaning pipeline includes transformations such as:

- exact duplicate removal
- numeric normalization
- date normalization

Cleaning plans contain stable step IDs and information about:

- reason
- affected fields
- risk
- source
- status

## 8. Build the Canonical Retail Dataset

The retail layer combines the cleaned order data with the sample retail dimensions. Run:

```bash
python app/retail/run_canonical.py
```

The canonical dataset is written to `data/cleaned/canonical_retail.csv`.

The canonical layer provides a predictable analytical structure containing information such as:

- order
- customer
- product
- store
- date
- quantity
- price
- discount
- return status
- category
- region
- revenue

## 9. Retail Capabilities

The implementation includes several retail-specific capabilities.

### Product/category normalization

Category variants are normalized into canonical categories while preserving the original value and normalization metadata. Examples include variants such as `tee`, `t shirt`, and `t-shirt` being normalized to `T-Shirt`.

The normalization process preserves:

- original value
- normalized value
- normalization method
- confidence

### Safe joins

Retail joins explicitly specify relationship expectations. The join layer validates:

- required join keys
- dimension uniqueness
- relationship type
- matched rows
- unmatched rows
- duplicate keys
- fan-out behavior

Unexpected join behavior is rejected rather than silently producing inflated analytical results.

### Retail metrics

The retail analytics layer provides metrics including:

- total revenue
- average order value
- return rate
- category sales
- store sales
- inventory stockout frequency

Revenue is calculated deterministically from the canonical retail data.

## 10. Conversational Analytics

The conversational layer supports structured analytical questions such as:

> What is the top category by revenue in the West?

The architecture is:

```text
User Question
     │
     ▼
LLM Planner
     │
     ▼
Structured QueryPlan
     │
     ▼
Plan Validator
     │
     ▼
Deterministic Executor
     │
     ▼
Evidence Builder
     │
     ▼
Answer + Evidence
```

The LLM does not directly execute Python or SQL. It produces a structured plan that is validated by the application before execution.

## 11. AI vs Application Responsibility

### LLM responsibilities

The LLM is used for:

- interpreting natural-language questions
- selecting an allowed analytical intent
- selecting allowed fields
- selecting supported filters
- selecting supported aggregations
- producing a structured query plan
- interpreting conversational follow-ups

### Application responsibilities

The application controls:

- schema validation
- allowed fields
- allowed operators
- allowed aggregations
- query limits
- resource limits
- execution
- numerical calculations
- joins
- evidence generation
- session persistence
- refusal/clarification handling
- model failure handling

This separation prevents model output from becoming arbitrary executable code.

## 12. Query Safety

Structured plans are validated before execution. The validator restricts:

- dataset names
- fields
- operators
- aggregation functions
- grouping fields
- sorting fields
- filter count
- metric count
- group-by count
- sort count
- result limits

The maximum result limit is bounded. The chat execution path does not execute arbitrary Python generated by the model and does not execute unrestricted SQL generated by the model.

## 13. Grounded Answers and Evidence

Numerical answers are calculated by the deterministic executor. The LLM does not invent numerical results.

Evidence records information such as:

- dataset/version
- selected columns
- filters
- operations
- rows considered
- result preview
- execution information

Result previews are bounded. This allows a reviewer to inspect how a numerical answer was produced.

## 14. Multi-Turn Chat

Chat sessions support follow-up questions. For example:

```text
User: What is the top category by revenue in the West?
A:    T-Shirt ...

User: What about the East?
A:    Hoodies ...
```

The follow-up is interpreted using the previous structured plan. The application persists sessions under `artifacts/chat_sessions/`. Session IDs are validated to prevent path traversal.

## 15. Ambiguous Questions

The application does not guess when an important analytical dimension is ambiguous. For example:

> Which region generated the highest revenue?

can result in a clarification rather than silently choosing a region definition. The clarification response identifies the available interpretation/options where applicable.

## 16. Unsupported Questions

Questions requiring unavailable fields or unsupported business concepts are refused rather than answered using invented data. For example, if the dataset does not contain sufficient information for a customer lifetime value calculation, the system refuses the request instead of fabricating a result.

## 17. Prompt Injection Protection

Uploaded CSV contents are treated as untrusted data. Dataset cell values are not automatically included in the LLM planning context — for example, a malicious value inside a product name is treated as ordinary dataset content rather than as an instruction.

The application also treats model-generated structured plans as untrusted and validates them before execution.

## 18. Model Failure Handling

The Ollama planner handles:

- unavailable provider
- timeout
- invalid JSON
- invalid structured output

The evaluator and tests also use a deterministic mock planner so that core application behavior can be tested without depending on an external model.

## 19. API

Start the API with:

```bash
uvicorn app.api.main:app --host 127.0.0.1 --port 8002
```

The API runs at `http://127.0.0.1:8002`.

**Health check**

```bash
curl http://127.0.0.1:8002/health
```

**Profile a CSV**

```bash
curl -X POST http://127.0.0.1:8002/profile \
  -F "file=@data/sample/orders.csv"
```

**Clean a CSV**

```bash
curl -X POST http://127.0.0.1:8002/clean \
  -F "file=@data/sample/orders.csv"
```

**Build canonical retail data**

```bash
curl -X POST http://127.0.0.1:8002/retail/canonical
```

**Ask a conversational question**

```bash
curl -X POST http://127.0.0.1:8002/chat \
  -F "session_id=final-check" \
  -F "question=What is the top category by revenue in the West?"
```

The chat endpoint returns the structured answer and associated evidence.

## 20. Automated Evaluation

The evaluator is implemented as `app/evaluate.py`. Run:

```bash
python -m app.evaluate \
  --input cases.json \
  --output results.json \
  --artifacts-dir ./evaluation_output
```

The evaluator produces machine-readable JSON results. Current repository evaluation:

```text
Total:  8
Passed: 8
Failed: 0
```

The evaluator is designed so that a failure in one case does not abort the complete evaluation batch.

## 21. Test Suite

Run the complete test suite:

```bash
python -m pytest -q
```

Current baseline: **86 passed**

The tests cover areas including:

- profiling
- cleaning transformations
- validation
- idempotency
- retail normalization
- safe joins
- retail metrics
- structured query plans
- plan validation
- grounded chat
- multi-turn sessions
- session persistence
- refusal
- clarification
- model failure
- invalid model output
- prompt injection
- evaluator behavior

## 22. End-to-End Local Verification

A complete local verification sequence is:

```bash
source .venv/bin/activate

python data/sample/generate_data.py

python app/cleaning/run_cleaning.py

python app/retail/run_canonical.py

python -m pytest -q

python -m app.evaluate \
  --input cases.json \
  --output results.json \
  --artifacts-dir ./evaluation_output
```

Expected baseline:

```text
86 passed
8 total
8 passed
0 failed
```

Then start the API:

```bash
uvicorn app.api.main:app --host 127.0.0.1 --port 8002
```

And verify:

```bash
curl http://127.0.0.1:8002/health
```

## 23. Data and Artifact Handling

Input files are treated as untrusted. Uploaded files are stored in isolated run directories such as:

```text
data/raw/<run_id>/
data/cleaned/<run_id>/
artifacts/api_runs/<run_id>/
```

The API validates uploaded filenames and restricts accepted files to CSV uploads. Session IDs are validated before being used for persistent session storage. Generated artifacts should be reproducible from the source datasets and application code rather than treated as source code.

## 24. Security Decisions

Important security boundaries include:

- uploaded filenames are validated
- path traversal is prevented
- uploaded data is isolated by run ID
- model output is validated before execution
- arbitrary Python execution is not permitted
- unrestricted SQL execution is not permitted
- query complexity is bounded
- result limits are bounded
- Ollama configuration is supplied through environment variables
- dataset cells are treated as untrusted data
- session IDs are validated
- the LLM receives limited conversational planning context rather than unrestricted dataset contents

## 25. Known Limitations

The current implementation is intentionally scoped for the assessment. Known limitations include:

- The command-line cleaning implementation currently focuses on the sample orders schema.
- The canonical retail layer currently uses the provided retail datasets and fixed analytical schema.
- The conversational layer operates on the canonical retail dataset rather than dynamically constructing arbitrary dataset joins.
- The local Ollama planner depends on the configured Ollama provider for live natural-language planning.
- The deterministic mock planner is intentionally limited to the evaluation scenarios.
- API authentication/authorization is not implemented for production multi-user deployment.
- Artifact/session retention policies are not implemented.
- The sample normalization dictionary is intentionally limited.
- Some operational controls, such as production rate limiting and distributed request management, are outside the assessment scope.

These limitations are documented rather than hidden from the evaluator.

## 26. Design Tradeoffs

**Deterministic execution over model-generated code**
The system uses the model for planning rather than allowing generated code execution. This improves safety, reproducibility, auditability, and numerical grounding, at the cost of limiting the range of analyses that can be expressed.

**Local Ollama**
A local model keeps the demonstration self-contained and reduces the need to send retail data to an external provider. The tradeoff is that local model quality and availability depend on the developer environment.

**Canonical retail layer**
A canonical dataset simplifies conversational analytics and makes joins auditable. The tradeoff is that arbitrary uploaded schemas are not automatically transformed into every possible analytical model.

## 27. Submission Walkthrough

A short 3–5 minute demonstration should show:

1. Start the application/environment.
2. Generate or load the sample retail CSVs.
3. Run the profiling/cleaning workflow.
4. Show the cleaning plan and before/after result.
5. Demonstrate a retail capability such as category normalization or safe joins.
6. Ask: *What is the top category by revenue in the West?*
7. Show the answer and evidence.
8. Ask: *What about the East?*
9. Show multi-turn context.
10. Demonstrate an ambiguous question and clarification.
11. Optionally demonstrate an unsupported question/refusal.
12. Run the automated evaluator and tests.

## 28. Current Verification Status

At the current development checkpoint:

| Check | Status |
|---|---|
| Test suite | 86 passed |
| Evaluator cases | 8 passed / 0 failed |
| Evaluator batch | completes successfully |
| Multi-turn chat | verified |
| Grounded chat | verified |
| Prompt injection | tested |
| Model failures | tested |

The repository is still being hardened for final submission, particularly around:

- generic cleaning robustness
- unresolved data-quality issue reporting
- API test coverage
- reproducibility
- upload resource controls
- final submission documentation

## 29. Final Submission Checklist

Before pushing the final version:

- [ ] README complete
- [ ] `.env.example` complete
- [ ] Setup works from clean clone
- [ ] Sample data generation works
- [ ] Cleaning pipeline works
- [ ] Canonical retail build works
- [ ] API starts on port 8002
- [ ] API tests pass
- [ ] Full pytest suite passes
- [ ] Evaluator passes
- [ ] Evaluator output is machine-readable
- [ ] Cleaning issues are not silently hidden
- [ ] Upload size/type handling is verified
- [ ] Chat produces grounded numerical answers
- [ ] Multi-turn chat works
- [ ] Ambiguous questions clarify
- [ ] Unsupported questions refuse
- [ ] Prompt injection is handled
- [ ] Git history is meaningful
- [ ] No secrets are committed
- [ ] Final Git status reviewed

## 30. License / Assessment Context

This repository was developed as a technical assessment project for the Naukr.AI `DQ-AI-CHAT-03` challenge. The implementation prioritizes correctness, deterministic execution, safety, auditability, and reproducibility within the assessment scope.