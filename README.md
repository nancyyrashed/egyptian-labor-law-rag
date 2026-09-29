# Egyptian Labor Law Information Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions about Egyptian labor law in Arabic. It retrieves the relevant articles from the official text of **Labor Law No. 14 of 2025** and generates grounded answers with article citations. If the answer isn't in the indexed law, it abstains instead of guessing.

**Live demo:** https://egyptian-labor-law-rag.streamlit.app/

> ⚖️ **Disclaimer:** This project provides *legal information, not legal advice*. It is a learning project and can make mistakes. Do not rely on it for legal decisions; consult a qualified lawyer or the official text of the law.

---

## Source

- **Law:** Labor Law No. 14 of 2025 (effective 1 September 2025, replacing the 2003 law)
- **Publication:** Official Gazette, Ministry of Labour (112-page official PDF)
- **Scope:** this single law only, 298 articles. No other legislation is indexed, so questions about criminal, tax, family, property, or traffic law are out of scope by design.

---

## How it works

```
Question (Arabic)
   │
   ▼
Embed with multilingual-e5-base  ("query: " prefix)
   │
   ▼
Chroma vector search → top 8 chunks
   │
   ▼
Constrained prompt (answer ONLY from retrieved articles)
   │
   ▼
Groq LLM (openai/gpt-oss-120b, temperature 0)
   │
   ▼
Parse cited article numbers ──► cross-check against retrieved chunks
   │
   ▼
Answer + citations + hallucination check + disclaimer
```

The RAG loop is written in plain Python with no framework (no LangChain or LlamaIndex), to make each step visible.

### Key behaviors

- **Grounded answers only.** The system prompt forbids using general legal knowledge outside the retrieved articles.
- **Fixed-phrase abstention.** When the answer isn't supported, the model must reply with the exact phrase `غير موجود في القانون المفهرس`. It is a code constant, so abstention is detected by exact match rather than fuzzy text checks.
- **Verified citations.** Every article number the model cites is checked against what Chroma actually retrieved. A cited article that was never retrieved is flagged as a hallucinated citation and shown in the UI.
- **Robust citation parsing.** Handles Latin, Arabic-Indic (`٠-٩`), and Extended Arabic-Indic (`۰-۹`) digits, and citations spread over several lines.

---

## Architecture

| Layer | Choice |
|---|---|
| LLM | Groq API (free tier), `openai/gpt-oss-120b`, reasoning effort `low` |
| Embeddings | `intfloat/multilingual-e5-base` |
| Vector DB | Chroma (persistent, local), 309 chunks |
| Backend | FastAPI (`/ask`, `/health`) |
| Frontend | Streamlit, bilingual (Arabic RTL by default, English toggle) |
| Deployment | Streamlit Community Cloud (direct mode, see below) |

### Project structure

```
data/
├── raw/            # official PDF
├── processed/      # cleaned text, article JSON, chunks JSON, evaluation results
└── chroma/         # persisted vector index
src/
├── pipeline/       # extraction, cleaning, parsing, chunking, embedding, RAG loop (ask.py), evaluation
├── validation/     # corpus quality-control scripts
├── exploration/    # scripts used to investigate PDF problems
├── api/            # FastAPI backend
└── frontend/       # Streamlit app
```

---

## Building the knowledge base

### 1. PDF extraction and Arabic text cleaning

The official PDF did not extract as usable Arabic. Each problem was investigated at the Unicode level before being fixed, and cleaning was deliberately **conservative**: only validated transformations were applied, because this is a legal text.

| Problem | Handling |
|---|---|
| Arabic Presentation Forms | NFKC normalization |
| Characters stored in reversed order | Unicode BiDi algorithm (`python-bidi`). Naive string reversal was rejected because it also reverses numbers. |
| Private Use Area characters | Traced through the PDF's embedded font (`SimplifiedArabic`) to confirm they are Damma, Fathatan, and Shadda, then mapped explicitly |
| Tatweel (`ـ`) | Removed |
| Spaces inside words (`يكو ن`, `الط بية`) | Only individually confirmed broken-word patterns repaired. A generic "remove space between Arabic letters" rule was rejected because it would merge real words. |
| Repeated Gazette header on every page | Removed with a page-aware regex, validated against all 112 pages |
| Page 112 (publisher/deposit info) | Excluded |

Validation results on the final corpus: 111 pages, **298 articles (1–298) with none missing or duplicated**, 0 PUA characters, 0 Tatweel, 0 known broken words.

### 2. Article parsing

The cleaned text is split into 298 structured records (`article_number`, `article_text`, `law_name`, `source`, `page`), reusing the exact heading regex that was validated in the QC step. The title and preamble before Article 1 are intentionally excluded.

### 3. Embedding model selection

Three multilingual models were compared on retrieval over the full corpus using four verified questions:

| Model | Top-5 hit rate | Avg. rank of correct article |
|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 50% | 15.2 |
| `paraphrase-multilingual-mpnet-base-v2` | 50% | 21.5 |
| **`intfloat/multilingual-e5-base`** | **100%** | **1.8** |

The paraphrase models are trained for symmetric sentence similarity and tended to find the right topic cluster but not the specific article. E5 is trained for asymmetric query-to-passage retrieval, which matches this task. It requires `query:` / `passage:` prefixes, applied consistently everywhere.

### 4. Article-aware chunking

Most articles (293 of 298) are short and stay as one chunk. Five exceeded E5's 512-token limit and were split:

- Articles with numbered items (1, 41, 79, 134) are split at item boundaries, with the article's opening framing sentence repeated in every chunk.
- Article 253 has no numbered structure, so it is split at real sentence boundaries. PDF line-wrapping is undone first, since an earlier line-based split cut a sentence in half.
- One very long item (the "wage" definition in Article 1) still exceeded the limit alone, so a sentence-level fallback was added.

Result: **309 chunks, all verified under 512 tokens** with the real tokenizer.

---

## Evaluation

A 20-question test set was written with ground truth verified directly against the article text, not from memory: 15 in-scope questions across a range of topics (notice periods, child labor, rest-day pay, dismissal, disciplinary sanctions, sick leave, and more) and 5 out-of-scope questions (criminal, tax, property, family, and traffic law) to test abstention. Questions that turned out not to be answerable from this law were removed or reworded rather than kept.

| Metric | Result |
|---|---|
| Retrieval hit rate (in-scope) | 100% (15/15) |
| Citation accuracy | 100% (15/15) |
| Hallucinated citations | 0 |
| False abstention (refused an answerable question) | 0% (0/15) |
| Correct abstention (out-of-scope) | 100% (5/5) |

Per-question results are in `data/processed/phase3_eval_summary.md` and `phase3_eval_results.json`.

### Failures found and fixed along the way

1. **Retrieval miss on the notice-period question.** Article 156 states the notice length but never uses the phrase "مهلة الإخطار", while neighboring articles (158–164) use it repeatedly. At `top_k=5`, Article 156 ranked 8th. Raising `TOP_K` to **8** fixed it with no regressions.
2. **Rate limit during evaluation.** A live Groq 429 hit one question mid-run. The per-question error handling and incremental saving kept the other 19 results intact. A short delay between questions was added.
3. **One-character abstention slip.** The model wrote `المفهرد` instead of `المفهرس` when abstaining, which failed the exact-match check. The cause was sampling variance, so `temperature` was lowered to **0** rather than loosening the detection.
   
---

## Latency

Timing was measured before anything was changed. Retrieval takes about 0.1 s once warm; the wait is the Groq call. Three fixes were applied:

- **Lower reasoning effort** (`reasoning_effort="low"`) for the reasoning model, since the task is extractive. The full evaluation was re-run afterward with identical results.
- **Startup warm-up** that runs a real retrieval, so the first user doesn't pay for model and index loading.
- **LRU cache** for repeated questions (500 entries), which also saves free-tier quota. Errors are never cached.

Typical warm response time is roughly 1–3 seconds.

---

## Running locally

### Prerequisites

- Python 3.11
- A free [Groq API key](https://console.groq.com/)

### Setup

```bash
git clone <your-repo-url>
cd egyptian-labor-law-rag

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install sentence-transformers chromadb groq python-dotenv \
            fastapi uvicorn requests streamlit
```

To rebuild the knowledge base from the PDF, also install `pdfplumber python-bidi fonttools`. This is optional, since the processed data and Chroma index are already in the repo.

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_...
```

### Run the app (two terminals, from the project root)

```bash
# Terminal 1: backend
uvicorn src.api.main:app --reload

# Terminal 2: frontend
streamlit run src/frontend/streamlit_app.py
```

The frontend talks to the API at `http://localhost:8000/ask` by default (override with `API_URL`).

### Other commands

```bash
python src/pipeline/ask.py               # run the RAG loop on the smoke-test questions
python src/pipeline/evaulate_phase3.py   # run the 20-question evaluation
```

### Rebuilding the knowledge base (optional)

```bash
python src/pipeline/build_knowledge_base.py   # PDF → cleaned text
python src/pipeline/parse_articles.py         # text → 298 article records
python src/pipeline/chunck_articles.py        # articles → 309 chunks
python src/pipeline/embed_and_store.py        # chunks → Chroma
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none (required) | Groq authentication |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | LLM used for generation |
| `GROQ_REASONING_EFFORT` | `low` | Set to an empty string for models that don't support it |
| `GROQ_MAX_COMPLETION_TOKENS` | `1500` | Includes hidden reasoning tokens for reasoning models |
| `RAG_MODE` | `api` | `api` calls FastAPI; `direct` runs the pipeline inside Streamlit |
| `API_URL` | `http://localhost:8000/ask` | Backend URL in `api` mode |

### API

| Endpoint | Description |
|---|---|
| `GET /health` | `{"status": "ok", "ready": true}` once resources have loaded |
| `POST /ask` | Body: `{"question": "..."}`. Returns `answer`, `retrieved_articles`, `cited_articles`, `is_abstention`, `hallucinated_citations` |

Errors return distinct, localized responses: 400 (empty question), 503 (still loading), 429 (Groq rate limit), 500 (anything else, with details logged server-side only).

---

## Deployment

The live app runs on **Streamlit Community Cloud**. Other free options were checked against current documentation and rejected: Hugging Face Docker Spaces require a paid plan, and Google Cloud Run requires a payment method on file.

Because that platform only runs the Streamlit script, the frontend has a `direct` mode (`RAG_MODE=direct`) that runs the pipeline in-process instead of calling FastAPI. The FastAPI backend remains in the repo and is the way to run the app locally. To deploy your own copy, set `GROQ_API_KEY` and `RAG_MODE = "direct"` in the platform's Secrets. The API key is never stored in the repo.

Free hosting has practical limits: an idle app may sleep and reload the model on the next visit, and the in-memory answer cache resets on restart.

---

## Design decisions and trade-offs

| Decision | Reasoning |
|---|---|
| Multilingual E5 embeddings | Chosen on measured retrieval results over the corpus, not on model size or reputation |
| Article-aware chunking | Legal articles are self-contained units; splitting only when required by token limits, and at item or sentence boundaries, keeps legal structure intact |
| Fixed abstain phrase | Makes abstention machine-checkable and avoids fuzzy matching |
| Citation cross-checking | Catches invented article numbers programmatically instead of relying on eyeballing |
| Plain Python, no framework | Keeps each RAG step visible and debuggable |
| `TOP_K=8` | Fixed a real retrieval miss; the cost is a slightly larger prompt |
| Direct mode on the live app | Necessary for a Streamlit-only free host; trades away the two-process architecture for the deployed version only |
| No response streaming | Warm answers arrive in about 1–3 s, so the added frontend complexity wasn't justified yet |

---

## Known limitations

- Covers only Law No. 14 of 2025. It does not include executive regulations, ministerial decrees, or other related laws (for example, the Social Insurance Law is referenced but not indexed).

## Acknowledgements

- **Source text:** Labor Law No. 14 of 2025, Official Gazette, published by the Egyptian Ministry of Labour.
- Built with [Groq](https://groq.com/), [Sentence-Transformers](https://www.sbert.net/), [Chroma](https://www.trychroma.com/), [FastAPI](https://fastapi.tiangolo.com/), and [Streamlit](https://streamlit.io/).
