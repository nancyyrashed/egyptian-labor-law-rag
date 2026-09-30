"""
Phase 4: FastAPI backend.

Wraps the existing Phase 2 pipeline (ask.py) behind a single HTTP
endpoint. No RAG logic is duplicated here - load_resources() and
ask() are imported directly, same pattern already used by
evaluate_phase3.py.

Resources (embedding model, Chroma collection, Groq client) are
loaded ONCE at startup, not per-request - re-loading the embedding
model on every request would make each call far slower than it
needs to be for an interactive site.

Rate-limit handling: a live endpoint doesn't have evaluate_phase3.py's
try/except + retry-friendly batch loop. If Groq's free-tier rate
limit is hit on a single request, that request should get a clear,
recoverable error - not a raw 500 crash.

Run:
    uvicorn src.api.main:app --reload

Requires the same .env as ask.py:
    GROQ_API_KEY=gsk_...
"""

import os
import sys
from collections import OrderedDict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Reuse Phase 2's pipeline directly, same pattern as evaluate_phase3.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from ask import load_resources, ask, retrieve  # noqa: E402


app = FastAPI(title="Egyptian Labor Law Information Assistant")

# Allows a locally-run Streamlit app (different port) to call this API.
# Tighten this to a specific origin before deploying publicly (Phase 5).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup, not per-request.
_resources = {}

# Small in-memory LRU cache of successful answers, keyed on the
# whitespace-normalized question. Repeat questions (the example and
# sidebar buttons send the same text every time) return instantly and
# don't spend Groq rate-limit quota. Errors are never cached.
CACHE_MAX = 500
_cache: "OrderedDict[str, dict]" = OrderedDict()


@app.on_event("startup")
def startup():
    print("Loading embedding model, Chroma collection, and Groq client...")
    embed_model, collection, groq_client = load_resources()
    _resources["embed_model"] = embed_model
    _resources["collection"] = collection
    _resources["groq_client"] = groq_client

    # Warm BOTH stages so the first real question doesn't pay for them:
    # the first encode() call pays one-time PyTorch initialization, and
    # the first Chroma query loads the index from disk. retrieve() runs
    # both (embedding + collection query) in one call.
    retrieve(embed_model, collection, "warmup")
    print("Ready.")


class AskRequest(BaseModel):
    question: str
    lang: str = "ar"  # "ar" | "en" - language the answer is written in


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieved_articles: list[int]
    cited_articles: list[int]
    is_abstention: bool
    hallucinated_citations: list[int]


@app.get("/health")
def health():
    return {"status": "ok", "ready": bool(_resources)}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    question = request.question.strip()
    lang = "en" if request.lang == "en" else "ar"

    if not question:
        raise HTTPException(status_code=400, detail="السؤال فارغ")

    if not _resources:
        raise HTTPException(
            status_code=503,
            detail="النظام لا يزال قيد التحميل، يرجى المحاولة بعد قليل",
        )

    # Language is part of the key: same question, different language answer.
    cache_key = f"{lang}|" + " ".join(question.split())
    if cache_key in _cache:
        _cache.move_to_end(cache_key)
        return _cache[cache_key]

    try:
        result = ask(
            _resources["embed_model"],
            _resources["collection"],
            _resources["groq_client"],
            question,
            verbose=False,
            lang=lang,
        )
    except Exception as e:
        # Groq's rate-limit error exposes a status_code attribute in
        # the SDK's exception hierarchy (as does most HTTP-client-
        # based SDKs). Checked defensively via getattr rather than
        # importing a specific exception class, since this should be
        # verified against your installed groq SDK version's actual
        # exception type - inspect the real exception here if this
        # doesn't trigger correctly on a genuine rate-limit error.
        status_code = getattr(e, "status_code", None)

        if status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="عدد الطلبات كبير حاليًا، يرجى المحاولة بعد قليل",
            )

        # Any other failure (timeout, network error, etc.) - don't
        # leak the raw exception to the client, but don't silently
        # swallow it either.
        print(f"[ERROR] /ask failed for question {question!r}: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail="حدث خطأ أثناء معالجة السؤال، يرجى المحاولة مرة أخرى",
        )

    response = AskResponse(
        question=result["question"],
        answer=result["answer"],
        retrieved_articles=result["retrieved_articles"],
        cited_articles=result["cited_articles"],
        is_abstention=result["is_abstention"],
        hallucinated_citations=result["hallucinated_citations"],
    )

    _cache[cache_key] = response
    if len(_cache) > CACHE_MAX:
        _cache.popitem(last=False)  # evict least recently used

    return response
