"""
Phase 2: Retrieval -> Groq -> Grounded Answer + Citations.

The actual RAG loop, in plain Python, no framework:

  1. Embed the user's Arabic question (with the "query: " prefix e5
     requires - same convention validated in Phase 1).
  2. Query the persisted Chroma collection (built in embed_and_store.py)
     for the top-k most similar chunks.
  3. Build a prompt that constrains the model to answer ONLY from the
     retrieved articles, states clearly this is legal information (not
     legal advice), and must abstain with a fixed, detectable phrase if
     the answer isn't actually supported by what was retrieved.
  4. Send it to Groq, parse the answer + the article number(s) the
     model claims it used.
  5. Cross-check the model's cited articles against what was ACTUALLY
     retrieved, so a hallucinated citation (an article number the
     model mentions that was never in the retrieved context) is
     caught programmatically, not just by eyeballing the text.

Checkpoint this satisfies: any answer can be traced to the exact
article(s) that produced it, and wrong/hallucinated answers can be
explained rather than just observed.

Requires a .env file with:
    GROQ_API_KEY=gsk_...

Run:
    python src/pipeline/ask.py
"""

import json
import os
import re
import time

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


load_dotenv()

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "labor_law_chunks"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

# Groq's recommended replacement for the now-deprecated
# llama-3.3-70b-versatile (deprecated 2026-08-16). Override via env var
# if Groq deprecates this one too - don't hardcode a model name deep
# in the prompt logic below.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

TOP_K = 8

# Reasoning effort for gpt-oss models on Groq ("low" | "medium" | "high").
# This task is extractive (find the article, restate it, cite it), so
# "low" cuts the hidden thinking time on every question. Set
# GROQ_REASONING_EFFORT="" to disable the parameter entirely (e.g. if you
# switch to a model that does not support it). Re-run the Phase 3 eval
# after changing this.
REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

# Cap on generated tokens. For reasoning models this includes the hidden
# reasoning tokens, so keep it generous to avoid truncated answers.
MAX_COMPLETION_TOKENS = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "1500"))

# Fixed, exact phrase the model must use to abstain. Kept as a
# constant (not just described in the prompt) so code can check for
# it programmatically in Phase 3's evaluation, rather than relying on
# fuzzy text matching.
ABSTAIN_PHRASE = "غير موجود في القانون المفهرس"

SYSTEM_PROMPT = f"""أنت مساعد معلومات قانونية متخصص في قانون العمل المصري رقم 14 لسنة 2025.

قواعد صارمة يجب اتباعها:

1. أجب فقط بناءً على نصوص المواد المرفقة أدناه. لا تستخدم أي معرفة
   قانونية عامة أو معلومات من خارج هذه المواد، حتى لو كانت تبدو صحيحة.

2. إذا لم تكن الإجابة موجودة بوضوح في المواد المرفقة، يجب أن تجيب
   حرفياً بالعبارة التالية فقط: "{ABSTAIN_PHRASE}"
   لا تخمن ولا تحاول الاستنتاج من سياق غير مباشر.

3. في نهاية كل إجابة (ما عدا حالة عدم الوجود)، اذكر بالضبط أرقام
   المواد التي استخدمتها بهذا الشكل تماماً في سطر منفصل:
   المصدر: مادة X أو المواد: X، Y

4. هذا النظام يقدم معلومات قانونية عامة فقط، وليس استشارة قانونية.
   ذكّر المستخدم بذلك بإيجاز في نهاية أي إجابة فعلية (ليس عند
   الامتناع عن الإجابة).

5. لا تدمج معلومات من مادتين لتكوّن قاعدة جديدة غير منصوص عليها
   حرفياً في أي منهما.
"""


def load_resources():
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    groq_client = Groq()  # reads GROQ_API_KEY from env automatically
    return embed_model, collection, groq_client


def retrieve(embed_model, collection, question, top_k=TOP_K):
    query_embedding = embed_model.encode(f"query: {question}")

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({
            "article_number": meta["article_number"],
            "chunk_text": doc,
            "distance": dist,
        })
    return retrieved


def build_user_prompt(question, retrieved_chunks):
    articles_block = "\n\n".join(
        f"[مادة {c['article_number']}]\n{c['chunk_text']}"
        for c in retrieved_chunks
    )
    return f"""المواد المسترجعة:

{articles_block}

سؤال المستخدم: {question}"""


def call_groq(groq_client, question, retrieved_chunks):
    user_prompt = build_user_prompt(question, retrieved_chunks)

    kwargs = {}
    if REASONING_EFFORT:
        kwargs["reasoning_effort"] = REASONING_EFFORT

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,  # low temperature - this is a grounded-fact task, not creative writing
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        **kwargs,
    )
    return response.choices[0].message.content


def extract_cited_articles(answer_text):
    """
    Pulls out article numbers the model claims it used, from a line
    like 'المصدر: مادة 90' or 'المصدر: المواد: 90، 117'. Returns a set
    of ints. Best-effort - the model is instructed to follow this
    format, but this doesn't assume it always will.

    Deliberately scoped to ONLY the "المصدر" line itself (matched up
    to the next newline), not everything after its first occurrence.
    The citation line doesn't always come last - the disclaimer
    sentence sometimes follows it (see real Q2/Q3/Q4 outputs) - so
    grabbing "everything after المصدر" would silently sweep up any
    digit that happened to appear later in the response, e.g. in a
    disclaimer, a different sentence, or a second mention. Scoping to
    one line removes that failure mode entirely, regardless of where
    the citation line falls in the response.

    Matches BOTH Latin digits (0-9) and the Arabic-Indic / Extended
    Arabic-Indic digits (٠-٩ / ۰-۹) documented in chunck_articles.py -
    every real response so far happened to use Latin digits, but the
    model is generating Arabic text from a corpus that itself mixes
    both digit systems, so nothing guarantees it always will.

    Scans ALL lines containing "المصدر", not just the first. Real
    runs show the model sometimes cites one article, sometimes
    several - so far always grouped onto a single line (e.g.
    "المصدر: مادة 117، مادة 119"), but nothing in the prompt actually
    forces that; a version that only checked the first matching line
    would silently drop any citation the model put on a second
    "المصدر" line instead.
    """
    lines = re.findall(r'المصدر[^\n]*', answer_text)
    if not lines:
        return set()

    # Normalize Arabic-Indic / Extended Arabic-Indic digits to Latin
    # before extracting, so "١٢٤" is read as 124 instead of being
    # silently ignored by a Latin-only \d+ pattern.
    digit_map = {}
    for i, ch in enumerate("٠١٢٣٤٥٦٧٨٩"):  # Arabic-Indic
        digit_map[ch] = str(i)
    for i, ch in enumerate("۰۱۲۳۴۵۶۷۸۹"):  # Extended Arabic-Indic (Persian)
        digit_map[ch] = str(i)

    all_numbers = set()
    for line in lines:
        normalized = "".join(digit_map.get(ch, ch) for ch in line)
        all_numbers.update(int(n) for n in re.findall(r'\d+', normalized))

    return all_numbers


def ask(embed_model, collection, groq_client, question, top_k=TOP_K, verbose=True):
    t0 = time.perf_counter()
    retrieved_chunks = retrieve(embed_model, collection, question, top_k)
    t1 = time.perf_counter()
    retrieved_article_numbers = {c["article_number"] for c in retrieved_chunks}

    answer_text = call_groq(groq_client, question, retrieved_chunks)
    t2 = time.perf_counter()
    timing = {"retrieve_s": round(t1 - t0, 3), "groq_s": round(t2 - t1, 3)}

    is_abstention = ABSTAIN_PHRASE in answer_text
    cited_articles = set() if is_abstention else extract_cited_articles(answer_text)

    # The core hallucination check: did the model cite an article
    # number that was never even in the retrieved context? If so, it
    # didn't "find" that citation in what it was given - it invented
    # it, which is exactly the failure mode Phase 3 needs to catch.
    hallucinated = cited_articles - retrieved_article_numbers

    result = {
        "question": question,
        "answer": answer_text,
        "retrieved_articles": sorted(retrieved_article_numbers),
        "cited_articles": sorted(cited_articles),
        "is_abstention": is_abstention,
        "hallucinated_citations": sorted(hallucinated),
        "timing": timing,
    }

    print(f"[timing] retrieve {timing['retrieve_s']:.2f}s | groq {timing['groq_s']:.2f}s")

    if verbose:
        print(f"\nQ: {question}")
        print(f"Retrieved articles (top {top_k}): {result['retrieved_articles']}")
        print(f"\n{answer_text}")
        if is_abstention:
            print("\n[ABSTAINED]")
        elif hallucinated:
            print(f"\n[WARNING] Model cited article(s) not in retrieved context: {sorted(hallucinated)}")
        else:
            print(f"\n[OK] All cited articles ({result['cited_articles']}) were actually retrieved.")

    return result


def main():
    embed_model, collection, groq_client = load_resources()

    # Same known questions used throughout Phase 1, plus one
    # deliberately out-of-scope question to test abstention.
    test_questions = [
        "كم مدة فترة الاختبار المسموح بها للعامل؟",
        "ما هو الحد الأقصى لساعات العمل اليومية؟",
        "كم مدة إجازة الوضع للعاملة الحامل؟",
        "ما هي مدة الإجازة السنوية للعامل؟",
        "ما هي عقوبة السرقة في قانون العقوبات؟",  # out of scope - should abstain
    ]

    results_path = "data/processed/phase2_smoke_test_results.json"
    all_results = []

    for i, q in enumerate(test_questions, start=1):
        try:
            result = ask(embed_model, collection, groq_client, q)
        except Exception as e:
            # A single failed call (rate limit, timeout, transient
            # network error) should not cost you every question that
            # already succeeded before it - especially once this same
            # loop is reused for Phase 3's 15-20 question batch, where
            # a mid-run failure was previously silent data loss.
            print(f"\n[ERROR] Question {i} failed: {q}")
            print(f"        {type(e).__name__}: {e}")
            result = {
                "question": q,
                "answer": None,
                "error": str(e),
            }
        all_results.append(result)

        # Save after every question, not just at the end, so a crash
        # on question 15 of 20 still leaves the first 14 on disk.
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)

    n_errors = sum(1 for r in all_results if r.get("answer") is None)
    print(f"\nSaved {len(all_results)} results to {results_path}"
          f"{f' ({n_errors} failed - see error field)' if n_errors else ''}")


if __name__ == "__main__":
    main()