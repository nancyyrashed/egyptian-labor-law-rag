"""
Phase 1: Token length check + hands-on embedding experiment,
using the FINAL selected embedding model.

Model selection history: paraphrase-multilingual-MiniLM-L12-v2 was
tested first and scored only 50% top-5 hit rate in a full-corpus
retrieval test (see test_retrieval.py) - it found the right topic
but not the right article for 2 of 4 real questions. Switching to
intfloat/multilingual-e5-base (trained specifically for asymmetric
query-to-passage retrieval, unlike the paraphrase models) raised
that to 100%, with an average rank of 1.8. This script now tests
the SELECTED model, not the original candidate.

Two things happen here, both using your real data:

1. We check how many TOKENS (not characters) your longest articles
   actually produce with THIS model's tokenizer. e5-base has a
   different max sequence length than MiniLM did, so the "which
   articles need splitting" list may be different from before -
   don't reuse the old MiniLM-based list.

2. The hands-on experiment: embed a few real Arabic legal sentences
   from your own corpus (some similar in meaning, one unrelated),
   print the raw vectors, and compute cosine similarity ourselves.
   e5 requires "query: " / "passage: " prefixes to work correctly -
   this is not optional, so both sentences are prefixed as passages
   for this comparison.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer


ARTICLES_PATH = "data/processed/labor_law_articles.json"
MODEL_NAME = "intfloat/multilingual-e5-base"


def load_articles():
    with open(ARTICLES_PATH, encoding="utf-8") as f:
        return json.load(f)


def check_tokenization(model, articles):
    print("=" * 60)
    print("STEP 1: TOKEN LENGTH CHECK (not just character count)")
    print("=" * 60)

    max_seq_length = model.max_seq_length
    print(f"Model's max sequence length: {max_seq_length} tokens")
    print()

    # Sort by character length, check the longest ones first
    by_length = sorted(articles, key=lambda a: len(a["article_text"]), reverse=True)

    print(f"{'Article':>8} | {'Chars':>6} | {'Tokens':>7} | Exceeds limit?")
    print("-" * 55)

    exceeds = []
    for a in by_length[:15]:
        tokens = model.tokenizer.encode(a["article_text"])
        n_tokens = len(tokens)
        flag = "YES - WILL BE TRUNCATED" if n_tokens > max_seq_length else ""
        if n_tokens > max_seq_length:
            exceeds.append(a["article_number"])
        print(f"{a['article_number']:>8} | {len(a['article_text']):>6} | {n_tokens:>7} | {flag}")

    print()
    print("Articles that will be silently truncated if NOT split:", exceeds)
    print()
    return exceeds


def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def hands_on_embedding_experiment(model, articles):
    print("=" * 60)
    print("STEP 2: HANDS-ON EMBEDDING EXPERIMENT (real corpus sentences)")
    print("=" * 60)

    # Real sentences pulled directly from your corpus.
    # A: Article 124 - the core annual leave entitlement
    # B: Article 125 - the annual leave scheduling/procedure rule
    #    (same topic as A, genuinely different wording, not a duplicate)
    # C: Article 1 - the employer ("صاحب العمل") definition, an
    #    unrelated topic, used as the negative-control comparison
    #
    # e5 requires the "passage: " prefix for correct results, even
    # in this simple pairwise comparison (not just full retrieval).
    sentence_a = "passage: يستحق العامل إجازة سنوية بأجر"
    sentence_b = "passage: يجب أن يحصل العامل على إجازة سنوية مدتها خمسة عشر يومًا"
    sentence_unrelated = "passage: صاحب العمل كل شخص طبيعى أو اعتبارى يستخدم عاملًا أو أكثر لقاء أجر"

    sentences = {
        "A (annual leave, phrasing 1)": sentence_a,
        "B (annual leave, phrasing 2)": sentence_b,
        "C (unrelated - employer definition)": sentence_unrelated,
    }

    embeddings = {}
    for label, sentence in sentences.items():
        vec = model.encode(sentence)
        embeddings[label] = vec
        print(f"\n{label}: {sentence!r}")
        print(f"  Embedding shape: {vec.shape}")
        print(f"  First 8 values:  {np.round(vec[:8], 3)}")

    print()
    print("--- Cosine similarity (computed manually, not via a library call) ---")
    labels = list(embeddings.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            sim = cosine_similarity(embeddings[labels[i]], embeddings[labels[j]])
            print(f"  {labels[i]}  vs  {labels[j]}  ->  {sim:.4f}")

    print()
    print("Expect: A-vs-B (same meaning, different wording) should score")
    print("noticeably HIGHER than A-vs-C or B-vs-C (unrelated topic).")
    print("If it doesn't, that's a signal this model may not retrieve")
    print("well on this corpus, and we should test an alternative.")


def main():
    print(f"Loading model: {MODEL_NAME} (first run downloads it, ~1.1GB)")
    model = SentenceTransformer(MODEL_NAME)

    articles = load_articles()

    check_tokenization(model, articles)
    hands_on_embedding_experiment(model, articles)


if __name__ == "__main__":
    main()
