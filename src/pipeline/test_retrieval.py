"""
Phase 1: Real retrieval test across the full corpus.

Instead of comparing 3 isolated sentences, this embeds ALL 298 real
articles and checks where the CORRECT article actually ranks when a
realistic question is compared against the whole corpus. This is a
much closer simulation of real retrieval than a sentence-pair test.

Tests two candidate embedding models side by side, so we can compare
them on real data before committing to one.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer


ARTICLES_PATH = "data/processed/labor_law_articles.json"

MODELS_TO_TEST = [
    "paraphrase-multilingual-MiniLM-L12-v2",      # candidate 1 - rejected, 50% hit rate
    "paraphrase-multilingual-mpnet-base-v2",       # candidate 2 - rejected, 50% hit rate
    "intfloat/multilingual-e5-base",               # SELECTED - 100% hit rate, avg rank 1.8
]

# e5 models are trained for ASYMMETRIC retrieval (short query vs
# longer passage) and require "query: " / "passage: " prefixes at
# encode time - this isn't optional, skipping it measurably hurts
# their performance. The other two models don't need any prefix.
E5_MODELS = {"intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large"}

# Real questions, phrased naturally (not copied from the article
# text), with the correct article number verified by manually
# reading the corpus.
TEST_QUESTIONS = [
    {
        "question": "كم مدة فترة الاختبار المسموح بها للعامل؟",
        "correct_article": 90,
    },
    {
        "question": "ما هو الحد الأقصى لساعات العمل اليومية؟",
        "correct_article": 117,
    },
    {
        "question": "كم مدة إجازة الوضع للعاملة الحامل؟",
        "correct_article": 54,
    },
    {
        "question": "ما هي مدة الإجازة السنوية للعامل؟",
        "correct_article": 124,
    },
]


def load_articles():
    with open(ARTICLES_PATH, encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity_matrix(query_vec, corpus_vecs):
    query_norm = query_vec / np.linalg.norm(query_vec)
    corpus_norms = corpus_vecs / np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
    return corpus_norms @ query_norm


def test_model(model_name, articles):
    print("=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    model = SentenceTransformer(model_name)
    is_e5 = model_name in E5_MODELS

    article_numbers = [a["article_number"] for a in articles]
    article_texts = [a["article_text"] for a in articles]

    if is_e5:
        # e5 requires this exact prefix convention for passages
        article_texts_for_encoding = [f"passage: {t}" for t in article_texts]
    else:
        article_texts_for_encoding = article_texts

    print(f"Embedding all {len(article_texts)} articles...")
    corpus_embeddings = model.encode(
        article_texts_for_encoding, show_progress_bar=True, batch_size=32
    )

    results = []

    for test_case in TEST_QUESTIONS:
        question = test_case["question"]
        correct_article = test_case["correct_article"]

        query_text = f"query: {question}" if is_e5 else question
        query_embedding = model.encode(query_text)
        similarities = cosine_similarity_matrix(query_embedding, corpus_embeddings)

        ranked_indices = np.argsort(similarities)[::-1]
        ranked_articles = [article_numbers[i] for i in ranked_indices]

        rank_of_correct = ranked_articles.index(correct_article) + 1
        top_5 = ranked_articles[:5]

        results.append({
            "question": question,
            "correct_article": correct_article,
            "rank_of_correct": rank_of_correct,
            "top_5": top_5,
            "found_in_top_5": correct_article in top_5,
        })

        print(f"\nQ: {question}")
        print(f"   Correct article: {correct_article}")
        print(f"   Rank of correct article: {rank_of_correct} (out of 298)")
        print(f"   Top 5 retrieved: {top_5}")
        print(f"   Found in top 5: {'YES' if correct_article in top_5 else 'NO'}")

    hit_rate = sum(r["found_in_top_5"] for r in results) / len(results)
    print(f"\n--- {model_name} summary ---")
    print(f"Top-5 hit rate: {hit_rate:.0%} ({sum(r['found_in_top_5'] for r in results)}/{len(results)})")

    return results


def main():
    articles = load_articles()

    all_results = {}
    for model_name in MODELS_TO_TEST:
        all_results[model_name] = test_model(model_name, articles)

    print("\n")
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    for model_name, results in all_results.items():
        hit_rate = sum(r["found_in_top_5"] for r in results) / len(results)
        avg_rank = sum(r["rank_of_correct"] for r in results) / len(results)
        print(f"{model_name}")
        print(f"   Top-5 hit rate: {hit_rate:.0%}")
        print(f"   Average rank of correct article: {avg_rank:.1f}")


if __name__ == "__main__":
    main()