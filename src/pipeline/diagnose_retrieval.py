"""
Diagnostic: for one question, show where the EXPECTED article
actually ranked, beyond just the top-5 ask.py normally returns.

Reuses embed_model/collection loading from ask.py directly. Not a
permanent part of the pipeline - a one-off tool for investigating
the Article 156 retrieval miss found in Phase 3.

Run:
    python diagnose_retrieval.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "pipeline"))
from ask import load_resources  # noqa: E402


QUESTION = "ما هي مدة مهلة الإخطار بإنهاء عقد العمل غير محدد المدة؟"
EXPECTED_ARTICLE = 156
N_RESULTS = 30  # cast a much wider net than the normal top_k=5


def main():
    embed_model, collection, _ = load_resources()

    query_embedding = embed_model.encode(f"query: {QUESTION}")
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=N_RESULTS,
    )

    print(f"Question: {QUESTION}")
    print(f"Expected article: {EXPECTED_ARTICLE}\n")
    print(f"{'Rank':<5} {'Article':<8} {'Distance':<10} {'Chunk id'}")
    print("-" * 50)

    found_rank = None
    for rank, (meta, dist, chunk_id) in enumerate(
        zip(results["metadatas"][0], results["distances"][0], results["ids"][0]),
        start=1,
    ):
        article = meta["article_number"]
        flag = "  <-- expected" if article == EXPECTED_ARTICLE else ""
        if article == EXPECTED_ARTICLE:
            found_rank = rank
        print(f"{rank:<5} {article:<8} {dist:<10.4f} {chunk_id}{flag}")

    print()
    if found_rank:
        print(f"Article {EXPECTED_ARTICLE} found at rank {found_rank} "
              f"(top_k=5 in production would have missed it).")
    else:
        print(f"Article {EXPECTED_ARTICLE} NOT found even within top {N_RESULTS} "
              "results. This points to a genuine semantic-distance issue, not "
              "just a borderline ranking one.")


if __name__ == "__main__":
    main()
