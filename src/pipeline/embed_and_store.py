"""
Phase 1: Embed the finalized chunks and store them in Chroma.

Takes labor_law_chunks.json (309 chunks, produced by chunck_articles.py
and verified against e5-base's real tokenizer) and:

  1. Encodes each chunk's text with the SELECTED model
     (intfloat/multilingual-e5-base), using the "passage: " prefix
     e5 requires for anything being stored/retrieved-against (as
     opposed to a query).
  2. Stores each chunk's embedding + full metadata (article_number,
     law_name, source, page, item_range, chunk_index) in a local,
     persistent Chroma collection.
  3. Runs a real end-to-end sanity check: embeds ALL FOUR known test
     questions from test_retrieval.py with the "query: " prefix,
     queries the ACTUAL Chroma collection (not an in-memory numpy
     comparison), and reports hit rate + average rank - the same two
     numbers test_retrieval.py reported for the raw (unchunked)
     articles (100% hit rate, avg rank 1.8) - so the two can be
     compared directly. One question passing isn't enough evidence
     that chunking didn't change retrieval behavior; all four is.

This is the step that turns "the model works in theory" (test_embeddings.py,
test_retrieval.py) into "the database I actually built works."

Run:
    python src/pipeline/embed_and_store.py
"""

import json

import chromadb
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = "data/processed/labor_law_chunks.json"
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "labor_law_chunks"
MODEL_NAME = "intfloat/multilingual-e5-base"

# The SAME four verified questions used in test_retrieval.py (against
# the raw, unchunked articles: 100% top-5 hit rate, avg rank 1.8),
# reused here to check whether chunking + real Chroma storage changed
# that picture at all. Keep these in sync with test_retrieval.py.
SANITY_QUESTIONS = [
    {"question": "كم مدة فترة الاختبار المسموح بها للعامل؟", "correct_article": 90},
    {"question": "ما هو الحد الأقصى لساعات العمل اليومية؟", "correct_article": 117},
    {"question": "كم مدة إجازة الوضع للعاملة الحامل؟", "correct_article": 54},
    {"question": "ما هي مدة الإجازة السنوية للعامل؟", "correct_article": 124},
]

# Baseline from test_retrieval.py, against the raw unchunked articles -
# printed here so the two numbers sit side by side instead of relying
# on memory of a previous console output.
BASELINE_HIT_RATE = 1.0
BASELINE_AVG_RANK = 1.8


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_chunk_id(chunk):
    """
    Stable, human-readable, guaranteed-unique id per chunk. Includes
    chunk_index so a multi-chunk article (e.g. Article 1's 8 chunks)
    doesn't collide.
    """
    return f"article_{chunk['article_number']}_chunk_{chunk['chunk_index']}"


def embed_chunks(model, chunks):
    print(f"Embedding {len(chunks)} chunks with '{MODEL_NAME}'...")
    # e5 requires the "passage: " prefix for anything being indexed/
    # retrieved-against - this is the same convention already
    # validated in test_embeddings.py and test_retrieval.py.
    texts = [f"passage: {c['chunk_text']}" for c in chunks]
    embeddings = model.encode(
        texts, show_progress_bar=True, batch_size=32
    )
    return embeddings


def build_metadata(chunk):
    """
    Chroma metadata values must be str/int/float/bool - None is not
    allowed, so item_range (which is None for non-split articles)
    needs a safe fallback.
    """
    return {
        "article_number": chunk["article_number"],
        "law_name": chunk["law_name"],
        "source": chunk["source"],
        "page": chunk["page"],
        "chunk_index": chunk["chunk_index"],
        "item_range": chunk["item_range"] if chunk["item_range"] is not None else "",
    }


def store_in_chroma(chunks, embeddings):
    print(f"\nWriting to persistent Chroma store at '{CHROMA_DIR}'...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh build each run, so re-running this script after re-chunking
    # doesn't leave stale/duplicate entries behind.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [build_chunk_id(c) for c in chunks]
    documents = [c["chunk_text"] for c in chunks]
    metadatas = [build_metadata(c) for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Stored {collection.count()} chunks in collection '{COLLECTION_NAME}'.")
    return collection


def sanity_check_query(model, collection):
    """
    Real end-to-end check: run ALL FOUR known questions against the
    ACTUAL persisted Chroma collection (not a numpy simulation), and
    report hit rate + average rank so they can be compared directly
    against test_retrieval.py's baseline on the raw, unchunked
    articles (100% hit rate, avg rank 1.8).
    """
    print("\n--- SANITY CHECK: real Chroma queries (all 4 known questions) ---")

    per_question_results = []

    for case in SANITY_QUESTIONS:
        question = case["question"]
        correct_article = case["correct_article"]

        query_embedding = model.encode(f"query: {question}")
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5,
        )

        retrieved_ids = results["ids"][0]
        retrieved_articles = [m["article_number"] for m in results["metadatas"][0]]
        retrieved_distances = results["distances"][0]

        found = correct_article in retrieved_articles
        rank = retrieved_articles.index(correct_article) + 1 if found else None

        print(f"\nQ: {question}")
        print(f"Expected article: {correct_article}")
        for cid, article_num, dist in zip(retrieved_ids, retrieved_articles, retrieved_distances):
            flag = "  <-- expected" if article_num == correct_article else ""
            print(f"  {cid:<25} article {article_num:<5} distance={dist:.4f}{flag}")

        if found:
            print(f"  -> FOUND at rank {rank}")
        else:
            print(f"  -> NOT FOUND in top 5")

        per_question_results.append({
            "question": question,
            "correct_article": correct_article,
            "found": found,
            "rank": rank,
        })

    hit_rate = sum(r["found"] for r in per_question_results) / len(per_question_results)
    ranked = [r["rank"] for r in per_question_results if r["found"]]
    avg_rank = sum(ranked) / len(ranked) if ranked else float("inf")

    hit_rate_str = f"{hit_rate:.0%}"
    avg_rank_str = f"{avg_rank:.1f}" if ranked else "n/a"

    print("\n--- SUMMARY: chunked + persisted Chroma vs. raw-article baseline ---")
    print(f"{'Metric':<20} {'This run (chunked)':<22} {'Baseline (raw articles)'}")
    print(f"{'Top-5 hit rate':<20} {hit_rate_str:<22} {BASELINE_HIT_RATE:.0%}")
    print(f"{'Avg rank (of hits)':<20} {avg_rank_str:<22} {BASELINE_AVG_RANK}")

    if hit_rate < BASELINE_HIT_RATE:
        print("\nWARNING: hit rate dropped after chunking + storage.")
        print("  Investigate which question(s) newly failed and why -")
        print("  e.g. the chunk containing the answer may have been split")
        print("  in a way that diluted the relevant text, or an item_range")
        print("  chunk lost the framing context an isolated article had.")
    elif avg_rank > BASELINE_AVG_RANK + 1:
        print("\nNOTE: hit rate held, but average rank got noticeably worse.")
        print("  Not a failure, but worth keeping top_k >= 5 in Phase 2's")
        print("  retrieval step rather than assuming the top hit is always")
        print("  the right one - multi-article context may matter more")
        print("  post-chunking than it did on whole, unsplit articles.")
    else:
        print("\nPASS - chunking + real Chroma storage held up against baseline.")


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = embed_chunks(model, chunks)
    collection = store_in_chroma(chunks, embeddings)
    sanity_check_query(model, collection)


if __name__ == "__main__":
    main()
