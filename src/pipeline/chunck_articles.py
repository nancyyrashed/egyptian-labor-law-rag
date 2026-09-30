"""
Phase 1: Chunk articles for embedding.

Strategy, based on real analysis of the corpus:

- 293 articles are short enough (under e5-base's 512-token limit)
  to stay as ONE chunk each.
- 4 articles (1, 41, 79, 134) have a natural numbered-item structure
  ("١- ...", "٢- ...") and are split at those boundaries, grouping
  consecutive items to stay under a safe token budget. Each chunk
  keeps the article's opening framing line, so a chunk containing
  only item 9 still reads as "one of the worker's duties," not a
  floating fragment.
- 1 article (253) has no numbered structure at all - just prose
  paragraphs - so it's split by paragraph instead.

Numbered-item detection handles a real, confirmed quirk in this
corpus: article numbering mixes standard Arabic-Indic digits (٠-٩)
and Extended Arabic-Indic/Persian digits (۰-۹) even within the same
list (e.g. Article 1 uses ٢٧ then ۲۸ back-to-back). Both are matched.
"""

import json
import re


ARTICLES_PATH = "data/processed/labor_law_articles.json"
OUTPUT_PATH = "data/processed/labor_law_chunks.json"

# Articles confirmed (via test_embeddings.py, run against e5-base)
# to exceed the 512-token limit and need splitting.
ARTICLES_NEEDING_SPLIT = {1, 41, 79, 253, 134}

# Target character budget per chunk. Based on observed ratios in
# this corpus (~3.0-3.2 chars per e5 token), a ~1200 character
# budget stays safely under 512 tokens with margin. This is a
# character-based PROXY - see verify_with_tokenizer() below, which
# does the real check if sentence-transformers is available.
CHAR_BUDGET = 1200

ITEM_PATTERN = re.compile(r'(?:^|\n)\s*([٠-٩۰-۹]{1,3})\s*[-–]\s*')


def load_articles():
    with open(ARTICLES_PATH, encoding="utf-8") as f:
        return json.load(f)


def split_into_items(text):
    """
    Returns (opening_line, [(item_number, item_text), ...]).
    opening_line is everything before the first numbered item.
    """
    matches = list(ITEM_PATTERN.finditer(text))

    if not matches:
        return text, []

    opening_line = text[:matches[0].start()].strip()

    items = []
    for i, m in enumerate(matches):
        item_number = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        item_text = text[start:end].strip()
        items.append((item_number, item_text))

    return opening_line, items


def group_items_by_budget(opening_line, items, char_budget):
    groups = []
    current_group = []
    current_length = len(opening_line)

    for item_number, item_text in items:
        item_length = len(item_text)

        # NEW: if this single item alone would blow the budget even in
        # its own empty group, split it by sentence instead of keeping
        # it whole.
        if len(opening_line) + item_length > char_budget:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_length = len(opening_line)

            sentences = split_into_paragraphs(item_text)
            sub_groups = group_paragraphs_by_budget(sentences, char_budget - len(opening_line))
            for sg in sub_groups:
                groups.append([(item_number, " ".join(sg))])
            continue

        if current_group and current_length + item_length > char_budget:
            groups.append(current_group)
            current_group = []
            current_length = len(opening_line)

        current_group.append((item_number, item_text))
        current_length += item_length

    if current_group:
        groups.append(current_group)

    return groups


def split_into_paragraphs(text):
    """
    Splits into actual SENTENCES, not raw PDF lines. Lines in this
    corpus are wrapped at whatever width the original PDF used, not
    at sentence boundaries - e.g. one real sentence can end with
    "وجود خطر" on one line and continue "داهم على صحة..." on the
    next. Joining all lines back into continuous text first (undoing
    the PDF's line wrapping), then splitting on the actual sentence-
    ending period, gives real semantic units instead of arbitrary
    line-wrap fragments.
    """
    continuous_text = " ".join(
        line.strip() for line in text.split("\n") if line.strip()
    )

    # Arabic sentences in this corpus end with " ." (space + period).
    # Split there, keeping the period attached to each sentence.
    raw_sentences = re.split(r'(?<=\s\.)|(?<=\.)\s+', continuous_text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    return sentences


def group_paragraphs_by_budget(paragraphs, char_budget):
    groups = []
    current_group = []
    current_length = 0

    for p in paragraphs:
        if current_group and current_length + len(p) > char_budget:
            groups.append(current_group)
            current_group = []
            current_length = 0
        current_group.append(p)
        current_length += len(p)

    if current_group:
        groups.append(current_group)

    return groups


def chunk_article(article):
    """Returns a list of chunk dicts for a single article record."""
    article_number = article["article_number"]
    text = article["article_text"]
    base_metadata = {
        "article_number": article_number,
        "law_name": article["law_name"],
        "source": article["source"],
        "page": article["page"],
    }

    if article_number not in ARTICLES_NEEDING_SPLIT:
        return [{
            **base_metadata,
            "chunk_index": 0,
            "item_range": None,
            "chunk_text": text,
        }]

    opening_line, items = split_into_items(text)

    if items:
        groups = group_items_by_budget(opening_line, items, CHAR_BUDGET)
        chunks = []
        for i, group in enumerate(groups):
            item_numbers = [num for num, _ in group]
            item_texts = "\n".join(f"{num}- {txt}" for num, txt in group)
            chunk_text = f"{opening_line}\n{item_texts}".strip()
            chunks.append({
                **base_metadata,
                "chunk_index": i,
                "item_range": f"{item_numbers[0]}-{item_numbers[-1]}" if len(item_numbers) > 1 else item_numbers[0],
                "chunk_text": chunk_text,
            })
        return chunks

    # No numbered items found (e.g. Article 253) - split by paragraph
    paragraphs = split_into_paragraphs(text)
    groups = group_paragraphs_by_budget(paragraphs, CHAR_BUDGET)
    chunks = []
    for i, group in enumerate(groups):
        chunks.append({
            **base_metadata,
            "chunk_index": i,
            "item_range": None,
            "chunk_text": "\n".join(group),
        })
    return chunks


def run_sanity_checks(all_chunks, articles):
    print("--- SANITY CHECKS ---")
    print("Total chunks:", len(all_chunks))

    single_chunk_articles = sum(1 for a in articles if a["article_number"] not in ARTICLES_NEEDING_SPLIT)
    print(f"Articles left as 1 chunk: {single_chunk_articles} (expected 293)")

    for article_number in sorted(ARTICLES_NEEDING_SPLIT):
        n_chunks = sum(1 for c in all_chunks if c["article_number"] == article_number)
        print(f"Article {article_number} split into {n_chunks} chunks")

    lengths = [len(c["chunk_text"]) for c in all_chunks]
    over_budget = [c for c in all_chunks if len(c["chunk_text"]) > CHAR_BUDGET * 1.5]
    print(f"\nLongest chunk: {max(lengths)} chars")
    print(f"Chunks more than 50% over budget ({int(CHAR_BUDGET*1.5)} chars): {len(over_budget)}")
    if over_budget:
        for c in over_budget:
            print(f"  Article {c['article_number']}, chunk {c['chunk_index']}: {len(c['chunk_text'])} chars")


def verify_with_tokenizer(all_chunks):
    """
    Optional real verification against e5-base's actual tokenizer,
    if sentence-transformers is installed. The CHAR_BUDGET above is
    a proxy - this confirms whether it actually held.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n(sentence-transformers not installed here - skipping real")
        print(" tokenizer verification. The char-based budget above is a")
        print(" proxy; run this again in an environment with it installed")
        print(" to confirm no chunk actually exceeds 512 tokens.)")
        return

    print("\n--- REAL TOKENIZER VERIFICATION (e5-base) ---")
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    max_len = model.max_seq_length

    exceeds = []
    for c in all_chunks:
        n_tokens = len(model.tokenizer.encode(c["chunk_text"]))
        if n_tokens > max_len:
            exceeds.append((c["article_number"], c["chunk_index"], n_tokens))

    if exceeds:
        print(f"WARNING: {len(exceeds)} chunks still exceed {max_len} tokens:")
        for article_number, chunk_index, n_tokens in exceeds:
            print(f"  Article {article_number}, chunk {chunk_index}: {n_tokens} tokens")
    else:
        print(f"All {len(all_chunks)} chunks are within the {max_len}-token limit.")


def main():
    articles = load_articles()

    all_chunks = []
    for article in articles:
        all_chunks.extend(chunk_article(article))

    run_sanity_checks(all_chunks, articles)
    verify_with_tokenizer(all_chunks)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
