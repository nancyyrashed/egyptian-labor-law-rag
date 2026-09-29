"""
Parse Articles 1-298 into structured records.

Reuses the exact heading-detection regex and page-tracking logic
already validated in src/validation/check_article_headings.py, so
the article boundaries here are guaranteed consistent with what was
already confirmed: 298 headings, 1-298, no gaps, no duplicates.

For each heading found, everything between it and the next heading
(or end of file) becomes that article's text.

Output: one JSON file containing a list of records:
    {
        "article_number": 27,
        "article_text": "...",
        "law_name": "Labor Law No. 14 of 2025",
        "source": "Official Gazette",
        "page": 21
    }
"""

import json
import re


INPUT_PATH = "data/processed/labor_law_clean_final_v2.txt"
OUTPUT_PATH = "data/processed/labor_law_articles.json"

LAW_NAME = "Labor Law No. 14 of 2025"
SOURCE = "Official Gazette"

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# Identical to the validated pattern in check_article_headings.py
ARTICLE_HEADING_PATTERN = re.compile(
    r"^\s*مادة\s*\(\s*([٠-٩۰-۹0-9]+)\s*\)\s*:?\s*$"
)

PAGE_MARKER_PATTERN = re.compile(r"^===== PAGE (\d+) =====$")


def normalize_digits(value):
    value = value.translate(ARABIC_DIGITS)
    value = value.translate(PERSIAN_DIGITS)
    return int(value)


def find_headings(lines):
    """
    Returns a list of dicts: {article_number, line_index, page}
    line_index is 0-based, matching the `lines` list itself.
    """
    headings = []
    current_page = None

    for line_index, line in enumerate(lines):
        page_match = PAGE_MARKER_PATTERN.match(line.strip())
        if page_match:
            current_page = int(page_match.group(1))
            continue

        heading_match = ARTICLE_HEADING_PATTERN.match(line)
        if heading_match:
            article_number = normalize_digits(heading_match.group(1))
            headings.append({
                "article_number": article_number,
                "line_index": line_index,
                "page": current_page,
            })

    return headings


def extract_article_text(lines, start_line_index, end_line_index):
    """
    Everything strictly between the heading line itself and the
    start of the next heading (or end of file), joined and
    whitespace-trimmed. Page-marker lines inside this range are
    dropped, since an article's body can legitimately span a page
    boundary and the marker line itself isn't part of the legal text.
    """
    body_lines = lines[start_line_index + 1:end_line_index]

    body_lines = [
        line for line in body_lines
        if not PAGE_MARKER_PATTERN.match(line.strip())
    ]

    text = "".join(body_lines).strip()
    return text


def parse_articles():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    headings = find_headings(lines)

    print(f"Found {len(headings)} article headings.")

    records = []

    for i, heading in enumerate(headings):
        start = heading["line_index"]

        if i + 1 < len(headings):
            end = headings[i + 1]["line_index"]
        else:
            end = len(lines)

        article_text = extract_article_text(lines, start, end)

        records.append({
            "article_number": heading["article_number"],
            "article_text": article_text,
            "law_name": LAW_NAME,
            "source": SOURCE,
            "page": heading["page"],
        })

    return records


def run_sanity_checks(records):
    print()
    print("--- SANITY CHECKS ---")

    numbers = [r["article_number"] for r in records]
    print("Total records:", len(records))
    print("Expected: 298 |", "OK" if len(records) == 298 else "MISMATCH")

    unique_numbers = sorted(set(numbers))
    missing = [n for n in range(1, 299) if n not in unique_numbers]
    print("Missing article numbers:", missing if missing else "None")

    duplicates = sorted(n for n in set(numbers) if numbers.count(n) > 1)
    print("Duplicate article numbers:", duplicates if duplicates else "None")

    empty_or_short = [
        r["article_number"] for r in records
        if len(r["article_text"]) < 5
    ]
    print(
        "Articles with suspiciously short/empty text (<5 chars):",
        empty_or_short if empty_or_short else "None"
    )

    lengths = [len(r["article_text"]) for r in records]
    print("Shortest article text length:", min(lengths))
    print("Longest article text length:", max(lengths))
    print("Average article text length:", round(sum(lengths) / len(lengths), 1))

    print()
    print("--- ARTICLE 1 (spot check) ---")
    print(records[0])
    print()
    print("--- ARTICLE 298 (spot check) ---")
    print(records[-1])


def main():
    records = parse_articles()
    run_sanity_checks(records)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print()
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()