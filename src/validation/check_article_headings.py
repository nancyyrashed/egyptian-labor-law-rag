import re

INPUT_PATH = "data/processed/labor_law_clean_final_v2.txt"


ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩",
    "0123456789"
)

PERSIAN_DIGITS = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹",
    "0123456789"
)


def normalize_digits(value):
    value = value.translate(ARABIC_DIGITS)
    value = value.translate(PERSIAN_DIGITS)
    return int(value)


# Matches actual article headings such as:
# مادة (١)
# مادة (13) :
# مادة (۲۹۸) :
ARTICLE_HEADING_PATTERN = re.compile(
    r"^\s*مادة\s*\(\s*([٠-٩۰-۹0-9]+)\s*\)\s*:?\s*$"
)


def check_article_headings():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    headings = []

    current_page = None

    for line_number, line in enumerate(lines, start=1):

        # Track page markers such as:
        # ===== PAGE 57 =====
        page_match = re.match(
            r"^===== PAGE (\d+) =====$",
            line.strip()
        )

        if page_match:
            current_page = int(page_match.group(1))
            continue

        match = ARTICLE_HEADING_PATTERN.match(line)

        if match:
            article_number = normalize_digits(match.group(1))

            headings.append({
                "article": article_number,
                "line": line_number,
                "page": current_page,
                "text": line.strip(),
            })

    article_numbers = [item["article"] for item in headings]

    unique_articles = sorted(set(article_numbers))

    duplicates = sorted(
        number
        for number in set(article_numbers)
        if article_numbers.count(number) > 1
    )

    missing = [
        number
        for number in range(1, 299)
        if number not in unique_articles
    ]

    print("=" * 70)
    print("ARTICLE HEADING VALIDATION")
    print("=" * 70)

    print()
    print("Actual article headings found:", len(headings))
    print("Unique article numbers:", len(unique_articles))

    if unique_articles:
        print("First article:", min(unique_articles))
        print("Last article:", max(unique_articles))

    print()
    print("-" * 70)
    print("EXPECTED RANGE")
    print("-" * 70)

    print("Expected articles: 1-298")
    print("Expected count: 298")
    print("Actual unique count:", len(unique_articles))

    print()
    print("-" * 70)
    print("MISSING ARTICLES")
    print("-" * 70)

    if missing:
        print("Missing article numbers:")
        print(missing)
    else:
        print("None")

    print()
    print("-" * 70)
    print("DUPLICATE ARTICLE HEADINGS")
    print("-" * 70)

    if duplicates:
        for number in duplicates:
            matches = [
                item for item in headings
                if item["article"] == number
            ]

            print(f"Article {number}:")
            for item in matches:
                print(
                    f"  Page {item['page']}, "
                    f"line {item['line']}: "
                    f"{item['text']}"
                )
    else:
        print("None")

    print()
    print("-" * 70)
    print("ARTICLE HEADING LOCATIONS")
    print("-" * 70)

    for item in headings:
        print(
            f"Article {item['article']:>3} "
            f"| Page {item['page']:>3} "
            f"| Line {item['line']}"
        )

    print()
    print("=" * 70)
    print("ARTICLE HEADING VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    check_article_headings()