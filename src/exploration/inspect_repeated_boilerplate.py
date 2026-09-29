from pathlib import Path
import re


INPUT_FILE = Path(
    "data/processed/labor_law_clean_final_v2.txt"
)


# ---------------------------------------------------------
# Load corpus
# ---------------------------------------------------------

text = INPUT_FILE.read_text(encoding="utf-8")


# ---------------------------------------------------------
# Split document into pages
# ---------------------------------------------------------

page_pattern = re.compile(
    r"===== PAGE (\d+) =====\n"
)

matches = list(page_pattern.finditer(text))

pages = {}

for i, match in enumerate(matches):
    page_number = int(match.group(1))

    start = match.end()

    if i + 1 < len(matches):
        end = matches[i + 1].start()
    else:
        end = len(text)

    pages[page_number] = text[start:end]


# ---------------------------------------------------------
# Arabic digit conversion
# ---------------------------------------------------------

ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩",
    "0123456789"
)


def digits_to_int(value):
    """
    Convert Arabic-Indic or Western digits to an integer.
    """
    return int(value.translate(ARABIC_DIGITS))


# ---------------------------------------------------------
# Gazette header pattern
# ---------------------------------------------------------

GAZETTE_HEADER_BASE = (
    r"الجریدة\s+الرسمیة"
    r"\s*[-–—]\s*"
    r"العدد\s*١٨"
    r"\s*\(تابع\)"
    r"\s*فى\s*٣"
    r"\s*مایو"
    r"\s*سنة\s*٢٠٢٥"
)


GAZETTE_HEADER_LINE = re.compile(
    rf"^\s*"
    rf"(?:([٠-٩0-9]+)\s+)?"
    rf"{GAZETTE_HEADER_BASE}"
    rf"(?:\s+([٠-٩0-9]+))?"
    rf"\s*$"
)


# ---------------------------------------------------------
# Investigate Gazette headers
# ---------------------------------------------------------

print("=" * 70)
print("REPEATED GAZETTE HEADER INVESTIGATION")
print("=" * 70)

print(f"\nPages found: {len(pages)}")


matched_pages = []
unmatched_pages = []
invalid_page_numbers = []

header_examples = []


for page_number in sorted(pages):

    lines = [
        line.strip()
        for line in pages[page_number].splitlines()
        if line.strip()
    ]

    page_matches = []

    for line_number, line in enumerate(lines, start=1):

        match = GAZETTE_HEADER_LINE.match(line)

        if match:
            page_matches.append(
                (line_number, line, match)
            )

    if page_matches:

        matched_pages.append(page_number)

        for line_number, line, match in page_matches:

            leading_page = match.group(1)
            trailing_page = match.group(2)

            detected_page_number = None

            if leading_page:
                detected_page_number = digits_to_int(
                    leading_page
                )

            if trailing_page:
                detected_page_number = digits_to_int(
                    trailing_page
                )

            header_examples.append(
                (
                    page_number,
                    line_number,
                    line,
                    detected_page_number
                )
            )

            # Validate that the number beside the header
            # is actually the current PDF page number.
            if detected_page_number != page_number:
                invalid_page_numbers.append(
                    (
                        page_number,
                        line_number,
                        line,
                        detected_page_number
                    )
                )

    else:
        unmatched_pages.append(page_number)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("GAZETTE HEADER RESULTS")
print("-" * 70)

print(
    f"\nPages containing the expected Gazette header: "
    f"{len(matched_pages)} / {len(pages)}"
)

print(
    f"Pages without the expected Gazette header: "
    f"{len(unmatched_pages)}"
)

print(
    f"Header lines detected: "
    f"{len(header_examples)}"
)


# ---------------------------------------------------------
# Pages without header
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("PAGES WITHOUT EXPECTED HEADER")
print("-" * 70)

if unmatched_pages:
    print(unmatched_pages)
else:
    print("None")


# ---------------------------------------------------------
# Invalid page-number checks
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("PAGE NUMBER VALIDATION")
print("-" * 70)

if invalid_page_numbers:

    print(
        f"\nFound {len(invalid_page_numbers)} "
        "header(s) where the detected page number "
        "does not match the current page."
    )

    for (
        page_number,
        line_number,
        line,
        detected_page_number
    ) in invalid_page_numbers:

        print(
            f"\nPAGE {page_number}, LINE {line_number}"
        )

        print(
            f"Detected page number: "
            f"{detected_page_number}"
        )

        print(f"Line: {line}")

else:

    print(
        "\nAll detected header page numbers "
        "match their corresponding page numbers."
    )


# ---------------------------------------------------------
# Show representative examples
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("HEADER EXAMPLES")
print("-" * 70)

for (
    page_number,
    line_number,
    line,
    detected_page_number
) in header_examples[:10]:

    print(
        f"\nPAGE {page_number}, LINE {line_number}"
    )

    print(f"Line: {line}")

    print(
        f"Detected page number: "
        f"{detected_page_number}"
    )


# ---------------------------------------------------------
# Final conclusion
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("INVESTIGATION COMPLETE")
print("=" * 70)