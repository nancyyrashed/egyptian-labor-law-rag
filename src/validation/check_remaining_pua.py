import pdfplumber
import unicodedata
from collections import Counter


PDF_PATH = "data/raw/labor_law_14_2025.pdf"


print("--- INVESTIGATING REMAINING PRIVATE USE CHARACTERS ---")


# ---------------------------------------------------------
# Extract the raw PDF text
# ---------------------------------------------------------

with pdfplumber.open(PDF_PATH) as pdf:

    all_text = []

    for page in pdf.pages:

        text = page.extract_text()

        if text:
            all_text.append(text)


raw_text = "\n".join(all_text)


# ---------------------------------------------------------
# Apply only NFKC normalization
# ---------------------------------------------------------
# We intentionally do NOT apply BiDi or our previous PUA
# replacements here.
#
# We want to investigate exactly which Private Use
# characters are produced directly by the PDF extraction.

normalized_text = unicodedata.normalize(
    "NFKC",
    raw_text
)


# ---------------------------------------------------------
# Find all Private Use Area characters
# ---------------------------------------------------------

pua_counter = Counter()

for char in normalized_text:

    if "\uE000" <= char <= "\uF8FF":

        pua_counter[char] += 1


print("\n--- PRIVATE USE CHARACTERS FOUND ---")

print(
    "Unique PUA characters:",
    len(pua_counter)
)

print(
    "Total PUA occurrences:",
    sum(pua_counter.values())
)


# ---------------------------------------------------------
# Display each PUA character
# ---------------------------------------------------------

for char, count in pua_counter.most_common():

    codepoint = f"U+{ord(char):04X}"

    print(
        f"\nCharacter: {repr(char)}"
    )

    print(
        f"Code point: {codepoint}"
    )

    print(
        "Unicode name:",
        unicodedata.name(char, "UNKNOWN")
    )

    print(
        "Occurrences:",
        count
    )


    # -----------------------------------------------------
    # Show examples of where the character occurs
    # -----------------------------------------------------

    print("Contexts:")

    shown = 0

    for index, current_char in enumerate(normalized_text):

        if current_char == char:

            start = max(
                0,
                index - 40
            )

            end = min(
                len(normalized_text),
                index + 41
            )

            context = normalized_text[start:end]

            print(
                repr(context)
            )

            shown += 1

            # Three examples are enough for investigation.
            if shown >= 3:
                break