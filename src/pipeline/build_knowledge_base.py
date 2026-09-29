"""
Egyptian Labor Law — PDF Extraction & Cleaning Pipeline

Consolidates the 7 verified stages discovered during development into
one sequential pipeline. Each stage was originally its own script;
this file merges them into one pipeline.

Stages:

    1. Extract PDF text + fix Unicode compatibility forms + BiDi reorder
       + fix Private Use Area diacritic characters.

    2. Repair a specific "ف/التي" word-splitting pattern.

    3. Repair tatweel (kashida) elongation-character artifacts
       + diacritic/alif spacing artifacts.

    4. Repair broken/split words — round 1.

    5. Repair broken/split words — round 2.

    6. Repair broken/split words — round 3.

    7. Remove the repeated Egyptian Gazette header from each page.

Run this file directly to go from the raw PDF to the final cleaned
text file in one pass:

    python build_knowledge_base.py
"""

import os
import re
import unicodedata

import pdfplumber
from bidi.algorithm import get_display

from normalize_arabic import normalize_private_arabic_marks


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

PDF_PATH = "data/raw/labor_law_14_2025.pdf"
OUTPUT_PATH = "data/processed/labor_law_clean_final_v2.txt"


# ===========================================================
# STAGE 1 — Extract + normalize + reorder + fix PUA diacritics
# ===========================================================

def clean_arabic_text(text):
    """
    Apply the core Arabic cleaning steps, in the verified order:

    1. NFKC converts compatibility characters (e.g. Arabic
       Presentation Forms) into standard Unicode equivalents.

    2. BiDi processing corrects the visual/logical ordering of the
       Arabic text produced by PDF extraction.

    3. Verified Private Use Area characters are converted into
       standard Unicode Arabic diacritics.
    """

    text = unicodedata.normalize("NFKC", text)
    text = get_display(text)
    text = normalize_private_arabic_marks(text)

    return text


# ===========================================================
# STAGE 2 — Repair "ف/التي"-type word splits
# ===========================================================

STAGE_2_REPAIRS = {
    "فـ ى": "فى",
    "ف ى": "فى",
    "الت ى": "التى",
    "والت ى": "والتى",
    "يومً ا": "يومًا",
    "قرارً ا": "قرارًا",
    "ورقيً ا": "ورقيًا",
    "أ ى": "أى",
    "طرف ى": "طرفى",
    "الق ومى": "القومى",
}


def apply_stage_2(text):
    print("--- STAGE 2: REPAIRING 'ف/التي' WORD SPLITS ---")

    for old, new in STAGE_2_REPAIRS.items():

        count = text.count(old)

        if count:
            print(f"  {count:4} | {old!r} -> {new!r}")

        text = text.replace(old, new)

    return text


# ===========================================================
# STAGE 3 — Repair tatweel + diacritic/alif spacing
# ===========================================================

_DIACRITICS = (
    "\u064B"  # Fathatan
    "\u064C"  # Dammatan
    "\u064D"  # Kasratan
    "\u064E"  # Fatha
    "\u064F"  # Damma
    "\u0650"  # Kasra
    "\u0651"  # Shadda
    "\u0652"  # Sukun
    "\u0653"
    "\u0654"
    "\u0655"
    "\u0670"
)


# A diacritic followed by whitespace and an alif is an
# extraction artifact in this document.
_DIACRITIC_ALIF_PATTERN = re.compile(
    r"([" + _DIACRITICS + r"])[ \t]+ا"
)


def apply_stage_3(text):
    print(
        "--- STAGE 3: REPAIRING TATWEEL + "
        "DIACRITIC/ALIF SPACING ---"
    )

    tatweel_count = text.count("ـ")

    if tatweel_count:
        print(
            f"  Removing {tatweel_count} tatweel (ـ) characters"
        )

    text = text.replace("ـ", "")

    matches = len(
        _DIACRITIC_ALIF_PATTERN.findall(text)
    )

    if matches:
        print(
            f"  Fixing {matches} diacritic+space+alif sequences"
        )

    text = _DIACRITIC_ALIF_PATTERN.sub(
        lambda m: m.group(1) + "ا",
        text
    )

    return text

# ===========================================================
# STAGE 4 — Repair broken/split words, round 1
# ===========================================================

STAGE_4_REPAIRS = {
    "الت ى": "التى",
    "األساس ى": "األساسى",
    "األ جر": "الأجر",
    "ثمان ى": "ثمانى",
    "النقاب ى": "النقابى",
    "الع مل": "العمل",
    "الذ ى": "الذى",
    "يكو ن": "يكون",
    "مفص الً": "مفصلًا",
    "ل رئيس": "لرئيس",
    "ا لمحاكم": "المحاكم",
    "الط بية": "الطبية",
}


def apply_stage_4(text):
    print(
        "--- STAGE 4: REPAIRING BROKEN WORDS (ROUND 1) ---"
    )

    for old, new in STAGE_4_REPAIRS.items():

        count = text.count(old)

        if count:
            print(f"  {count:4} | {old!r} -> {new!r}")

        text = text.replace(old, new)

    return text


# ===========================================================
# STAGE 5 — Repair broken/split words, round 2
# ===========================================================

STAGE_5_REPAIRS = {
    "ا لقانون": "القانون",
    "مسئو الً": "مسؤولًا",
    "م رض": "مرض",
    "سالمت ها": "سلامتها",
}


def apply_stage_5(text):
    print(
        "--- STAGE 5: REPAIRING BROKEN WORDS (ROUND 2) ---"
    )

    for old, new in STAGE_5_REPAIRS.items():

        count = text.count(old)

        if count:
            print(f"  {count:4} | {old!r} -> {new!r}")

        text = text.replace(old, new)

    return text


# ===========================================================
# STAGE 6 — Repair broken/split words, round 3
# ===========================================================

STAGE_6_REPAIRS = {
    "التو فيق": "التوفيق",
    "أحكا م": "أحكام",
    "المح كوم": "المحكوم",
}


def apply_stage_6(text):
    print(
        "--- STAGE 6: REPAIRING BROKEN WORDS (ROUND 3) ---"
    )

    for old, new in STAGE_6_REPAIRS.items():

        count = text.count(old)

        if count:
            print(f"  {count:4} | {old!r} -> {new!r}")

        text = text.replace(old, new)

    return text


# ===========================================================
# STAGE 7 — Remove repeated Gazette headers
# ===========================================================

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


ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩",
    "0123456789"
)


def digits_to_int(value):
    return int(value.translate(ARABIC_DIGITS))


def remove_gazette_header(page_text, page_number):
    """
    Remove the repeated Egyptian Gazette header from one page.

    The page number must appear immediately before or after
    the known Gazette header and must match the current page.
    """

    lines = page_text.splitlines()
    cleaned_lines = []

    for line in lines:

        match = GAZETTE_HEADER_LINE.match(line)

        if match:

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

            if detected_page_number == page_number:
                continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# ===========================================================
# PDF extraction
# ===========================================================

def extract_pdf():
    print("--- STAGE 1: EXTRACTING & CLEANING PDF ---")

    with pdfplumber.open(PDF_PATH) as pdf:
        print("Number of pages:", len(pdf.pages))

        all_pages = []

        for page_number, page in enumerate(pdf.pages, start=1):

            # Page 112 contains publisher/deposit information,
            # not part of the labor law text.
            if page_number == 112:
                print(
                    "Page 112: excluded from RAG corpus "
                    "(publisher/deposit material)."
                )
                continue

            raw_text = page.extract_text()

            if not raw_text:
                print(f"Page {page_number}: no text extracted.")
                continue

            cleaned_text = clean_arabic_text(raw_text)

            # Stage 7 — Remove validated Gazette header
            cleaned_text = remove_gazette_header(
                cleaned_text,
                page_number
            )

            all_pages.append({
                "page": page_number,
                "text": cleaned_text
            })

    return all_pages


# ===========================================================
# Convert pages to one text blob
# ===========================================================

def pages_to_text(pages):
    """
    Join extracted pages into one text blob with page markers.
    """

    parts = []

    for page in pages:

        parts.append(
            f"\n\n===== PAGE {page['page']} =====\n\n"
        )

        parts.append(page["text"])

    return "".join(parts)


# ===========================================================
# Main pipeline
# ===========================================================

def main():

    pages = extract_pdf()

    text = pages_to_text(pages)

    text = apply_stage_2(text)

    text = apply_stage_3(text)

    text = apply_stage_4(text)

    text = apply_stage_5(text)

    text = apply_stage_6(text)

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(text)

    print("\n--- PIPELINE COMPLETE ---")
    print("Pages processed:", len(pages))
    print("Final character count:", len(text))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()