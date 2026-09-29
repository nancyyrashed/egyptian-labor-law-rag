import pdfplumber
import unicodedata
from bidi.algorithm import get_display
from collections import Counter


PDF_PATH = "data/raw/labor_law_14_2025.pdf"


# ---------------------------------------------------------
# 1. Extract text from several pages
# ---------------------------------------------------------

with pdfplumber.open(PDF_PATH) as pdf:

    # We inspect several pages to make sure the problem
    # is not specific to the first page.
    pages_to_check = [0, 1, 10, 20, 50, 100]

    all_text = []

    for page_number in pages_to_check:

        page = pdf.pages[page_number]

        raw_text = page.extract_text()

        if raw_text:
            all_text.append(raw_text)


# Combine the text from all selected pages.
combined_text = "\n".join(all_text)


# ---------------------------------------------------------
# 2. Apply the same cleaning steps from inspect_pdf
# ---------------------------------------------------------

# First convert compatibility characters into standard Unicode.
normalized_text = unicodedata.normalize("NFKC", combined_text)

# Then handle the bidirectional ordering of the Arabic text.
bidi_text = get_display(normalized_text)


# ---------------------------------------------------------
# 3. Find Arabic characters and count their occurrences
# ---------------------------------------------------------

arabic_characters = Counter()

for char in bidi_text:

    # Check whether the character belongs to the main
    # Arabic Unicode block.
    if "\u0600" <= char <= "\u06FF":

        arabic_characters[char] += 1


# ---------------------------------------------------------
# 4. Display the Arabic characters found
# ---------------------------------------------------------

print("--- ARABIC CHARACTERS FOUND ---")

for char, count in arabic_characters.most_common():

    print(
        repr(char),
        f"U+{ord(char):04X}",
        unicodedata.name(char, "UNKNOWN"),
        f"-> {count} occurrences"
    )

# ---------------------------------------------------------
# 5. Inspect unusual Arabic/Persian variants
# ---------------------------------------------------------

variants_to_check = {
    "ی": "Persian Yeh",
    "ھ": "Heh Doachashmee",
    "۰": "Extended Arabic-Indic Zero",
    "۱": "Extended Arabic-Indic One",
    "۲": "Extended Arabic-Indic Two",
    "۳": "Extended Arabic-Indic Three",
    "۷": "Extended Arabic-Indic Seven",
    "۸": "Extended Arabic-Indic Eight",
}


print("\n--- UNUSUAL CHARACTER CONTEXT ---")

for variant, description in variants_to_check.items():

    print(f"\n{repr(variant)} - {description}")

    found = False

    # Find every occurrence of the character.
    for index, char in enumerate(bidi_text):

        if char == variant:

            found = True

            # Show some text before and after the character
            # so we can understand how it is being used.
            start = max(0, index - 30)
            end = min(len(bidi_text), index + 31)

            context = bidi_text[start:end]

            print(f"Context: {repr(context)}")

    if not found:
        print("Not found.")

# ---------------------------------------------------------
# 6. Inspect Private Use Area characters
# ---------------------------------------------------------

print("\n--- PRIVATE USE AREA CHARACTERS ---")

for index, char in enumerate(bidi_text):

    # U+E000–U+F8FF is the Unicode Private Use Area.
    if "\uE000" <= char <= "\uF8FF":

        start = max(0, index - 50)
        end = min(len(bidi_text), index + 51)

        print(
            f"\nCharacter: {repr(char)}"
            f"\nCode point: U+{ord(char):04X}"
            f"\nContext: {repr(bidi_text[start:end])}"
        )

# ---------------------------------------------------------
# 7. Investigate Private Use Area characters
# ---------------------------------------------------------

print("\n--- PRIVATE USE AREA CHARACTER DETAILS ---")

# These are the two Private Use Area characters we found
# during the previous inspection.
private_use_chars = ["\uE821", "\uE823"]

for char in private_use_chars:

    print(f"\nCharacter: {repr(char)}")
    print(f"Code point: U+{ord(char):04X}")

    # Unicode does not assign a standard meaning to Private Use
    # Area characters, so this tells us what Unicode itself knows
    # about the character.
    print(
        "Unicode name:",
        unicodedata.name(char, "UNKNOWN")
    )

    # The Unicode category tells us what general type of character
    # this is. For example, letters are "L", marks are "M", etc.
    print(
        "Unicode category:",
        unicodedata.category(char)
    )

    # Bidirectional class tells us how the character participates
    # in right-to-left / left-to-right text processing.
    print(
        "Bidirectional class:",
        unicodedata.bidirectional(char)
    )


# ---------------------------------------------------------
# 8. Compare the characters before and after normalization
# ---------------------------------------------------------

print("\n--- RAW vs NORMALIZED TEXT AROUND PRIVATE USE CHARACTERS ---")

with pdfplumber.open(PDF_PATH) as pdf:

    # Check a few pages where the Private Use Area characters
    # were already observed.
    pages_to_check = [0, 1, 10, 20, 50, 100]

    for page_number in pages_to_check:

        raw_text = pdf.pages[page_number].extract_text()

        if not raw_text:
            continue

        # Apply the same transformations used earlier.
        normalized = unicodedata.normalize("NFKC", raw_text)
        bidi = get_display(normalized)

        for private_char in private_use_chars:

            # Look for this character in the BiDi output.
            position = bidi.find(private_char)

            if position == -1:
                continue

            print(f"\nPage {page_number + 1}")
            print(f"Character: {repr(private_char)}")

            # Show the surrounding text from the final BiDi output.
            start = max(0, position - 40)
            end = min(len(bidi), position + 41)

            print("BiDi context:")
            print(repr(bidi[start:end]))

            # Also show the same area from the normalized text.
            #
            # This helps determine whether NFKC changed the
            # character or whether it was already present in
            # the extracted PDF text.
            normalized_position = normalized.find(private_char)

            if normalized_position != -1:

                start = max(0, normalized_position - 40)
                end = min(len(normalized), normalized_position + 41)

                print("NFKC context:")
                print(repr(normalized[start:end]))

            break


# ---------------------------------------------------------
# 9. Inspect PDF character-level information
# ---------------------------------------------------------

print("\n--- PDF CHARACTER-LEVEL INSPECTION ---")

with pdfplumber.open(PDF_PATH) as pdf:

    # Start with page 1 because it is easy to reproduce.
    page = pdf.pages[0]

    # pdfplumber exposes the individual characters extracted
    # from the PDF through page.chars.
    #
    # Each character contains information such as:
    # - text
    # - font
    # - size
    # - position on the page
    #
    # This can help us determine whether E821/E823 are actually
    # missing letters or font-specific glyphs.
    for char_info in page.chars:

        char = char_info.get("text", "")

        if char in private_use_chars:

            print("\nPrivate-use glyph found:")

            print("Text:", repr(char))
            print("Code point:", f"U+{ord(char):04X}")
            print("Font:", char_info.get("fontname"))
            print("Font size:", char_info.get("size"))
            print("x0:", char_info.get("x0"))
            print("x1:", char_info.get("x1"))
            print("top:", char_info.get("top"))
            print("bottom:", char_info.get("bottom"))

            print("Full character information:")
            print(char_info)

# ---------------------------------------------------------
# 9. Inspect the actual PDF glyphs
# ---------------------------------------------------------

print("\n--- PDF CHARACTER-LEVEL INSPECTION ---")

with pdfplumber.open(PDF_PATH) as pdf:

    # These pages contain the Private Use Area characters
    # based on the investigation above.
    pages_to_check = [10, 20, 50, 100]

    for page_number in pages_to_check:

        page = pdf.pages[page_number]

        print(f"\n========== PAGE {page_number + 1} ==========")

        # page.chars contains information about every character
        # that pdfplumber extracted from this page.
        for index, char_info in enumerate(page.chars):

            char = char_info.get("text", "")

            # We only care about the two Private Use Area
            # characters we are investigating.
            if char not in private_use_chars:
                continue

            print("\nPrivate-use glyph found:")

            print(
                "Character:",
                repr(char)
            )

            print(
                "Code point:",
                f"U+{ord(char):04X}"
            )

            print(
                "Font:",
                char_info.get("fontname")
            )

            print(
                "Font size:",
                char_info.get("size")
            )

            print(
                "x0:",
                char_info.get("x0")
            )

            print(
                "x1:",
                char_info.get("x1")
            )

            print(
                "top:",
                char_info.get("top")
            )

            print(
                "bottom:",
                char_info.get("bottom")
            )

            # Print the complete dictionary once so we can see
            # all metadata that pdfplumber provides for the glyph.
            print("\nFull PDF character information:")
            print(char_info)

            # -------------------------------------------------
            # Inspect neighboring characters
            # -------------------------------------------------

            print("\nNeighboring characters:")

            # Show a few characters before and after the
            # Private Use Area character in PDF extraction order.
            start = max(0, index - 4)
            end = min(len(page.chars), index + 5)

            for neighbor in page.chars[start:end]:

                print(
                    repr(neighbor.get("text", "")),
                    "font=",
                    neighbor.get("fontname"),
                    "x0=",
                    neighbor.get("x0"),
                    "x1=",
                    neighbor.get("x1")
                )

# ---------------------------------------------------------
# 10. Inspect the actual PDF font objects
# ---------------------------------------------------------

print("\n--- PDF FONT MAPPING ---")

with pdfplumber.open(PDF_PATH) as pdf:

    page = pdf.pages[10]

    pdf_page = page.page_obj
    resources = pdf_page.resources
    fonts = resources.get("Font", {})

    print("\nFonts found:")

    for font_name, font_ref in fonts.items():

        print("\n" + "=" * 60)
        print("Font:", font_name)
        print("Reference:", font_ref)
        print("=" * 60)

        try:
            # Resolve the PDF object reference into
            # the actual font dictionary.
            font_object = pdf.doc.getobj(font_ref.objid)

            print("\nResolved font object:")
            print(font_object)

            print("\nFont object type:")
            print(type(font_object))

            # PDFMiner font dictionaries are usually
            # PDFStream/PDFDict-like objects.
            try:
                print("\nSubtype:")
                print(font_object.get("Subtype"))
            except Exception as e:
                print("Could not read Subtype:", e)

            try:
                print("\nBaseFont:")
                print(font_object.get("BaseFont"))
            except Exception as e:
                print("Could not read BaseFont:", e)

            try:
                print("\nEncoding:")
                print(font_object.get("Encoding"))
            except Exception as e:
                print("Could not read Encoding:", e)

            try:
                print("\nToUnicode:")
                print(font_object.get("ToUnicode"))
            except Exception as e:
                print("Could not read ToUnicode:", e)

        except Exception as e:

            print("\nCould not resolve font:")
            print(repr(e))

# ---------------------------------------------------------
# 11. Inspect the ToUnicode map for SimplifiedArabic
# ---------------------------------------------------------

print("\n--- F4 ToUnicode MAP ---")

with pdfplumber.open(PDF_PATH) as pdf:

    page = pdf.pages[10]

    pdf_page = page.page_obj
    fonts = pdf_page.resources.get("Font", {})

    # F4 is the SimplifiedArabic font.
    f4_ref = fonts["F4"]

    # Resolve F4.
    f4 = pdf.doc.getobj(f4_ref.objid)

    print("\nF4 font:")
    print(f4)

    # Get the ToUnicode object.
    tounicode_ref = f4.get("ToUnicode")

    print("\nToUnicode reference:")
    print(tounicode_ref)

    # Resolve the ToUnicode stream.
    tounicode = pdf.doc.getobj(tounicode_ref.objid)

    print("\nToUnicode object:")
    print(tounicode)

    print("\nToUnicode object type:")
    print(type(tounicode))

    # The ToUnicode object is normally a PDF stream.
    # Decode the stream so we can inspect the CMap text.
    try:
        cmap_data = tounicode.get_data()

        print("\n--- RAW ToUnicode CMap ---")
        print(cmap_data.decode("utf-8", errors="replace"))

    except Exception as e:

        print("\nCould not decode ToUnicode stream:")
        print(repr(e))

# ---------------------------------------------------------
# 12. Inspect the SimplifiedArabic descendant font
# ---------------------------------------------------------

print("\n--- SIMPLIFIEDARABIC DESCENDANT FONT ---")

with pdfplumber.open(PDF_PATH) as pdf:

    page = pdf.pages[10]

    fonts = page.page_obj.resources.get("Font", {})

    # F4 = FNTSBS+SimplifiedArabic
    f4_ref = fonts["F4"]
    f4 = pdf.doc.getobj(f4_ref.objid)

    print("\nF4:")
    print(f4)

    # Type0 fonts store the actual font information
    # inside DescendantFonts.
    descendant_ref = f4["DescendantFonts"][0]

    print("\nDescendant font reference:")
    print(descendant_ref)

    descendant = pdf.doc.getobj(descendant_ref.objid)

    print("\nDescendant font:")
    print(descendant)

    print("\nDescendant font type:")
    print(type(descendant))

    print("\n--- IMPORTANT FONT FIELDS ---")

    for key in [
        "Subtype",
        "BaseFont",
        "CIDSystemInfo",
        "DW",
        "W",
        "FontDescriptor"
    ]:

        try:
            print(f"\n{key}:")
            print(descendant.get(key))
        except Exception as e:
            print(f"Could not read {key}: {e}")

# ---------------------------------------------------------
# 13. Inspect the embedded SimplifiedArabic font
# ---------------------------------------------------------

print("\n--- EMBEDDED SIMPLIFIEDARABIC FONT ---")

with pdfplumber.open(PDF_PATH) as pdf:

    page = pdf.pages[10]

    fonts = page.page_obj.resources.get("Font", {})

    # F4 = SimplifiedArabic
    f4 = pdf.doc.getobj(fonts["F4"].objid)

    # Get descendant CIDFont
    descendant_ref = f4["DescendantFonts"][0]
    descendant = pdf.doc.getobj(descendant_ref.objid)

    # Get FontDescriptor
    descriptor_ref = descendant["FontDescriptor"]
    descriptor = pdf.doc.getobj(descriptor_ref.objid)

    print("\nFontDescriptor:")
    print(descriptor)

    print("\nFontDescriptor type:")
    print(type(descriptor))

    print("\n--- FONT FILES ---")

    for key in [
        "FontFile",
        "FontFile2",
        "FontFile3"
    ]:

        try:
            value = descriptor.get(key)

            print(f"\n{key}:")
            print(value)

        except Exception as e:

            print(f"\nCould not read {key}:")
            print(e)

# ---------------------------------------------------------
# 14. Extract the embedded SimplifiedArabic TrueType font
# ---------------------------------------------------------

print("\n--- EXTRACT EMBEDDED FONT ---")

with pdfplumber.open(PDF_PATH) as pdf:

    page = pdf.pages[10]

    fonts = page.page_obj.resources.get("Font", {})

    # F4 = FNTSBS+SimplifiedArabic
    f4 = pdf.doc.getobj(fonts["F4"].objid)

    # Get descendant CID font
    descendant_ref = f4["DescendantFonts"][0]
    descendant = pdf.doc.getobj(descendant_ref.objid)

    # Get FontDescriptor
    descriptor_ref = descendant["FontDescriptor"]
    descriptor = pdf.doc.getobj(descriptor_ref.objid)

    # Get embedded TrueType font
    font_file_ref = descriptor["FontFile2"]

    print("FontFile2 reference:")
    print(font_file_ref)

    font_stream = pdf.doc.getobj(font_file_ref.objid)

    print("\nFont stream:")
    print(font_stream)

    # Decode the embedded font bytes.
    font_data = font_stream.get_data()

    output_path = "data/processed/SimplifiedArabic_embedded.ttf"

    with open(output_path, "wb") as f:
        f.write(font_data)

    print("\nFont extracted successfully.")
    print("Saved to:")
    print(output_path)

    print("\nFont size:")
    print(len(font_data), "bytes")