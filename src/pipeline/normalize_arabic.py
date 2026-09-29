import unicodedata


def normalize_private_arabic_marks(text):
    """
    Convert verified PDF-specific Arabic diacritic characters
    from the Private Use Area into their standard Unicode forms.

    These mappings were verified by inspecting the PDF's font,
    ToUnicode map, embedded TrueType font, and glyph names.
    """

    replacements = {
        "\uE821": "\u064F",  # U+E821 -> Arabic Damma ُ
        "\uE823": "\u064B",  # U+E823 -> Arabic Fathatan ً
        "\uE825": "\u0651",  # U+E825 -> Arabic Shadda ّ
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def normalize_arabic_text(text):
    """
    Apply the Arabic text normalization steps established
    during the PDF investigation.
    """

    # NFKC converts compatibility characters from the PDF
    # into their standard Unicode equivalents.
    text = unicodedata.normalize("NFKC", text)

    # Convert the PDF-specific Private Use Arabic marks
    # into standard Unicode Arabic diacritics.
    text = normalize_private_arabic_marks(text)

    return text


# ---------------------------------------------------------
# Test the verified Private Use character mappings
# ---------------------------------------------------------

if __name__ == "__main__":

    test_text = (
        "يعتبر هذا قرارا مقدما "
        "ويحدد الحكم."
    )

    print("--- BEFORE NORMALIZATION ---")
    print(test_text)

    normalized = normalize_arabic_text(test_text)

    print("\n--- AFTER NORMALIZATION ---")
    print(normalized)