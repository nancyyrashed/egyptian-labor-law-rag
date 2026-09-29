import pdfplumber
import unicodedata
from bidi.algorithm import get_display

PDF_PATH = "data/raw/labor_law_14_2025.pdf"

# Open the PDF file. The "with" statement automatically closes
# the PDF after we finish working with it.
with pdfplumber.open(PDF_PATH) as pdf:

    # PDF pages are stored in a list-like structure.
    # [0] means the first page because Python uses zero-based indexing.
    page = pdf.pages[0]

    # Extract the text from the selected page.
    raw_text = page.extract_text()


# ---------------------------------------------------------
# 1. Display the raw extracted text
# ---------------------------------------------------------

print("--- RAW EXTRACTED TEXT ---")
print(raw_text[:500])

# [:500] displays only the first 500 characters so we can
# inspect the extraction without printing the entire page.


# ---------------------------------------------------------
# 2. Apply Unicode NFKC normalization
# ---------------------------------------------------------

# PDF extraction may contain compatibility characters such as
# Arabic Presentation Forms instead of standard Arabic letters.
#
# NFKC converts these compatibility characters into their
# standard Unicode equivalents.
normalized_text = unicodedata.normalize("NFKC", raw_text)


# ---------------------------------------------------------
# 3. Display the normalized text
# ---------------------------------------------------------

print("\n--- NFKC NORMALIZED TEXT ---")
print(normalized_text[:500])


# ---------------------------------------------------------
# 4. Compare the first characters after NFKC
# ---------------------------------------------------------

print("\n--- FIRST 50 CHARACTERS AFTER NFKC ---")

# Go through the first 50 characters of the normalized text.
for char in normalized_text[:50]:

    # Ignore spaces and other whitespace characters.
    if char.isspace():
        continue

    # ord(char) converts the character into its Unicode number.
    #
    # Example:
    # "ا" -> U+0627
    #
    # unicodedata.name() gives us the official Unicode name
    # of the character.
    print(
        repr(char),
        f"U+{ord(char):04X}",
        unicodedata.name(char, "UNKNOWN")
    )


# ---------------------------------------------------------
# 5. Check the actual character order of each word
# ---------------------------------------------------------

print("\n--- WORD CHARACTER ORDER TEST ---")

# split() separates the text into individual words using
# whitespace such as spaces and newlines.
words = normalized_text.split()

# enumerate() gives us both the position of the word (i)
# and the word itself.
#
# We only inspect the first 10 words to keep the output manageable.
for i, word in enumerate(words[:10]):

    print(f"\nWord {i}: {repr(word)}")

    # Check every character in the current word individually.
    # This lets us see the order Python actually stores the
    # characters, rather than relying only on how the text looks.
    for char in word:

        if char.isspace():
            continue

        print(
            f"{repr(char)} "
            f"U+{ord(char):04X} "
            f"{unicodedata.name(char, 'UNKNOWN')}"
        )


# ---------------------------------------------------------
# 6. Test reversing Arabic characters only
# ---------------------------------------------------------

def reverse_arabic_characters(text):
    """
    Test whether reversing Arabic characters inside each word
    can fix the extracted text order.

    Numbers and punctuation are intended to remain in place.
    """

    # This will store the processed words.
    corrected_words = []

    # Process each word separately.
    for word in text.split():

        # Store Arabic characters separately so that we can
        # test reversing them.
        arabic_chars = []

        # Store the positions of non-Arabic characters.
        # These include numbers and punctuation.
        non_arabic_positions = []

        for index, char in enumerate(word):

            # These Unicode ranges contain Arabic characters
            # and Arabic Presentation Forms.
            #
            # U+0600–U+06FF  -> Arabic block
            # U+FB50–U+FDFF  -> Arabic Presentation Forms-A
            # U+FE70–U+FEFF  -> Arabic Presentation Forms-B
            if (
                "\u0600" <= char <= "\u06FF"
                or "\uFB50" <= char <= "\uFDFF"
                or "\uFE70" <= char <= "\uFEFF"
            ):
                arabic_chars.append(char)

            else:
                # Remember the original position of characters
                # that are not Arabic, such as numbers and punctuation.
                non_arabic_positions.append((index, char))

        # Reverse only the Arabic characters.
        arabic_chars.reverse()

        # Convert the original word into a list so that
        # individual character positions can be replaced.
        result = list(word)

        # Keep track of which reversed Arabic character
        # should be inserted next.
        arabic_index = 0

        # Go through the original word again.
        for index, char in enumerate(word):

            # Identify the positions that originally contained
            # Arabic characters.
            if (
                "\u0600" <= char <= "\u06FF"
                or "\uFB50" <= char <= "\uFDFF"
                or "\uFE70" <= char <= "\uFEFF"
            ):

                # Replace the original Arabic character with
                # the next character from the reversed list.
                result[index] = arabic_chars[arabic_index]

                # Move to the next reversed Arabic character.
                arabic_index += 1

        # Convert the list back into a string and store the word.
        corrected_words.append("".join(result))

    # Join all processed words back together with spaces.
    return " ".join(corrected_words)


# Apply the manual-reversal approach to the normalized text.
corrected_text = reverse_arabic_characters(normalized_text)


# ---------------------------------------------------------
# 7. Display the corrected text
# ---------------------------------------------------------

print("\n--- TEST: CORRECTED ARABIC ORDER ---")
print(corrected_text[:500])

# This is only a test.
# We later discovered that manually reversing Arabic characters
# also causes problems with numbers and mixed Arabic/non-Arabic text.


# ---------------------------------------------------------
# 8. Test BiDi handling
# ---------------------------------------------------------

# The Unicode Bidirectional (BiDi) algorithm determines how
# right-to-left Arabic text should be ordered when it appears
# together with left-to-right content such as numbers and punctuation.
#
# get_display() produces the display-order representation.
bidi_text = get_display(normalized_text)

print("\n--- TEST: BIDI OUTPUT ---")
print(bidi_text[:500])


# ---------------------------------------------------------
# 9. Inspect the BIDI result at the character level
# ---------------------------------------------------------

print("\n--- FIRST WORDS AFTER BIDI ---")

# Split the BiDi result into words so we can inspect them individually.
bidi_words = bidi_text.split()

# Inspect the first 10 words character by character.
# This verifies whether the BiDi operation actually changed
# the character order in the Python string.
for i, word in enumerate(bidi_words[:10]):

    print(f"\nWord {i}: {repr(word)}")

    for char in word:

        if char.isspace():
            continue

        print(
            f"{repr(char)} "
            f"U+{ord(char):04X} "
            f"{unicodedata.name(char, 'UNKNOWN')}"
        )