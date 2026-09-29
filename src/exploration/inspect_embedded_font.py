from fontTools.ttLib import TTFont


# Path to the TrueType font that we extracted from the PDF.
# We extracted this font because the PDF was using Private Use
# Area characters, and we needed to inspect the actual glyphs
# stored inside the font to determine what they represented.
FONT_PATH = "data/processed/SimplifiedArabic_embedded.ttf"


print("--- INSPECTING EMBEDDED SIMPLIFIEDARABIC FONT ---")


# Load the embedded TrueType font using fontTools.
# This gives us access to the font's glyphs, Unicode mappings,
# metrics, and other internal tables.
font = TTFont(FONT_PATH)


# ---------------------------------------------------------
# 1. Inspect the font tables
# ---------------------------------------------------------

print("\nFont tables:")
print(font.keys())


# Get the font's glyph order.
# A TrueType font stores its glyphs using numeric glyph IDs.
# We need this because the PDF told us that its CIDs map directly
# to glyph IDs through CIDToGIDMap = Identity.
glyph_order = font.getGlyphOrder()


print("\nNumber of glyphs:")
print(len(glyph_order))


# ---------------------------------------------------------
# 2. Inspect the specific glyphs from the PDF
# ---------------------------------------------------------

print("\n--- TARGET GLYPHS ---")


# From the PDF's ToUnicode mapping, we found:
#
#     CID 0105 → U+E821
#     CID 0107 → U+E823
#     CID 0109 → U+E825
#
# The PDF also uses CIDToGIDMap = Identity, meaning the CID
# corresponds directly to the same glyph ID.
#
# Therefore, we inspect glyph IDs 0x0105, 0x0107 and 0x0109.
target_glyph_ids = [0x0105, 0x0107, 0x0109]


for glyph_id in target_glyph_ids:

    print("\n" + "=" * 60)

    print(f"Glyph ID: {glyph_id} (0x{glyph_id:04X})")


    # Make sure the requested glyph ID actually exists
    # in the embedded font before accessing it.
    if glyph_id < len(glyph_order):

        glyph_name = glyph_order[glyph_id]

        print("Glyph name:")
        print(glyph_name)

    else:

        print("Glyph ID is outside glyph table.")


# ---------------------------------------------------------
# 3. Inspect the font's Unicode mappings
# ---------------------------------------------------------

print("\n--- UNICODE CMAP ENTRIES ---")


# The cmap table tells us which Unicode code points are associated
# with which glyphs in the font.
#
# We specifically want to check the Private Use Area characters
# that appeared during PDF extraction:
#
#     U+E821
#     U+E823
#     U+E825
#
# These characters do not have standard Unicode meanings, so
# inspecting the font's cmap helps us identify their actual
# font-specific glyph names.
for table in font["cmap"].tables:

    print("\nCmap platform:")
    print(table.platformID)

    print("Cmap encoding:")
    print(table.platEncID)

    print("Number of mappings:")
    print(len(table.cmap))


    # Look specifically for the Private Use Area characters
    # that we discovered during the PDF extraction investigation.
    interesting = {
        codepoint: glyph_name
        for codepoint, glyph_name in table.cmap.items()
        if codepoint in [
            0xE821,
            0xE823,
            0xE825
        ]
    }


    # If any of the Private Use characters are present in this
    # cmap table, print the mapping so we can determine which
    # font glyph each one represents.
    if interesting:

        print("\nPrivate Use mappings found:")

        for codepoint, glyph_name in interesting.items():

            print(
                f"U+{codepoint:04X}"
                f" -> {glyph_name}"
            )


# ---------------------------------------------------------
# 4. Inspect the target glyph metrics
# ---------------------------------------------------------

print("\n--- TARGET GLYPH METRICS ---")


# The hmtx table contains horizontal metrics for each glyph,
# including its advance width and left side bearing.
#
# We inspect these metrics because the earlier PDF-level
# coordinate investigation suggested that these glyphs were
# positioned as overlaid marks rather than normal letters.
if "hmtx" in font:

    hmtx = font["hmtx"]


    for glyph_id in target_glyph_ids:

        # Make sure the glyph exists before looking up its name.
        if glyph_id < len(glyph_order):

            glyph_name = glyph_order[glyph_id]


            # Check that the glyph has an entry in the font's
            # horizontal metrics table.
            if glyph_name in hmtx.metrics:

                advance_width, left_side_bearing = hmtx.metrics[glyph_name]


                print(
                    f"\nGlyph {glyph_id} ({glyph_name})"
                )

                print(
                    "Advance width:",
                    advance_width
                )

                print(
                    "Left side bearing:",
                    left_side_bearing
                )