import re
import unicodedata
from collections import Counter

TEXT_PATH = "data/processed/labor_law_clean_final_v2.txt"

with open(TEXT_PATH, "r", encoding="utf-8") as file:
    text = file.read()

print("--- FINAL TEXT QUALITY CHECK ---")

print("Total characters:", len(text))
print("Total lines:", len(text.splitlines()))

# --------------------------------------------------
# 1. Check remaining Private Use Area characters
# --------------------------------------------------

pua_chars = [
    char for char in text
    if 0xE000 <= ord(char) <= 0xF8FF
]

print("\nPUA characters found:", len(pua_chars))

pua_counts = Counter(pua_chars)

for char, count in sorted(pua_counts.items(), key=lambda x: ord(x[0])):
    print(
        f"U+{ord(char):04X} "
        f"occurrences={count}"
    )

# --------------------------------------------------
# 2. Check Tatweel
# --------------------------------------------------

tatweel_count = text.count("ـ")

print("\nTatweel occurrences:", tatweel_count)

# --------------------------------------------------
# 3. Check known broken words
# --------------------------------------------------

KNOWN_BROKEN = [
    "فـ ى",
    "ف ى",
    "الت ى",
    "والت ى",
    "يومً ا",
    "قرارً ا",
    "ورقيً ا",
    "أ ى",
    "طرف ى",
    "الق ومى",
    "األساس ى",
    "األ جر",
    "ثمان ى",
    "النقاب ى",
    "الع مل",
    "الذ ى",
    "يكو ن",
    "مفص الً",
    "ل رئيس",
    "ا لمحاكم",
    "الط بية",
    "ا لقانون",
    "مسئو الً",
    "م رض",
    "سالمت ها",
    "التو فيق",
    "أحكا م",
    "المح كوم",
]

print("\n--- KNOWN BROKEN WORD CHECK ---")

remaining_broken = []

for broken in KNOWN_BROKEN:
    count = text.count(broken)

    if count:
        remaining_broken.append((broken, count))
        print(f"FOUND: {broken} -> {count}")

if not remaining_broken:
    print("All previously identified broken words: 0")

# --------------------------------------------------
# 4. Page markers
# --------------------------------------------------

page_numbers = re.findall(
    r"===== PAGE (\d+) =====",
    text
)

print("\n--- PAGE CHECK ---")
print("Page markers:", len(page_numbers))

if page_numbers:
    print("First page:", page_numbers[0])
    print("Last page:", page_numbers[-1])

expected_pages = [str(i) for i in range(1, 113)]

if page_numbers == expected_pages:
    print("Pages 1-112 are present and ordered correctly.")
else:
    print("WARNING: page sequence differs from expected 1-112.")

# --------------------------------------------------
# 5. Article markers
# --------------------------------------------------

article_patterns = {
    "مادة (...)": r"مادة\s*\([٠-٩0-9]+\)",
    "المادة (...)": r"المادة\s*\([٠-٩0-9]+\)",
}

print("\n--- ARTICLE MARKERS ---")

for name, pattern in article_patterns.items():
    matches = re.findall(pattern, text)
    print(f"{name}: {len(matches)}")

# --------------------------------------------------
# 6. Show actual article numbers
# --------------------------------------------------

article_numbers = re.findall(
    r"مادة\s*\(([٠-٩0-9]+)\)",
    text
)

print("\nDetected article-number references:", len(article_numbers))

print("First 20:")
print(article_numbers[:20])

print("\nLast 20:")
print(article_numbers[-20:])

# --------------------------------------------------
# 7. Inspect beginning, middle and end
# --------------------------------------------------

print("\n--- BEGINNING SAMPLE ---")
print(text[:1500])

print("\n--- MIDDLE SAMPLE ---")

middle = len(text) // 2
print(text[middle:middle + 1500])

print("\n--- END SAMPLE ---")
print(text[-1500:])

print("\n--- FINAL QC COMPLETE ---")