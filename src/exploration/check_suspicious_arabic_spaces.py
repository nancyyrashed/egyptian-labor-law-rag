import re
from collections import Counter

INPUT_PATH = "data/processed/labor_law_clean_final_v2.txt"

ARABIC = r"\u0600-\u06FF"

# Arabic character + space + Arabic character
PATTERN = re.compile(
    rf"([{ARABIC}])[ \t]+([{ARABIC}])"
)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

matches = PATTERN.findall(text)

counter = Counter(
    f"{left} {right}"
    for left, right in matches
)

print("=" * 70)
print("SUSPICIOUS ARABIC INTERNAL SPACING DIAGNOSTIC")
print("=" * 70)

print()
print("Total Arabic-space-Arabic matches:", len(matches))
print("Unique patterns:", len(counter))

print()
print("-" * 70)
print("MOST FREQUENT PATTERNS")
print("-" * 70)

for pattern, count in counter.most_common(100):
    print(f"{count:>5} | {pattern}")

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)