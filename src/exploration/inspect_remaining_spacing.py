import re
from collections import Counter

INPUT_PATH = "data/processed/labor_law_clean_final_v2.txt"

# Specific suspicious examples already observed during QC.
TARGETS = [
    "التأم ينات",
    "ينج زها",
    "قانونالمرافق",
    "قامتبأداء",
]


def find_occurrences(text, target, window=100):
    occurrences = []

    start = 0

    while True:
        index = text.find(target, start)

        if index == -1:
            break

        context_start = max(0, index - window)
        context_end = min(
            len(text),
            index + len(target) + window
        )

        context = text[context_start:context_end]

        occurrences.append({
            "position": index,
            "context": context,
        })

        start = index + len(target)

    return occurrences


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    print("=" * 70)
    print("TARGETED INSPECTION OF REMAINING SPACING ARTIFACTS")
    print("=" * 70)

    for target in TARGETS:
        occurrences = find_occurrences(text, target)

        print()
        print("-" * 70)
        print(f"TARGET: {target}")
        print(f"OCCURRENCES: {len(occurrences)}")
        print("-" * 70)

        if not occurrences:
            print("Not found.")
            continue

        for i, occurrence in enumerate(occurrences, start=1):
            print()
            print(f"Occurrence {i}")
            print(f"Position: {occurrence['position']}")
            print("Context:")
            print(occurrence["context"])

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()