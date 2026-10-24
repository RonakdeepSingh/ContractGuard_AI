import json
import re
from collections import Counter
from pathlib import Path

FILE = Path("data") / "CUADv1.json"

with open(FILE, encoding="utf-8") as f:
    cuad = json.load(f)

contracts = cuad["data"]
print("Number of contracts:", len(contracts))

category_total = Counter()
category_found = Counter()

for contract in contracts:
    for para in contract["paragraphs"]:
        for qa in para["qas"]:
            m = re.search(r'related to "(.*?)"', qa["question"])
            category = m.group(1) if m else "UNKNOWN"
            category_total[category] += 1
            if len(qa["answers"]) > 0:
                category_found[category] += 1

print("Number of clause categories:", len(category_total))
print()
print("How many contracts contain each clause type:")
for cat, n in category_found.most_common():
    print(f"{cat:50s} {n}")

print()
print("Example: 'Cap On Liability' clause")
for contract in contracts:
    done = False
    for para in contract["paragraphs"]:
        for qa in para["qas"]:
            if 'related to "Cap On Liability"' in qa["question"] and qa["answers"]:
                print("Contract:", contract["title"])
                print("Clause text:", qa["answers"][0]["text"][:400])
                done = True
                break
        if done:
            break
    if done:
        break