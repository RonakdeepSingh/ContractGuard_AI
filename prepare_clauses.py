import json
import re
from collections import Counter
from pathlib import Path

RISK_CATEGORIES = {
    "Uncapped Liability": "You could be liable for unlimited damages.",
    "Cap On Liability": "Limits how much can be claimed. Check if the cap is fair to you.",
    "Liquidated Damages": "Fixed penalty amounts you may have to pay.",
    "Ip Ownership Assignment": "Your work or ideas may become the other party's property.",
    "Joint Ip Ownership": "Shared ownership of IP can limit how you use it.",
    "Non-Compete": "Stops you from working with competitors.",
    "Exclusivity": "Locks you to a single partner or supplier.",
    "Termination For Convenience": "One side can walk away without a reason.",
    "Renewal Term": "The contract may renew for long periods.",
    "Notice Period To Terminate Renewal": "Miss the notice window and you are locked in again.",
    "Anti-Assignment": "Limits your ability to transfer the contract.",
    "Change Of Control": "Triggers if your company is sold or merges.",
    "Irrevocable Or Perpetual License": "Rights granted can never be taken back.",
    "Most Favored Nation": "You must give the other side your best prices.",
    "Minimum Commitment": "You must buy or pay at least a fixed amount.",
}

SRC = Path("data") / "CUADv1.json"
OUT = Path("clauses.json")

with open(SRC, encoding="utf-8") as f:
    cuad = json.load(f)

clauses = []
seen = set()

for contract in cuad["data"]:
    title = contract["title"]
    for para in contract["paragraphs"]:
        for qa in para["qas"]:
            m = re.search(r'related to "(.*?)"', qa["question"])
            if not m:
                continue
            category = m.group(1)
            if category not in RISK_CATEGORIES:
                continue
            for ans in qa["answers"]:
                text = " ".join(ans["text"].split())
                if len(text) < 40 or text in seen:
                    continue
                seen.add(text)
                clauses.append({
                    "id": len(clauses),
                    "contract": title,
                    "category": category,
                    "why_risky": RISK_CATEGORIES[category],
                    "text": text,
                })

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(clauses, f, indent=2)

print("Saved", len(clauses), "clauses to", OUT)
print()
counts = Counter(c["category"] for c in clauses)
for cat, n in counts.most_common():
    print(f"{cat:40s} {n}")

print()
print("Sample clause:")
s = clauses[0]
print("Category:", s["category"])
print("Why risky:", s["why_risky"])
print("Text:", s["text"][:300])