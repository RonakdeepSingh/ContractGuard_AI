import sys

import pipeline4
from contract_reader import read_contract, split_into_clauses

NO_RISK = pipeline4.NO_RISK

path = sys.argv[1] if len(sys.argv) > 1 else "sample_contract.txt"

text = read_contract(path)
clauses = split_into_clauses(text)
print(f"Read {path}: found {len(clauses)} clauses")
if not clauses:
    sys.exit("No readable text found. If this is a scanned PDF, it needs OCR first.")

# the pipeline expects clauses separated by blank lines
out = pipeline4.app.invoke({"contract_text": "\n\n".join(clauses)})

print()
print("=== CONTRACT RISK REPORT WITH SUGGESTED CHANGES ===")
print()
shown = 0
for i, (r, a, red) in enumerate(zip(out["results"], out["analysis"], out["redlines"]), start=1):
    if r["label"] == NO_RISK:
        continue
    shown += 1
    print(f"Clause {i}: {r['clause'][:100]}...")
    if a is None:
        print("   (analysis unavailable this run, please try again)")
        print()
        continue
    print(f"   Type: {a.clause_type}   |   Risk: {a.risk_level}")
    print(f"   In plain English: {a.explanation}")
    if red is not None:
        print(f"   Suggested rewrite: {red.suggested_clause}")
        print(f"   What changed: {red.what_changed}")
        print(f"   If they refuse, ask for: {red.fallback}")
    elif a.risk_level != "Low":
        print("   (no rewrite available this run, please try again)")
    print()
if shown == 0:
    print("No risky clauses were found.")
print("Note: these are AI-generated suggestions, not legal advice.")
print(f"Gemini calls made in this run: {pipeline4.calls_made}")