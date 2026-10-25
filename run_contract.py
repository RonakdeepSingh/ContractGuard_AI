import sys

import pipeline5
from contract_reader import read_contract, split_into_clauses

path = sys.argv[1] if len(sys.argv) > 1 else "sample_contract.txt"

clauses = split_into_clauses(read_contract(path))
print(f"Read {path}: found {len(clauses)} clauses")
if not clauses:
    sys.exit("No readable text found. If this is a scanned PDF, it needs OCR first.")

# the pipeline expects clauses separated by blank lines
print()
print(pipeline5.analyze("\n\n".join(clauses)))