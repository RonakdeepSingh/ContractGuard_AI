import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

MIN_LEN = 60       # ignore fragments shorter than this
MAX_LEN = 1500     # split clauses longer than this

# A line that starts a new numbered clause, e.g. "5. TERMINATION", "5.2 The Customer shall", "Section 7"
NEW_CLAUSE = re.compile(
    r'^\s*(?:\d+(?:\.\d+)*[.)]?\s+["“(A-Z]|(?:ARTICLE|Article|SECTION|Section|CLAUSE|Clause)\s+[0-9IVXivx]+)'
)
SUB_ITEM = re.compile(r"^(?:\(?[A-Za-z0-9]{1,4}\)|[a-z])")
PAGE_NUMBER = re.compile(r"^\s*(?:Page\s+)?\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)


def read_txt(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def read_docx(path):
    """A .docx file is a zip; the text lives in word/document.xml."""
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    paragraphs = []
    for p in root.iter(ns + "p"):
        paragraphs.append("".join(t.text or "" for t in p.iter(ns + "t")))
    return "\n\n".join(paragraphs)


def read_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def read_contract(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".txt":
        return read_txt(path)
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".pdf":
        return read_pdf(path)
    raise ValueError("Unsupported file type: " + suffix + " (use .txt, .docx or .pdf)")


def split_long(text):
    """Break a very long clause into pieces at sentence ends."""
    if len(text) <= MAX_LEN:
        return [text]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.;])\s+", text):
        if current and len(current) + len(sentence) > MAX_LEN:
            pieces.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current:
        pieces.append(current)
    return pieces


def split_into_clauses(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1. group lines into pieces: a blank line or a new numbered heading starts a new piece
    pieces = []
    for block in re.split(r"\n\s*\n", text):
        current = []
        for line in block.split("\n"):
            if PAGE_NUMBER.match(line) or not line.strip():
                continue
            if NEW_CLAUSE.match(line) and current:
                pieces.append(" ".join(current))
                current = []
            current.append(line.strip())
        if current:
            pieces.append(" ".join(current))

    # 2. glue short pieces onto a neighbouring clause
    clauses, carry = [], ""
    for piece in pieces:
        piece = " ".join(piece.split())
        if len(piece) < MIN_LEN:
            if clauses and not carry and SUB_ITEM.match(piece):
                clauses[-1] = clauses[-1] + " " + piece   # "(a) ..." belongs to the clause before it
            else:
                carry = (carry + " " + piece).strip()      # a heading belongs to the clause after it
            continue
        clauses.append((carry + " " + piece).strip())
        carry = ""

    # 3. split anything too long
    final = []
    for c in clauses:
        final.extend(split_long(c))
    return final


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_contract.txt"
    clauses = split_into_clauses(read_contract(path))
    print(f"Read {path}: found {len(clauses)} clauses")
    for i, c in enumerate(clauses[:8], start=1):
        print(f"{i}. {c[:110]}")