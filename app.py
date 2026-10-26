import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

import pipeline4 as p4
import pipeline5
from contract_reader import read_contract, split_into_clauses

app = FastAPI(title="ContractGuard AI")
PAGE = Path(__file__).parent / "index.html"
ALLOWED = {".pdf", ".docx", ".txt"}


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE.read_text(encoding="utf-8")


def build_items(out):
    items = []
    rows = zip(out["results"], out["analysis"], out["redlines"], out["reviews"])
    for i, (r, a, red, rev) in enumerate(rows, start=1):
        if r["label"] == p4.NO_RISK:
            continue
        low_conf = a is None and r["top_score"] < p4.AI_THRESHOLD
        item = {"number": i, "clause": r["clause"], "label": r["label"],
                "analysis_ok": a is not None, "low_confidence": low_conf}
        if a is not None:
            revised = rev is not None and rev.verdict == "Revised"
            item.update({
                "type": a.clause_type,
                "risk": a.risk_level,
                "explanation": rev.final_explanation if revised else a.explanation,
                "what_to_check": a.what_to_check,
            })
            if red is not None:
                rewrite = rev.final_rewrite if revised else red.suggested_clause
                item.update({
                    "rewrite": rewrite,
                    "what_changed": rev.final_what_changed if revised else red.what_changed,
                    "fallback": rev.final_fallback if revised else red.fallback,
                    "critic": "Not available" if rev is None else rev.verdict,
                    "issues": rev.issues if revised else [],
                    "new_figures": pipeline5.new_numbers(r["clause"], rewrite),
                })
        items.append(item)
    return items


@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=400, detail="Please upload a .pdf, .docx or .txt file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        path = tmp.name
    try:
        clauses = split_into_clauses(read_contract(path))
    except Exception:
        raise HTTPException(status_code=400, detail="That file could not be read. Is it damaged or password-protected?")
    finally:
        os.remove(path)

    if not clauses:
        raise HTTPException(status_code=400, detail="No readable text found. Scanned (image-only) PDFs are not supported yet.")

    before = p4.calls_made
    try:
        out = pipeline5.app.invoke({"contract_text": "\n\n".join(clauses)})
    except Exception as e:
        raise HTTPException(status_code=500, detail="The analysis failed: " + str(e)[:200])

    items = build_items(out)
    return {
        "filename": file.filename,
        "clauses_read": len(clauses),
        "items": items,
        "gemini_calls": p4.calls_made - before,
        "some_unavailable": any(not it["analysis_ok"] and not it["low_confidence"] for it in items),
    }
