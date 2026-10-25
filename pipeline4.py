import hashlib
import json
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from pipeline import ContractState, classifier_agent, retriever_agent, split_clauses

load_dotenv()

GEMINI_MODEL = "gemini-3.8-flash"  # if Google renames it, change it here only
NO_RISK = "No risky pattern found"
CHUNK = 8  # clauses reviewed per Gemini call
CACHE_FILE = Path("cache.json")
calls_made = 0  # counts real Gemini calls in this run


# ---------- What Gemini must return ----------
class ClauseAnalysis(BaseModel):
    clause_number: int = Field(description="The clause number exactly as given in the prompt")
    clause_type: str = Field(description="What kind of clause this really is, for example 'Uncapped Liability'")
    risk_level: Literal["Low", "Medium", "High"]
    explanation: str = Field(description="2 to 3 sentences in plain English for a non-lawyer")
    what_to_check: str = Field(description="One practical thing the signer should ask for or check")


class AnalysisBatch(BaseModel):
    items: list[ClauseAnalysis]


class ClauseRedline(BaseModel):
    clause_number: int = Field(description="The clause number exactly as given in the prompt")
    suggested_clause: str = Field(description="The full rewritten clause, in contract language, more balanced and fair")
    what_changed: str = Field(description="2 to 3 plain-English sentences explaining what was changed and why")
    fallback: str = Field(description="One sentence: a smaller compromise to ask for if the other side refuses the rewrite")


class RedlineBatch(BaseModel):
    items: list[ClauseRedline]


llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0, max_retries=0)
analyzer_llm = llm.with_structured_output(AnalysisBatch)
negotiator_llm = llm.with_structured_output(RedlineBatch)


class BatchState(ContractState):
    analysis: list
    redlines: list


# ---------- Saved results, so repeat runs cost nothing ----------
def load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")


def cached_call(structured_llm, schema, prompt):
    global calls_made
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache = load_cache()
    if key in cache:
        print("(using a saved result, no Gemini call needed)")
        return schema(**cache[key])

    for attempt in range(3):
        try:
            calls_made += 1
            result = structured_llm.invoke(prompt)
            cache[key] = result.model_dump()
            save_cache(cache)
            time.sleep(5)  # stay under the per-minute limit
            return result
        except Exception as e:
            msg = str(e)
            print("Gemini hiccup:", msg[:200])
            if "PerDay" in msg or "per day" in msg.lower():
                print("Your daily Gemini quota looks used up. Try again tomorrow.")
                return None
            wait = 30 * (attempt + 1)
            print("Waiting", wait, "seconds and trying again...")
            time.sleep(wait)
    return None


# ---------- Agent: ANALYZER (reviews flagged clauses together) ----------
def build_analysis_prompt(state, idxs):
    blocks = []
    for pos, i in enumerate(idxs, start=1):
        r = state["results"][i]
        precedents = "\n".join(
            f"   - [{m['category']}] {m['text'][:300]}" for m in state["matches"][i][:2]
        )
        blocks.append(
            f"Clause {pos}:\n{r['clause']}\n"
            f"Similarity search hint (may be wrong): {r['label']}\n"
            f"Similar clauses from real contracts:\n{precedents}"
        )
    joined = "\n\n".join(blocks)
    return f"""You are helping a small business owner who is about to sign a contract. They are not a lawyer.

Review each clause below. For every clause, read the clause itself and decide what it really is. The similarity search hint is only a hint and may be wrong. Explain using only what the clause says and do not invent details.

Return exactly one item per clause, using the same clause_number as shown.

{joined}"""


def analyzer_agent(state: BatchState):
    analysis = [None] * len(state["results"])
    flagged = [i for i, r in enumerate(state["results"]) if r["label"] != NO_RISK]
    for start in range(0, len(flagged), CHUNK):
        idxs = flagged[start:start + CHUNK]
        batch = cached_call(analyzer_llm, AnalysisBatch, build_analysis_prompt(state, idxs))
        if batch is None:
            continue
        by_number = {item.clause_number: item for item in batch.items}
        for pos, i in enumerate(idxs, start=1):
            analysis[i] = by_number.get(pos)
    return {"analysis": analysis}


# ---------- Agent: NEGOTIATION (rewrites the risky clauses together) ----------
def build_redline_prompt(state, idxs):
    blocks = []
    for pos, i in enumerate(idxs, start=1):
        r = state["results"][i]
        a = state["analysis"][i]
        top = state["matches"][i][0]
        blocks.append(
            f"Clause {pos}:\n{r['clause']}\n"
            f"Type: {a.clause_type} | Risk: {a.risk_level}\n"
            f"Problem: {a.explanation}\n"
            f"Example wording from a similar real contract (style only, not necessarily fair): {top['text'][:300]}"
        )
    joined = "\n\n".join(blocks)
    return f"""You are helping a small business owner negotiate a contract. They are not a lawyer.

Rewrite each clause below so it is more balanced and fair to the small business owner, keeping the same purpose and the same party names. Only change what is needed to fix the problem. Do not add facts, numbers or names that are not in the original clause, and use a clear placeholder like [X days] where a figure must be agreed.

Return exactly one item per clause, using the same clause_number as shown.

{joined}"""


def negotiation_agent(state: BatchState):
    redlines = [None] * len(state["results"])
    need = [
        i for i, a in enumerate(state["analysis"])
        if a is not None and a.risk_level != "Low"
    ]
    for start in range(0, len(need), CHUNK):
        idxs = need[start:start + CHUNK]
        batch = cached_call(negotiator_llm, RedlineBatch, build_redline_prompt(state, idxs))
        if batch is None:
            continue
        by_number = {item.clause_number: item for item in batch.items}
        for pos, i in enumerate(idxs, start=1):
            redlines[i] = by_number.get(pos)
    return {"redlines": redlines}


graph = StateGraph(BatchState)
graph.add_node("split", split_clauses)
graph.add_node("retriever", retriever_agent)
graph.add_node("classifier", classifier_agent)
graph.add_node("analyzer", analyzer_agent)
graph.add_node("negotiator", negotiation_agent)
graph.add_edge(START, "split")
graph.add_edge("split", "retriever")
graph.add_edge("retriever", "classifier")
graph.add_edge("classifier", "analyzer")
graph.add_edge("analyzer", "negotiator")
graph.add_edge("negotiator", END)
app = graph.compile()


if __name__ == "__main__":
    with open("sample_contract.txt", encoding="utf-8") as f:
        text = f.read()

    out = app.invoke({"contract_text": text})

    print()
    print("=== CONTRACT RISK REPORT WITH SUGGESTED CHANGES ===")
    print()
    for i, (r, a, red) in enumerate(zip(out["results"], out["analysis"], out["redlines"]), start=1):
        if r["label"] == NO_RISK:
            continue
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
        print()
    print("Note: these are AI-generated suggestions, not legal advice.")
    print(f"Gemini calls made in this run: {calls_made}")