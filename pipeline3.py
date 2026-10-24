import time

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from pipeline import classifier_agent, retriever_agent, split_clauses
from pipeline2 import AnalyzedState, analyzer_agent, llm

load_dotenv()


class Redline(BaseModel):
    problem: str = Field(description="One sentence: what is unfair or one-sided about the original clause")
    suggested_clause: str = Field(description="The full rewritten clause, in contract language, more balanced and fair")
    what_changed: str = Field(description="2 to 3 plain-English sentences explaining what was changed and why")
    fallback: str = Field(description="One sentence: a smaller compromise to ask for if the other side refuses the rewrite")


negotiator_llm = llm.with_structured_output(Redline)


class NegotiatedState(AnalyzedState):
    redlines: list


def build_redline_prompt(clause, analysis, matches):
    examples = "\n".join(f"- [{m['category']}] {m['text'][:400]}" for m in matches[:3])
    return f"""You are helping a small business owner negotiate a contract. They are not a lawyer.

Original clause:
{clause}

Review of this clause:
- Type: {analysis.clause_type}
- Risk: {analysis.risk_level}
- Problem: {analysis.explanation}

Examples of how similar clauses are worded in other real contracts (for wording style only; they are not necessarily fair):
{examples}

Rewrite the clause so it is more balanced and fair to the small business owner, keeping the same purpose and the same party names.
Only change what is needed to fix the problem. Do not add facts, numbers or names that are not in the original clause, and use a clear placeholder like [X days] where a figure must be agreed."""


# ---------- Agent: NEGOTIATION ----------
def negotiation_agent(state: NegotiatedState):
    redlines = []
    for r, a, matches in zip(state["results"], state["analysis"], state["matches"]):
        if a is None or a.risk_level == "Low":
            redlines.append(None)
            continue

        prompt = build_redline_prompt(r["clause"], a, matches)
        result = None
        for attempt in range(3):
            try:
                result = negotiator_llm.invoke(prompt)
                break
            except Exception as e:
                wait = 20 * (attempt + 1)
                print("Gemini hiccup:", str(e)[:100], "- waiting", wait, "s")
                time.sleep(wait)
        redlines.append(result)
        time.sleep(4)  # stay under the free-tier request limit
    return {"redlines": redlines}


graph = StateGraph(NegotiatedState)
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

    print("=== CONTRACT RISK REPORT WITH SUGGESTED CHANGES ===")
    print()
    for i, (r, a, red) in enumerate(zip(out["results"], out["analysis"], out["redlines"]), start=1):
        if a is None:
            continue
        print(f"Clause {i}: {r['clause'][:100]}...")
        print(f"   Type: {a.clause_type}   |   Risk: {a.risk_level}")
        print(f"   In plain English: {a.explanation}")
        if red is not None:
            print(f"   Suggested rewrite: {red.suggested_clause}")
            print(f"   What changed: {red.what_changed}")
            print(f"   If they refuse, ask for: {red.fallback}")
        print()
    print("Note: these are AI-generated suggestions, not legal advice.")