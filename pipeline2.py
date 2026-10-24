import time
from typing import Literal

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from pipeline import ContractState, classifier_agent, retriever_agent, split_clauses

load_dotenv()

GEMINI_MODEL = "gemini-3.8-flash"  # if Google renames it, change it here only
NO_RISK = "No risky pattern found"


class Analysis(BaseModel):
    clause_type: str = Field(description="What kind of clause this really is, for example 'Uncapped Liability'")
    risk_level: Literal["Low", "Medium", "High"]
    explanation: str = Field(description="2 to 3 sentences in plain English for a non-lawyer")
    what_to_check: str = Field(description="One practical thing the signer should ask for or check")


llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
analyzer_llm = llm.with_structured_output(Analysis)


class AnalyzedState(ContractState):
    analysis: list


def build_prompt(clause, predicted, matches):
    precedents = "\n".join(f"- [{m['category']}] {m['text'][:400]}" for m in matches[:3])
    return f"""You are helping a small business owner who is about to sign a contract. They are not a lawyer.

Clause to review:
{clause}

A similarity search labelled this clause as "{predicted}". That label is only a hint and may be wrong, so read the clause itself and decide.

Similar clauses from real contracts, for comparison:
{precedents}

Explain the clause using only what the clause says. Do not invent details."""


# ---------- Agent: ANALYZER ----------
def analyzer_agent(state: AnalyzedState):
    analysis = []
    for r, matches in zip(state["results"], state["matches"]):
        if r["label"] == NO_RISK:
            analysis.append(None)
            continue

        prompt = build_prompt(r["clause"], r["label"], matches)
        result = None
        for attempt in range(3):
            try:
                result = analyzer_llm.invoke(prompt)
                break
            except Exception as e:
                wait = 20 * (attempt + 1)
                print("Gemini hiccup:", str(e)[:100], "- waiting", wait, "s")
                time.sleep(wait)
        analysis.append(result)
        time.sleep(4)  # stay under the free-tier request limit
    return {"analysis": analysis}


graph = StateGraph(AnalyzedState)
graph.add_node("split", split_clauses)
graph.add_node("retriever", retriever_agent)
graph.add_node("classifier", classifier_agent)
graph.add_node("analyzer", analyzer_agent)
graph.add_edge(START, "split")
graph.add_edge("split", "retriever")
graph.add_edge("retriever", "classifier")
graph.add_edge("classifier", "analyzer")
graph.add_edge("analyzer", END)
app = graph.compile()


if __name__ == "__main__":
    with open("sample_contract.txt", encoding="utf-8") as f:
        text = f.read()

    out = app.invoke({"contract_text": text})

    print("=== CONTRACT RISK REPORT ===")
    print()
    for i, (r, a) in enumerate(zip(out["results"], out["analysis"]), start=1):
        if a is None:
            continue
        print(f"Clause {i}: {r['clause'][:100]}...")
        print(f"   Type: {a.clause_type}   |   Risk: {a.risk_level}")
        print(f"   In plain English: {a.explanation}")
        print(f"   What to check: {a.what_to_check}")
        print()