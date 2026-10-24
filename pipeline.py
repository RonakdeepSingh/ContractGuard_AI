import os
import re
from collections import defaultdict
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "contractguard"
MODEL = "llama-text-embed-v2"
TOP_K = 5
THRESHOLD = 0.50  # minimum similarity to call a clause "risky"  # minimum similarity to call a clause "risky"
BATCH = 96

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(name=INDEX_NAME)


class ContractState(TypedDict):
    contract_text: str
    clauses: list
    matches: list
    results: list


# ---------- Step A: split the contract into clauses ----------
def split_clauses(state: ContractState):
    parts = re.split(r"\n\s*\n", state["contract_text"])
    clauses = [" ".join(p.split()) for p in parts]
    clauses = [c for c in clauses if len(c) >= 60]
    return {"clauses": clauses}


# ---------- Agent: RETRIEVER ----------
def retriever_agent(state: ContractState):
    all_matches = []
    clauses = state["clauses"]
    for start in range(0, len(clauses), BATCH):
        chunk = clauses[start:start + BATCH]
        vectors = pc.inference.embed(
            model=MODEL,
            inputs=[c[:3000] for c in chunk],
            parameters={"input_type": "query"},
        )
        for v in vectors:
            res = index.documents.search(
                namespace="__default__",
                top_k=TOP_K,
                score_by=[{"type": "dense_vector", "fields": ["embedding"], "values": v["values"]}],
                include_fields=["*"],
            )
            all_matches.append([
                {
                    "score": m.score,
                    "category": m.get("category"),
                    "why_risky": m.get("why_risky"),
                    "text": m.get("text"),
                }
                for m in res.matches
            ])
    return {"matches": all_matches}


# ---------- Agent: CLASSIFIER (nearest-neighbour voting) ----------
def classifier_agent(state: ContractState):
    results = []
    for clause, matches in zip(state["clauses"], state["matches"]):
        top_score = max((m["score"] for m in matches), default=0.0)
        if matches and top_score >= THRESHOLD:
            votes = defaultdict(float)
            for m in matches:
                votes[m["category"]] += m["score"]
            best = max(votes, key=votes.get)
            confidence = votes[best] / sum(votes.values())
            why = next(m["why_risky"] for m in matches if m["category"] == best)
            label = best
        else:
            label, confidence, why = "No risky pattern found", 0.0, ""
        results.append({
            "clause": clause,
            "label": label,
            "confidence": confidence,
            "top_score": top_score,
            "why_risky": why,
        })
    return {"results": results}


# ---------- Wire the agents together with LangGraph ----------
graph = StateGraph(ContractState)
graph.add_node("split", split_clauses)
graph.add_node("retriever", retriever_agent)
graph.add_node("classifier", classifier_agent)
graph.add_edge(START, "split")
graph.add_edge("split", "retriever")
graph.add_edge("retriever", "classifier")
graph.add_edge("classifier", END)
app = graph.compile()


if __name__ == "__main__":
    with open("sample_contract.txt", encoding="utf-8") as f:
        text = f.read()

    out = app.invoke({"contract_text": text})

    for i, r in enumerate(out["results"], start=1):
        print(f"Clause {i}: {r['clause'][:90]}...")
        print(f"   Label: {r['label']}  (vote share {r['confidence']:.2f}, best similarity {r['top_score']:.3f})")
        if r["why_risky"]:
            print(f"   Why it matters: {r['why_risky']}")
        print()