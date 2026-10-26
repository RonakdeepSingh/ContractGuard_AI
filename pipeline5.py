import re
from typing import Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import pipeline4 as p4
from pipeline import classifier_agent, retriever_agent, split_clauses

NO_RISK = p4.NO_RISK
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


# ---------- Simple rule-based check: figures that appear from nowhere ----------
def new_numbers(original, rewrite):
    rewrite = re.sub(r"\[[^\]]*\]", "", rewrite)  # [X days] placeholders are allowed
    return sorted(set(NUMBER.findall(rewrite)) - set(NUMBER.findall(original)))


# ---------- What the Critic must return ----------
class ClauseReview(BaseModel):
    clause_number: int = Field(description="The clause number exactly as given in the prompt")
    verdict: Literal["Approved", "Revised"]
    issues: list[str] = Field(description="Short plain-English list of the problems found; empty if Approved")
    final_explanation: str = Field(description="The explanation, corrected if needed (copy unchanged if Approved)")
    final_rewrite: str = Field(description="The rewritten clause, corrected if needed (copy unchanged if Approved)")
    final_what_changed: str = Field(description="What changed, updated if the rewrite was corrected")
    final_fallback: str = Field(description="The fallback request, updated if needed")


class ReviewBatch(BaseModel):
    items: list[ClauseReview]


critic_llm = p4.llm.with_structured_output(ReviewBatch)


class CriticState(p4.BatchState):
    reviews: list


def build_critic_prompt(state, idxs):
    blocks = []
    for pos, i in enumerate(idxs, start=1):
        r, a, red = state["results"][i], state["analysis"][i], state["redlines"][i]
        extra = new_numbers(r["clause"], red.suggested_clause)
        check = ("figures that are not in the original clause: " + ", ".join(extra)) if extra else "no new figures found"
        blocks.append(
            f"Clause {pos}:\n"
            f"Original clause: {r['clause']}\n"
            f"Explanation given: {a.explanation}\n"
            f"Proposed rewrite: {red.suggested_clause}\n"
            f"What changed (as stated): {red.what_changed}\n"
            f"Fallback: {red.fallback}\n"
            f"Automatic check: {check}"
        )
    joined = "\n\n".join(blocks)
    return f"""You are a careful reviewer checking another AI's work for a small business owner who is not a lawyer.

For each clause below, check three things:
1. Does the proposed rewrite add any fact, number, name, duty or idea that is not in the original clause? Figures inside [square brackets] are placeholders and are fine.
2. Does the rewrite change the meaning of the clause more than is needed to fix the problem?
3. Does the explanation make legal judgements or use strong words (such as illegal, improper, extraordinary, unfair) that the clause text does not support? It should describe what the clause says and the risk in neutral words.

If none of these problems exist, the verdict is "Approved" and you copy the texts unchanged.
If any problem exists, the verdict is "Revised": list the problems in plain English, then give corrected texts. Keep the same party names and purpose, and use placeholders like [X days] for any figure that must be agreed.

Return exactly one item per clause, using the same clause_number as shown.

{joined}"""


# ---------- Agent: CRITIC ----------
def critic_agent(state: CriticState):
    reviews = [None] * len(state["results"])
    need = [i for i in range(len(reviews)) if state["analysis"][i] is not None and state["redlines"][i] is not None]
    for start in range(0, len(need), p4.CHUNK):
        idxs = need[start:start + p4.CHUNK]
        batch = p4.cached_call(critic_llm, ReviewBatch, build_critic_prompt(state, idxs))
        if batch is None:
            continue
        by_number = {item.clause_number: item for item in batch.items}
        for pos, i in enumerate(idxs, start=1):
            reviews[i] = by_number.get(pos)
    return {"reviews": reviews}


graph = StateGraph(CriticState)
graph.add_node("split", split_clauses)
graph.add_node("retriever", retriever_agent)
graph.add_node("classifier", classifier_agent)
graph.add_node("analyzer", p4.analyzer_agent)
graph.add_node("negotiator", p4.negotiation_agent)
graph.add_node("critic", critic_agent)
graph.add_edge(START, "split")
graph.add_edge("split", "retriever")
graph.add_edge("retriever", "classifier")
graph.add_edge("classifier", "analyzer")
graph.add_edge("analyzer", "negotiator")
graph.add_edge("negotiator", "critic")
graph.add_edge("critic", END)
app = graph.compile()


# ---------- Build the final report text ----------
def format_report(out):
    lines = ["=== CONTRACT RISK REPORT (checked by Critic) ===", ""]
    shown = 0
    rows = zip(out["results"], out["analysis"], out["redlines"], out["reviews"])
    for i, (r, a, red, rev) in enumerate(rows, start=1):
        if r["label"] == NO_RISK:
            continue
        shown += 1
        lines.append(f"Clause {i}: {r['clause'][:100]}...")
        if a is None:
            if r["top_score"] < p4.AI_THRESHOLD:
                lines += [f"   Possible {r['label']} (low confidence, not reviewed by AI)", ""]
            else:
                lines += ["   (analysis unavailable this run, please try again)", ""]
            continue
        revised = rev is not None and rev.verdict == "Revised"
        lines.append(f"   Type: {a.clause_type}   |   Risk: {a.risk_level}")
        lines.append(f"   In plain English: {rev.final_explanation if revised else a.explanation}")
        if red is not None:
            rewrite = rev.final_rewrite if revised else red.suggested_clause
            lines.append(f"   Suggested rewrite: {rewrite}")
            lines.append(f"   What changed: {rev.final_what_changed if revised else red.what_changed}")
            lines.append(f"   If they refuse, ask for: {rev.final_fallback if revised else red.fallback}")
            if rev is None:
                lines.append("   Critic check: not available this run (shown unchecked)")
            elif revised:
                lines.append("   Critic check: REVISED - " + "; ".join(rev.issues))
            else:
                lines.append("   Critic check: Approved")
            extra = new_numbers(r["clause"], rewrite)
            if extra:
                lines.append("   Automatic check: figures not in the original clause: " + ", ".join(extra))
        elif a.risk_level != "Low":
            lines.append("   (no rewrite available this run, please try again)")
        lines.append("")
    if shown == 0:
        lines.append("No risky clauses were found.")
    lines.append("Note: these are AI-generated suggestions, not legal advice.")
    lines.append(f"Gemini calls made in this run: {p4.calls_made}")
    return "\n".join(lines)


def analyze(contract_text):
    return format_report(app.invoke({"contract_text": contract_text}))


if __name__ == "__main__":
    with open("sample_contract.txt", encoding="utf-8") as f:
        print(analyze(f.read()))
