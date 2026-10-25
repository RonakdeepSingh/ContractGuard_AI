import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "contractguard"
MODEL = "llama-text-embed-v2"
PER_CATEGORY = 20   # test clauses taken from each risky clause type
NEGATIVES = 150     # harmless clauses used to measure false alarms
TOP_K = 5
SEED = 42
HARMLESS_TYPES = ["Governing Law", "Parties", "Agreement Date", "Effective Date"]

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(name=INDEX_NAME)
random.seed(SEED)


def embed_all(texts):
    vectors = []
    for start in range(0, len(texts), 96):
        chunk = [t[:3000] for t in texts[start:start + 96]]
        for attempt in range(5):
            try:
                res = pc.inference.embed(model=MODEL, inputs=chunk, parameters={"input_type": "query"})
                vectors.extend(v["values"] for v in res)
                break
            except Exception as e:
                wait = 15 * (attempt + 1)
                print("Embed hiccup:", str(e)[:100], "- waiting", wait, "s")
                time.sleep(wait)
        else:
            raise RuntimeError("Embedding kept failing. Try again in a few minutes.")
        print(f"  embedded {min(start + 96, len(texts))} / {len(texts)}")
    return vectors


def search(vector, k):
    for attempt in range(5):
        try:
            res = index.documents.search(
                namespace="__default__",
                top_k=k,
                score_by=[{"type": "dense_vector", "fields": ["embedding"], "values": vector}],
                include_fields=["*"],
            )
            return [{"score": m.score, "category": m.get("category"), "contract": m.get("contract")} for m in res.matches]
        except Exception as e:
            wait = 10 * (attempt + 1)
            print("Search hiccup:", str(e)[:100], "- waiting", wait, "s")
            time.sleep(wait)
    raise RuntimeError("Search kept failing. Try again in a few minutes.")


# ---------- Test set 1: risky clauses (their own contract is hidden from the search) ----------
with open("clauses.json", encoding="utf-8") as f:
    clauses = json.load(f)
by_cat = defaultdict(list)
for c in clauses:
    by_cat[c["category"]].append(c)
tests = []
for cat, items in sorted(by_cat.items()):
    tests += random.sample(items, min(PER_CATEGORY, len(items)))

# ---------- Test set 2: harmless clauses (governing law, parties, dates) ----------
with open(Path("data") / "CUADv1.json", encoding="utf-8") as f:
    cuad = json.load(f)
harmless = []
for contract in cuad["data"]:
    for para in contract["paragraphs"]:
        for qa in para["qas"]:
            m = re.search(r'related to "(.*?)"', qa["question"])
            if m and m.group(1) in HARMLESS_TYPES:
                for ans in qa["answers"]:
                    text = " ".join(ans["text"].split())
                    if len(text) >= 60:
                        harmless.append(text)
harmless = list(dict.fromkeys(harmless))
negatives = random.sample(harmless, min(NEGATIVES, len(harmless)))

print(f"Testing {len(tests)} risky clauses and {len(negatives)} harmless clauses...")
print("Step 1: risky clauses")
pos_vecs = embed_all([c["text"] for c in tests])
pos_rows = []
for i, (c, v) in enumerate(zip(tests, pos_vecs), start=1):
    matches = [m for m in search(v, 20) if m["contract"] != c["contract"]][:TOP_K]
    if not matches:
        continue
    votes = defaultdict(float)
    for m in matches:
        votes[m["category"]] += m["score"]
    pos_rows.append({"true": c["category"], "pred": max(votes, key=votes.get), "best": max(m["score"] for m in matches)})
    if i % 50 == 0:
        print(f"  searched {i} / {len(tests)}")

print("Step 2: harmless clauses")
neg_vecs = embed_all(negatives)
neg_best = [max(m["score"] for m in search(v, TOP_K)) for v in neg_vecs]

# ---------- Results ----------
n = len(pos_rows)
correct = sum(r["true"] == r["pred"] for r in pos_rows)
print()
print("=" * 60)
print("RESULTS (similarity-voting Classifier, no Gemini involved)")
print("=" * 60)
print(f"Clause-type accuracy (own contract hidden): {correct} / {n} = {100 * correct / n:.1f}%")

print()
print("Accuracy per clause type:")
per = defaultdict(lambda: [0, 0])
for r in pos_rows:
    per[r["true"]][1] += 1
    per[r["true"]][0] += r["true"] == r["pred"]
for cat, (ok, tot) in sorted(per.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
    print(f"  {cat:36s} {ok:3d} / {tot:<3d} {100 * ok / tot:5.1f}%")

print()
print("Most common mix-ups (true -> predicted):")
conf = Counter((r["true"], r["pred"]) for r in pos_rows if r["true"] != r["pred"])
for (t, p), k in conf.most_common(5):
    print(f"  {t} -> {p}: {k}")

print()
print("Threshold trade-off (higher = fewer false alarms, but more risky clauses missed):")
print("  threshold | risky clauses flagged | harmless clauses wrongly flagged")
for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    flagged = sum(r["best"] >= thr for r in pos_rows)
    false_alarm = sum(b >= thr for b in neg_best)
    print(f"     {thr:.2f}   |        {100 * flagged / n:5.1f}%         |          {100 * false_alarm / len(neg_best):5.1f}%")
print()
print("Note: harmless clauses here are governing-law, parties and date clauses only.")