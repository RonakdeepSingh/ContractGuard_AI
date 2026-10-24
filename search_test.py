import os

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "contractguard"
MODEL = "llama-text-embed-v2"

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(name=INDEX_NAME)

query_text = (
    "This Agreement will automatically renew for successive one year terms "
    "unless either party gives written notice of non-renewal at least "
    "ninety days before the end of the then-current term."
)

q = pc.inference.embed(
    model=MODEL,
    inputs=[query_text],
    parameters={"input_type": "query"},
)

result = index.documents.search(
    namespace="__default__",
    top_k=3,
    score_by=[{"type": "dense_vector", "fields": ["embedding"], "values": q[0]["values"]}],
    include_fields=["*"],
)

print("Query clause:")
print(query_text)
print()
for i, m in enumerate(result.matches, start=1):
    print(f"--- Match {i} (similarity {m.score:.3f}) ---")
    print("Category:", m.get("category"))
    print("Why risky:", m.get("why_risky"))
    print("Text:", (m.get("text") or "")[:250])
    print()