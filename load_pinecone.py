import json
import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, SchemaBuilder

load_dotenv()

INDEX_NAME = "contractguard"
MODEL = "llama-text-embed-v2"
BATCH = 96

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# 1. Create the index (the "shelf" where clause vectors will live)
if not pc.indexes.exists(name=INDEX_NAME):
    schema = (
        SchemaBuilder()
        .add_dense_vector_field(name="embedding", dimension=1024, metric="cosine")
        .build()
    )
    pc.indexes.create(name=INDEX_NAME, schema=schema)
    print("Creating index...")

while not pc.indexes.describe(name=INDEX_NAME).status.ready:
    time.sleep(2)
print("Index is ready.")

index = pc.Index(name=INDEX_NAME)

# 2. Load the clauses from Step 4
with open("clauses.json", encoding="utf-8") as f:
    clauses = json.load(f)
print("Clauses to upload:", len(clauses))


def embed_with_retry(texts, tries=5):
    for attempt in range(tries):
        try:
            return pc.inference.embed(
                model=MODEL,
                inputs=texts,
                parameters={"input_type": "passage"},
            )
        except Exception as e:
            wait = 15 * (attempt + 1)
            print("Embed hiccup:", str(e)[:120], "- waiting", wait, "s")
            time.sleep(wait)
    raise RuntimeError("Embedding kept failing. Try again in a few minutes.")


# 3. Turn each clause into numbers (embedding) and upload in batches
for start in range(0, len(clauses), BATCH):
    batch = clauses[start:start + BATCH]
    vectors = embed_with_retry([c["text"][:3000] for c in batch])

    docs = []
    for c, v in zip(batch, vectors):
        docs.append({
            "_id": f"clause-{c['id']}",
            "embedding": v["values"],
            "text": c["text"][:3000],
            "category": c["category"],
            "why_risky": c["why_risky"],
            "contract": c["contract"],
        })

    index.documents.upsert(namespace="__default__", documents=docs)
    print(f"Uploaded {min(start + BATCH, len(clauses))} / {len(clauses)}")
    time.sleep(1)

time.sleep(5)
print("Done! All clauses are stored in Pinecone.")