import re
from pathlib import Path
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient("chroma_db")
try:
    client.delete_collection("kyc")
except Exception:
    pass
col = client.create_collection("kyc", metadata={"hnsw:space": "cosine"})

CLAUSE = re.compile(r"\n(?=\s*(?:\d{1,3}\.\d*\s|\([a-z]{1,4}\)\s))")

def chunk_page(text, max_chars=1000):
    parts = [p.strip() for p in CLAUSE.split(text) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + "\n" + p).strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) > 80]

pdf = next(Path("data").glob("MD18KYC*.pdf"))
ids, docs, metas = [], [], []
for pageno, page in enumerate(PdfReader(pdf).pages, start=1):
    for j, c in enumerate(chunk_page(page.extract_text() or "")):
        ids.append(f"p{pageno}-{j}")
        docs.append(c)
        metas.append({"page": pageno})

col.add(ids=ids, documents=docs, metadatas=metas,
        embeddings=model.encode(docs, normalize_embeddings=True, batch_size=64).tolist())
print(len(docs), "chunks")