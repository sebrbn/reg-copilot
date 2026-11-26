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

NEW_PARA = re.compile(r"^\d{1,3}\.\s")


def split_long(part, max_chars):
    """Break an oversized clause into pieces at sentence ends."""
    if len(part) <= max_chars:
        return [part]
    sentences = re.split(r"(?<=[.;:])\s+(?=[A-Z(\d\u2022])", part)
    pieces, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > max_chars:
            pieces.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        pieces.append(cur)
    return pieces


def chunk_page(text, max_chars=1000):
    parts = [p.strip() for p in CLAUSE.split(text) if p.strip()]
    parts = [piece for p in parts for piece in split_long(p, max_chars)]
    chunks, cur = [], ""
    for p in parts:
        starts_new_para = bool(NEW_PARA.match(p))
        if cur and (starts_new_para or len(cur) + len(p) > max_chars):
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