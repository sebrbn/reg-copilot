import re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
col = chromadb.PersistentClient("chroma_db").get_collection("kyc")
PREFIX = "Represent this sentence for searching relevant passages: "

data = col.get(include=["documents", "metadatas"])
IDS, DOCS, METAS = data["ids"], data["documents"], data["metadatas"]
BY_ID = {id_: (d, m) for id_, d, m in zip(IDS, DOCS, METAS)}


def tokenize(text):
    # lowercase words/numbers only, so "V -CIP" and "V-CIP" both become: v, cip
    return re.findall(r"[a-z0-9]+", text.lower())


bm25 = BM25Okapi([tokenize(d) for d in DOCS])


def vector_ids(q, k=30):
    emb = model.encode([PREFIX + q], normalize_embeddings=True).tolist()
    return col.query(query_embeddings=emb, n_results=k)["ids"][0]


def bm25_ids(q, k=30):
    scores = bm25.get_scores(tokenize(q))
    order = sorted(range(len(DOCS)), key=lambda i: -scores[i])[:k]
    return [IDS[i] for i in order]


def hybrid_ids(q, k=10, pool=30, c=60):
    fused = {}
    for ranking in (vector_ids(q, pool), bm25_ids(q, pool)):
        for rank, id_ in enumerate(ranking, start=1):
            fused[id_] = fused.get(id_, 0) + 1 / (c + rank)
    return sorted(fused, key=fused.get, reverse=True)[:k]


def get_chunk(id_):
    return BY_ID[id_]