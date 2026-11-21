import sys
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
col = chromadb.PersistentClient("chroma_db").get_collection("kyc")
PREFIX = "Represent this sentence for searching relevant passages: "

q = " ".join(sys.argv[1:])
res = col.query(query_embeddings=model.encode([PREFIX + q], normalize_embeddings=True).tolist(), n_results=5)
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"\n[p.{meta['page']}] distance={dist:.3f}\n{doc[:300]}")