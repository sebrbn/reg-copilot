
import os
import sys
import time

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from sentence_transformers import SentenceTransformer

load_dotenv()

MODELS = [m.strip() for m in os.getenv("GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-3.6-flash").split(",")]
PREFIX = "Represent this sentence for searching relevant passages: "

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

col = chromadb.PersistentClient(
    path="chroma_db"
).get_collection("kyc")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

SYSTEM = """You are a regulatory compliance assistant for the RBI KYC Master Direction.
Answer ONLY using the numbered sources provided. Rules:
1. Cite the page after every claim, like [p.28].
2. If the sources do not contain the answer, reply exactly:
   "The provided sources do not answer this."
3. Do not use outside knowledge, and do not guess.
4. If sources look incomplete or cut off, say so."""


def retrieve(q, k=6):
    emb = embedder.encode(
        [PREFIX + q],
        normalize_embeddings=True
    ).tolist()

    res = col.query(
        query_embeddings=emb,
        n_results=min(k, col.count())
    )

    if not res["documents"] or not res["documents"][0]:
        return []

    return list(zip(
        res["documents"][0],
        res["metadatas"][0]
    ))


def generate_answer(prompt, rounds=2):
    for r in range(rounds):
        for model in MODELS:
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM,
                        temperature=0,
                    ),
                )
                print(f"(answered by {model})")
                return resp.text
            except (errors.ServerError, errors.ClientError) as e:
                # 503/5xx = busy, 429 = rate limit, 404 = bad model name -> try the next one
                code = getattr(e, "code", None)
                if code not in (404, 429) and not isinstance(e, errors.ServerError):
                    raise
                print(f"{model} failed ({code}), trying next model...")
        if r < rounds - 1:
            time.sleep(10)
    raise RuntimeError("All models are busy right now. Try again later.")


def answer(q):

    chunks = retrieve(q)

    if not chunks:
        return "The provided sources do not answer this.", []

    context = "\n\n".join(
        f"[p.{m['page']}]\n{d}"
        for d, m in chunks
    )

    prompt = f"""
SOURCES:
{context}

QUESTION: {q}
"""

    text = generate_answer(prompt)

    return text, [m["page"] for _, m in chunks]


if __name__ == "__main__":

    q = " ".join(sys.argv[1:])

    if not q:
        q = input("Enter your regulatory question: ")

    text, pages = answer(q)

    print("\nANSWER:\n")
    print(text)

    print("\nPages retrieved:", sorted(set(pages)))
