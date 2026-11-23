import re
import chromadb
from sentence_transformers import SentenceTransformer
from hybrid import vector_ids, bm25_ids, hybrid_ids, get_chunk

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
col = chromadb.PersistentClient("chroma_db").get_collection("kyc")
PREFIX = "Represent this sentence for searching relevant passages: "

# (question in plain words, regex a correct chunk must contain)  # page in the PDF
TESTS = [
    ("para 18 V-CIP geo-tagging",
     r"geo\s?-\s?tagging"),                                                    # p.28
    ("does the video call have to record where the customer is",
     r"geo\s?-\s?tagging"),                                                    # p.28
    ("can a customer be verified remotely",
     r"Video[\s-]*based\s+Customer\s+Identification|first\s+option\s+to\s+the\s+customer\s+for\s+remote\s+onboarding"),  # p.13 or p.45                         # p.13
    ("what documents are needed for digital KYC",
     r"Digital\s+KYC\s+Process"),                                              # p.71
    ("how long do banks have to keep transaction records",
     r"at\s+least\s+five\s+years\s+from\s+the\s+date\s+of\s+transaction"),     # p.49
    ("how often must KYC be refreshed for risky customers",
     r"at\s+least\s+once\s+in\s+every\s+two\s+years"),                         # p.39
    ("how quickly must KYC be uploaded to the central registry",
     r"within\s+10\s+days\s+of\s+commencement"),                               # p.58
    ("what extra checks apply to politically exposed persons",
     r"Accounts\s+of\s+Politically\s+Exposed"),                                # p.46
    ("does each customer get a single unique ID code",
     r"Unique\s+Customer\s+Identification\s+Code\s+\(UCIC\)\s+shall\s+be\s+allotted"),  # p.61
    ("what is a shell bank",
     r"Shell\s+Bank\W+means"),                                                 # p.13
    ("when must the relaxations for small accounts be reviewed",
     r"reviewed\s+after\s+twenty-four\s+months"),                              # p.33
    ("which electronic utilities are used to file cash and suspicious transaction reports",
     r"Cash\s+Transaction\s+Reports"),                                         # p.51
]

def rank_in(ids, pattern, k=10):
    for i, id_ in enumerate(ids[:k]):
        if re.search(pattern, get_chunk(id_)[0], re.I):
            return i + 1
    return None


if __name__ == "__main__":
    METHODS = {"vector": vector_ids, "bm25": bm25_ids, "hybrid": hybrid_ids}
    top5 = {m: 0 for m in METHODS}
    print(f"{'vector':>7} {'bm25':>6} {'hybrid':>7} | question")
    for q, pattern in TESTS:
        row = {m: rank_in(fn(q), pattern) for m, fn in METHODS.items()}
        for m, r in row.items():
            top5[m] += r is not None and r <= 5
        print(f"{str(row['vector'] or 'MISS'):>7} {str(row['bm25'] or 'MISS'):>6} "
              f"{str(row['hybrid'] or 'MISS'):>7} | {q}")
    n = len(TESTS)
    print("\ntop-5:", {m: f"{c}/{n}" for m, c in top5.items()})