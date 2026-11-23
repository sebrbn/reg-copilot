import re
import chromadb
from sentence_transformers import SentenceTransformer

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
     r"Video[\s-]*based\s+Customer\s+Identification"),                         # p.13
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


def rank_of(q, pattern, k=10):
    emb = model.encode([PREFIX + q], normalize_embeddings=True).tolist()
    docs = col.query(query_embeddings=emb, n_results=k)["documents"][0]
    for i, d in enumerate(docs):
        if re.search(pattern, d, re.I):
            return i + 1
    return None


if __name__ == "__main__":
    found = top1 = top5 = 0
    for q, pattern in TESTS:
        r = rank_of(q, pattern)
        found += r is not None
        top1 += r == 1
        top5 += r is not None and r <= 5
        print(f"rank {r or 'MISS':>4} | {q}")
    n = len(TESTS)
    print(f"\ntop-1: {top1}/{n}   top-5: {top5}/{n}   top-10: {found}/{n}")