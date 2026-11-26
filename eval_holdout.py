import re
from hybrid import vector_ids, hybrid_ids, get_chunk
from rewrite import rewrite

TESTS = [
    ("who is the principal officer and what is their role",
     r"Principal\s+Officer\W+means"),                                            # p.9
    ("which details must be collected from a trust customer",
     r"names\s+of\s+the\s+beneficiaries,\s+trustees,\s+settlor"),                # p.36
    ("do all holders of a joint account have to be verified",
     r"CDD\s+Procedure\s+is\s+followed\s+for\s+all\s+the\s+joint\s+account\s+holders"),  # p.19
    ("when must a bank shut an account because of missing identity papers",
     r"RE\s+shall\s+close\s+the\s+account"),                                     # p.44
    ("can overseas Indians send certified copies of their ID",
     r"Non-Resident\s+Indians\s+\(NRIs\)\s+and\s+Persons\s+of\s+Indian\s+Origin"),  # p.6
    ("what ownership share makes someone the real owner of a company",
     r"more\s+than\s+10\s+percent\s+of\s+the\s+shares"),                         # p.5
    ("must banks screen customers against terrorist watch lists",
     r"Unlawful\s+Activities\s+\(Prevention\)"),                                 # p.53 etc.
    ("can a bank depend on another institution to do customer checks",
     r"rely\s+on\s+customer\s+due\s+diligence\s+done\s+by\s+a\s+third\s+party"),  # p.22
    ("what business proof is accepted for a one-person firm",
     r"in\s+the\s+name\s+of\s+the\s+sole\s+proprietor"),                         # p.35
]


def rank_in(ids, pattern, k=10):
    for i, id_ in enumerate(ids[:k]):
        if re.search(pattern, get_chunk(id_)[0], re.I):
            return i + 1
    return None


METHODS = {
    "vector": vector_ids,
    "hybrid": hybrid_ids,
    "rewrite": lambda q: hybrid_ids(rewrite(q)),
}

if __name__ == "__main__":
    top1 = {m: 0 for m in METHODS}
    top5 = {m: 0 for m in METHODS}
    print(f"{'vector':>7} {'hybrid':>7} {'rewrite':>8} | question")
    for q, pattern in TESTS:
        row = {m: rank_in(fn(q), pattern) for m, fn in METHODS.items()}
        for m, r in row.items():
            top1[m] += r == 1
            top5[m] += r is not None and r <= 5
        print(f"{str(row['vector'] or 'MISS'):>7} {str(row['hybrid'] or 'MISS'):>7} "
              f"{str(row['rewrite'] or 'MISS'):>8} | {q}")
    n = len(TESTS)
    print(f"\ntop-1: { {m: f'{c}/{n}' for m, c in top1.items()} }")
    print(f"top-5: { {m: f'{c}/{n}' for m, c in top5.items()} }")
    print("\nREWRITES:")
    for q, _ in TESTS:
        print(f"- {q}\n    -> {rewrite(q)}")