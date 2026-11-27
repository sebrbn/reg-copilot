from hybrid import hybrid_ids
from rewrite import rewrite


def fused_ids(q, k=10, pool=30, c=60):
    scores = {}
    for ranking in (hybrid_ids(q, k=pool), hybrid_ids(rewrite(q), k=pool)):
        for rank, id_ in enumerate(ranking, start=1):
            scores[id_] = scores.get(id_, 0) + 1 / (c + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]