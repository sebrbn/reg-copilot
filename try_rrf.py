vector_list = ["A", "B", "C", "D"]   # best first
bm25_list   = ["C", "A", "E", "B"]

scores = {}
for ranking in (vector_list, bm25_list):
    for rank, chunk in enumerate(ranking, start=1):
        scores[chunk] = scores.get(chunk, 0) + 1 / (60 + rank)

for chunk, s in sorted(scores.items(), key=lambda x: -x[1]):
    print(chunk, round(s, 5))