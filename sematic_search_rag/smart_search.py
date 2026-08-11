import json
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("words.json") as f:
    lines = json.load(f)

texts = [line["text"] for line in lines]
vectors = model.encode(texts)

question = input("Search for: ")
q_vector = model.encode(question)

scores = util.cos_sim(q_vector, vectors)[0]

results = []

for i, line in enumerate(lines):
    results.append((float(scores[i]), line["start"], line["text"]))

results.sort(reverse=True)

for score, start, text in results[:3]:
    print(round(score, 2), "|", start, "|", text)