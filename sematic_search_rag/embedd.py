import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("words.json") as f:
    lines = json.load(f)

print("I have", len(lines), "lines")

texts=[line["text"] for line in lines]
vectors=model.encode(texts)
print("shape:",vectors.shape)