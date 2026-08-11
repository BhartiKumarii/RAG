from sentence_transformers import SentenceTransformer,util

model = SentenceTransformer("all-MiniLM-L6-v2")

a = model.encode("price")
b = model.encode("cost")
c = model.encode("banana")

print("price vs cost   :", util.cos_sim(a, b))
print("price vs banana :", util.cos_sim(a, c))