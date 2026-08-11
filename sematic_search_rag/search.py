import json

with open("words.json") as f:
    lines=json.load(f)

question=input("search for ::")

for line in lines:
    if question in line["text"]:
        print(line["start"],line["text"])

