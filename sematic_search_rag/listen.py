from faster_whisper import WhisperModel
import json
model=WhisperModel("base")


segments  ,info=model.transcribe("videoplay.mp3")
results=[]
for s in segments:
    results.append({"start":s.start,"text":s.text})

with open("New_words.json","w") as f:
    json.dump(results,f,indent=2)
print("saved!")