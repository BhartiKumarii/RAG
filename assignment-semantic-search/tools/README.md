# Extraction toolkit

These scripts turn a lecture video into a searchable corpus. **You run them, you
do not build or change them.** Your job in this assignment starts *after* this
folder has produced its output: you take the transcript and the JSON here and
make search over it good.

Everything runs offline on your laptop. No cloud account, no API key.

## What each tool does

| Tool | In | Out | What it gives you |
|------|----|-----|-------------------|
| `transcribe_diarize.sh` | video | `*.diarized.srt/.txt` | speech as timestamped, speaker-labelled text |
| `extract_frames.sh` | video | `frames/frame_*.jpg` | one frame per second, named by millisecond |
| `ocr.py` | frames | `ocr.json` | on-screen slide text per frame |
| `yolo_detect.py` | frames | `objects.json` | objects detected in frames |
| `face_detect.py` | frames | `faces.json` | faces detected and clustered (opt-in) |
| `run_all.sh` | video | a full corpus folder | all of the above, in order |

> **Faces are off by default.** On a screen-recorded session the webcams are
> usually off, so face detection finds nothing and costs 45 to 60 minutes of
> CPU. Pass `--faces` only when the recording actually shows people. For this
> corpus you almost certainly do not need it.

Frames are named `frame_MMMMMMMM.jpg` where `MMMMMMMM` is the millisecond
timestamp. Every tool reads that name back to recover *when* something appeared,
so a search hit can point at a moment in the recording.

## One-shot

```bash
./run_all.sh "path/to/Day 1.webm" corpus/day1 4
#            video                  out-dir    known speaker count (optional)
```

That writes `corpus/day1/` with the transcript plus `ocr.json`, `objects.json`,
and `faces.json`.

## Step by step (if you want to run one stage)

```bash
# 1. transcript (speaker count optional; pass it if you know it)
./transcribe_diarize.sh "Day 1.webm" 4

# 2. frames at 1 fps
./extract_frames.sh "Day 1.webm" corpus/day1/frames 1

# 3-5. feed the frames to each detector (JSON in, JSON out)
python3 - <<'PY' | python3 ocr.py > corpus/day1/ocr.json
import json, os
d = "corpus/day1/frames"
print(json.dumps({"framePaths": [os.path.join(d,f) for f in sorted(os.listdir(d))]}))
PY
```

## Install once

```bash
brew install ffmpeg whisper-cpp tesseract
pip install -r requirements.txt
```

For a language other than English in OCR, install the Tesseract language pack
and set `TESSERACT_LANGS=eng+ara` (for example).

## Why this is handed to you

Pulling clean transcripts, frames, slide text, and detections out of raw video
is real engineering, but it is not the skill this assignment is about. We give
it to you finished so every hour you spend goes into the part that matters:
**embeddings, vector search, and measuring retrieval quality.**
