#!/usr/bin/env bash
#
# run_all.sh — turn one video into a complete corpus folder.
#
# Runs, in order:
#   1. transcribe_diarize.sh   speech  -> diarized, timestamped transcript
#   2. extract_frames.sh       video   -> frame_*.jpg (1 fps)
#   3. ocr.py                  frames  -> on-screen slide text
#   4. yolo_detect.py          frames  -> detected objects
#   5. face_detect.py          frames  -> detected/clustered faces   [opt-in]
#
# Face detection is OFF by default. On a screen-recorded session the webcams
# are usually off, so it finds nothing and costs ~45-60 min of CPU. Pass
# --faces only when the recording actually shows people.
#
# Everything runs offline. The result is a folder of transcript + JSON that
# the starter/ loader turns into searchable rows. You (the intern) do not
# change anything in here — you consume its output.
#
# Usage:
#   ./run_all.sh [--faces] <video-file> <corpus-out-dir> [num_speakers]
set -euo pipefail

DO_FACES=0
POS=()
for arg in "$@"; do
  case "$arg" in
    --faces) DO_FACES=1 ;;
    *) POS+=("$arg") ;;
  esac
done

if [[ ${#POS[@]} -lt 2 ]]; then
  grep '^#' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

VIDEO="${POS[0]}"
OUT="${POS[1]}"
NUM_SPEAKERS="${POS[2]:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ -f "$VIDEO" ]] || { echo "error: no such file: $VIDEO" >&2; exit 1; }
mkdir -p "$OUT/frames"

echo "== 1/5 transcribe + diarize ==" >&2
"$HERE/transcribe_diarize.sh" "$VIDEO" ${NUM_SPEAKERS:+$NUM_SPEAKERS}
# transcribe_diarize writes next to the video; move its outputs into OUT.
vdir="$(cd "$(dirname "$VIDEO")" && pwd)"
vbase="$(basename "${VIDEO%.*}")"
for ext in "transcript.txt" "transcript.srt" "transcript.vtt" ".diarized.txt" ".diarized.srt"; do
  src="$vdir/$vbase$ext"
  [[ -e "$src" ]] && mv "$src" "$OUT/"
done

echo "== 2/5 extract frames ==" >&2
"$HERE/extract_frames.sh" "$VIDEO" "$OUT/frames" 1

# Build the frame-path payload once and reuse it.
FRAMES_JSON="$(python3 - "$OUT/frames" <<'PY'
import json, os, sys
d = sys.argv[1]
paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("frame_"))
print(json.dumps({"framePaths": paths, "outputDir": d}))
PY
)"

echo "== 3/5 OCR (slide text) ==" >&2
echo "$FRAMES_JSON" | python3 "$HERE/ocr.py" > "$OUT/ocr.json"

echo "== 4/5 YOLO objects ==" >&2
echo "$FRAMES_JSON" | python3 "$HERE/yolo_detect.py" > "$OUT/objects.json"

if [[ "$DO_FACES" == "1" ]]; then
  echo "== 5/5 faces (this is slow) ==" >&2
  echo "$FRAMES_JSON" | python3 "$HERE/face_detect.py" > "$OUT/faces.json"
else
  echo "== 5/5 faces SKIPPED (pass --faces to enable) ==" >&2
fi

echo "done -> $OUT (transcript + ocr.json + objects.json${DO_FACES:+ + faces.json})" >&2
