#!/usr/bin/env bash
#
# transcribe-and-diarize.sh — extract audio, transcribe (Whisper), and
# speaker-diarize (sherpa-onnx) a video/audio file, fully locally.
#
# Usage:
#   ./transcribe-and-diarize.sh <media-file> [num_speakers]
#
#   <media-file>   any ffmpeg-readable file (.webm/.mp4/.mov/.m4a/.wav/...)
#   [num_speakers] optional fixed speaker count (e.g. 4). Omit or 0/-1 to
#                  auto-detect. Forcing the known count is usually cleaner.
#
# Outputs (next to the input file, using its base name):
#   <base> transcript.txt / .srt / .vtt   plain Whisper transcript
#   <base>.diarized.txt / .srt            transcript with SPEAKER_xx labels
#
# Everything runs offline. The Whisper model and the sherpa-onnx env/models
# are cached after first use, so re-runs skip all setup.
#
set -euo pipefail

# --- config ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIARIZE_DIR="$SCRIPT_DIR/.diarize"
VENV_PY="$DIARIZE_DIR/venv/bin/python"
WHISPER_MODEL="$HOME/.cache/whisper-models/ggml-large-v3-turbo-q5_0.bin"
SEG_MODEL="$DIARIZE_DIR/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB_MODEL="$DIARIZE_DIR/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"

# --- args -----------------------------------------------------------------
if [[ $# -lt 1 ]]; then
  grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1
fi
INPUT="$1"
NUM_SPEAKERS="${2:--1}"
[[ -f "$INPUT" ]] || { echo "error: no such file: $INPUT" >&2; exit 1; }

DIR="$(cd "$(dirname "$INPUT")" && pwd)"
BASE="$(basename "${INPUT%.*}")"
WAV="$DIR/$BASE.wav"

# --- preflight ------------------------------------------------------------
command -v ffmpeg      >/dev/null || { echo "error: ffmpeg not found (brew install ffmpeg)" >&2; exit 1; }
command -v whisper-cli >/dev/null || { echo "error: whisper-cli not found (brew install whisper-cpp)" >&2; exit 1; }
for f in "$WHISPER_MODEL" "$VENV_PY" "$SEG_MODEL" "$EMB_MODEL"; do
  [[ -e "$f" ]] || { echo "error: missing dependency: $f" >&2
    echo "       (run the Day 1 setup once to cache models + venv)" >&2; exit 1; }
done

# --- 1. extract 16 kHz mono WAV (required by both tools) -------------------
echo ">> [1/3] extracting audio -> $WAV"
ffmpeg -y -i "$INPUT" -vn -ar 16000 -ac 1 -c:a pcm_s16le "$WAV" -loglevel error

# --- 2. transcribe --------------------------------------------------------
echo ">> [2/3] transcribing (Whisper large-v3-turbo)"
whisper-cli -m "$WHISPER_MODEL" -f "$WAV" -l en \
  -otxt -osrt -ovtt -of "$DIR/$BASE transcript" 2>/dev/null

# --- 3. diarize + merge ---------------------------------------------------
echo ">> [3/3] diarizing (sherpa-onnx, num_speakers=$NUM_SPEAKERS)"
"$VENV_PY" "$DIARIZE_DIR/diarize.py" \
  "$WAV" "$DIR/$BASE transcript.srt" "$DIR/$BASE" "$NUM_SPEAKERS"

# --- cleanup intermediate WAV ---------------------------------------------
rm -f "$WAV"

echo ""
echo "Done. Outputs in: $DIR"
echo "  - $BASE transcript.txt / .srt / .vtt   (plain transcript)"
echo "  - $BASE.diarized.txt / .srt            (speaker-labeled)"
