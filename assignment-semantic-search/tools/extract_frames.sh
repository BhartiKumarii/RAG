#!/usr/bin/env bash
#
# extract_frames.sh — pull keyframes from a video for OCR / object / face
# detection. Frames are named frame_MMMMMMMM.jpg where MMMMMMMM is the
# millisecond timestamp, so the downstream tools recover the moment each
# detection happened.
#
# Usage:
#   ./extract_frames.sh <video-file> <output-dir> [fps]
#
#   <video-file>  any ffmpeg-readable video
#   <output-dir>  directory to write frame_*.jpg into (created if absent)
#   [fps]         frames per second to sample (default 1 = one frame/second)
#
# One frame per second is plenty for a talking-head or slide recording and
# keeps the frame count (and OCR time) sane. Raise it only if slides change
# faster than that.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  grep '^#' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

VIDEO="$1"
OUT="$2"
FPS="${3:-1}"

[[ -f "$VIDEO" ]] || { echo "error: no such file: $VIDEO" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "error: ffmpeg not found (brew install ffmpeg)" >&2; exit 1; }

mkdir -p "$OUT"

# -frame_pts 1 stamps each output frame with its presentation timestamp; with
# an fps filter the number ffmpeg writes is the frame count, so we instead use
# a strftime-free millisecond name via the timestamp expression.
ffmpeg -hide_banner -loglevel error -i "$VIDEO" \
  -vf "fps=${FPS}" \
  -frame_pts 1 \
  -q:v 3 \
  "$OUT/frame_%08d.jpg"

# ffmpeg's %08d with -frame_pts writes the pts in the stream time_base, which
# for fps output is the frame index, not milliseconds. Rename to milliseconds
# so the MMMMMMMM contract holds: index / fps * 1000.
count=0
for f in "$OUT"/frame_*.jpg; do
  [[ -e "$f" ]] || continue
  base="$(basename "$f")"
  idx="${base#frame_}"; idx="${idx%.jpg}"
  # strip leading zeros safely
  idx=$((10#$idx))
  ms=$(awk -v i="$idx" -v fps="$FPS" 'BEGIN{ printf "%08d", (i/fps)*1000 }')
  mv "$f" "$OUT/frame_${ms}.jpg"
  count=$((count+1))
done

echo "extract_frames: wrote $count frames to $OUT (fps=$FPS)" >&2
