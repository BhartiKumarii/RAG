"""Load a corpus folder into the segments table.

GIVEN. This parses the diarized transcript and the slide-text (OCR) output the
toolkit produced, splits the transcript into segments, attaches the on-screen
text that overlaps each segment in time, and inserts the rows (with a NULL
embedding, which embed.py fills later).

  python load_corpus.py ../corpus/day1 day1
  python load_corpus.py ../corpus/day2 day2

The ONE part you may want to edit is chunk_segments() — how transcript cues are
grouped into segments is your Day 9 lever. Everything else is plumbing.
"""
import glob
import json
import os
import re
import sys

from db import get_conn


def parse_srt(path: str) -> list[dict]:
    """Return cues as [{start, end, speaker, text}] from a .diarized.srt file."""
    blocks = re.split(r"\n\s*\n", open(path, encoding="utf-8", errors="replace").read().strip())
    cues = []
    for b in blocks:
        lines = b.strip().split("\n")
        if len(lines) < 2:
            continue
        m = re.search(
            r"(\d\d):(\d\d):(\d\d),(\d+)\s*-->\s*(\d\d):(\d\d):(\d\d),(\d+)", b
        )
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        text = " ".join(lines[2:]) if len(lines) > 2 else ""
        spk = None
        sm = re.match(r"\s*\[(SPEAKER_\d+)\]\s*", text)
        if sm:
            spk = sm.group(1)
            text = text[sm.end():]
        cues.append({"start": start, "end": end, "speaker": spk, "text": text.strip()})
    return cues


def chunk_segments(cues: list[dict]) -> list[dict]:
    """Group transcript cues into retrieval segments.

    DEFAULT: one cue = one segment. This is deliberately simple. Your Day 9
    lever is to change this — for example, merge consecutive cues from the same
    speaker until a segment reaches a target length, so each chunk holds one
    idea instead of one breath. Keep start = first cue's start so the hit still
    points at the right moment.
    """
    return cues


def find_srt(folder: str) -> str:
    hits = glob.glob(os.path.join(folder, "*.diarized.srt"))
    if not hits:
        raise SystemExit(f"no *.diarized.srt in {folder} (run the toolkit first)")
    return hits[0]


def load_ocr(folder: str) -> list[dict]:
    path = os.path.join(folder, "ocr.json")
    if not os.path.isfile(path):
        return []
    return json.load(open(path))


def ocr_for_span(ocr: list[dict], start: float, end: float) -> str:
    """All OCR text whose timestamp falls within [start, end], joined."""
    parts = [e["text"] for e in ocr if start <= e.get("timestamp", -1) <= end]
    return "\n".join(parts).strip() or None


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python load_corpus.py <corpus-folder> <video-id>")
    folder, video = sys.argv[1], sys.argv[2]

    cues = parse_srt(find_srt(folder))
    segments = chunk_segments(cues)
    ocr = load_ocr(folder)

    conn = get_conn()
    conn.execute("DELETE FROM segments WHERE video = %s", (video,))
    rows = 0
    for i, seg in enumerate(segments):
        if not seg["text"]:
            continue
        seg_id = f"{video}_{i:04d}"
        conn.execute(
            """INSERT INTO segments (id, video, seq, start_s, end_s, speaker, text, ocr_text)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                seg_id, video, i, seg["start"], seg["end"], seg.get("speaker"),
                seg["text"], ocr_for_span(ocr, seg["start"], seg["end"]),
            ),
        )
        rows += 1
    print(f"loaded {rows} segments for '{video}' ({len(ocr)} OCR entries available)")


if __name__ == "__main__":
    main()
