#!/usr/bin/env python3
"""On-screen text (OCR) using Tesseract.

Reads a JSON object from stdin:
  { "framePaths": ["frames/frame_00001000.jpg", ...] }

Writes a JSON array of on-screen text entries to stdout:
  [ { "frameIndex": 1000, "timestamp": 1.0, "text": "..." }, ... ]

On a screen-recorded training session this pulls the slide text that is
visible in each frame, which is often the single most useful signal for
search. Every fifth frame is sampled to keep it tractable.
"""

import json
import re
import subprocess
import sys

# Sample every Nth frame so OCR stays fast on long recordings.
SAMPLE_INTERVAL = 5

# Tesseract language packs. eng only by default; set TESSERACT_LANGS=eng+ara
# (etc.) if a session mixes scripts.
import os
TESSERACT_LANGS = os.environ.get("TESSERACT_LANGS", "eng")
TESSERACT_BIN = os.environ.get("TESSERACT_BIN", "tesseract")


def frame_info(path: str) -> tuple[int, float]:
    """(frame_index, timestamp_seconds) from a frame_MMMMMMMM.jpg name.

    8+ digits are milliseconds; shorter is a 1-fps sequential index.
    """
    m = re.search(r"frame_(\d+)", path)
    if not m:
        return (0, 0.0)
    digits = m.group(1)
    num = int(digits)
    return (num, num / 1000.0) if len(digits) >= 7 else (num, float(num))


def main() -> None:
    payload = json.loads(sys.stdin.read())
    frame_paths = payload if isinstance(payload, list) else payload["framePaths"]
    sampled = frame_paths[::SAMPLE_INTERVAL]

    print(
        f"OCR: {len(sampled)} of {len(frame_paths)} frames "
        f"(langs={TESSERACT_LANGS})",
        file=sys.stderr,
    )

    entries: list[dict] = []
    failures = 0
    for path in sampled:
        try:
            out = subprocess.run(
                [TESSERACT_BIN, path, "stdout", "-l", TESSERACT_LANGS, "--psm", "3"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            text = out.stdout.strip()
            if text:
                idx, ts = frame_info(path)
                entries.append({"frameIndex": idx, "timestamp": ts, "text": text})
        except Exception as err:  # noqa: BLE001 - best-effort per frame
            failures += 1
            if failures == 1:
                print(f"OCR: first failure on {path}: {err}", file=sys.stderr)

    print(f"OCR: {len(entries)} entries, {failures} failures", file=sys.stderr)
    json.dump(entries, sys.stdout)


if __name__ == "__main__":
    main()
