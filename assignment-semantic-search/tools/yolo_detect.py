#!/usr/bin/env python3
"""Object detection using YOLOv8n.

Reads JSON object from stdin with keys:
  - framePaths: list of frame image paths
  - outputDir:  directory to save individual object crops

Outputs ObjectDetectionSummary JSON to stdout.
"""

import json
import os
import re
import sys
from collections import defaultdict

from PIL import Image
from ultralytics import YOLO

CROP_SIZE = 128
MAX_CROPS_PER_LABEL = 8


def extract_frame_info(path: str) -> tuple[int, float]:
    """Extract frame index and timestamp from path.

    New format: frame_MMMMMMMM.jpg (8+ digit milliseconds)
    Old format: frame_NNNN.jpg (4-digit sequential index, 1fps assumed)

    Returns (frame_index, timestamp_seconds).
    """
    match = re.search(r"frame_(\d+)", path)
    if not match:
        return (0, 0.0)
    digits = match.group(1)
    num = int(digits)
    if len(digits) >= 7:
        return (num, num / 1000.0)
    else:
        return (num, float(num))


def save_object_crops(
    detections: list[dict],
    frame_paths_set: set[str],
    output_dir: str | None,
) -> list[dict]:
    """Crop detected objects and save as individual images.

    Returns a list of {label, cropFile} for each saved crop.
    """
    if not output_dir:
        return []

    # Group detections by label
    by_label: dict[str, list[dict]] = defaultdict(list)
    for det in detections:
        by_label[det["label"]].append(det)

    # Build a map from frameIndex -> frame path
    frame_map: dict[int, str] = {}
    for p in frame_paths_set:
        frame_map[extract_frame_info(p)[0]] = p

    crops: list[dict] = []
    crop_index = 0

    for label, dets in by_label.items():
        # Top N by confidence per label
        dets_sorted = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        dets_sorted = dets_sorted[:MAX_CROPS_PER_LABEL]

        for det in dets_sorted:
            frame_path = frame_map.get(det["frameIndex"])
            if not frame_path or not os.path.isfile(frame_path):
                continue
            try:
                img = Image.open(frame_path)
                x1, y1, x2, y2 = det["bbox"]
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(img.width, int(x2))
                y2 = min(img.height, int(y2))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                crop = img.crop((x1, y1, x2, y2))
                crop = crop.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS)

                filename = f"obj_crop_{crop_index:04d}.jpg"
                crop.save(os.path.join(output_dir, filename), quality=85)

                crops.append({"label": label, "cropFile": filename, "timestamp": det["timestamp"]})
                crop_index += 1
            except Exception:
                continue

    return crops


def main():
    # We emit JSON on stdout. ultralytics prints a model-download progress bar
    # (and other chatter) to stdout on first run, which would corrupt that JSON.
    # Keep the real stdout aside and send all library output to stderr; only the
    # final summary is written back to the real stdout.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    payload = json.loads(sys.stdin.read())

    # Support both old (bare array) and new (object with framePaths) formats
    if isinstance(payload, list):
        frame_paths = payload
        output_dir = None
    else:
        frame_paths = payload["framePaths"]
        output_dir = payload.get("outputDir")

    model = YOLO("yolov8n.pt")

    detections: list[dict] = []
    object_freq: dict[str, int] = {}

    for path in frame_paths:
        frame_index, timestamp = extract_frame_info(path)

        results = model(path, verbose=False)

        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                if confidence < 0.3:
                    continue

                detections.append({
                    "frameIndex": frame_index,
                    "timestamp": timestamp,
                    "label": label,
                    "confidence": round(confidence, 3),
                    "bbox": [round(x, 1) for x in bbox],
                })

                object_freq[label] = object_freq.get(label, 0) + 1

    unique_objects = sorted(object_freq.keys())

    # Save individual crop images
    crops = save_object_crops(detections, set(frame_paths), output_dir)

    summary = {
        "detections": detections,
        "uniqueObjects": unique_objects,
        "objectFrequency": object_freq,
        "personCount": {"max": 0, "averagePerScene": 0},
    }

    if crops:
        summary["objectCrops"] = crops

    json.dump(summary, real_stdout)


if __name__ == "__main__":
    main()
