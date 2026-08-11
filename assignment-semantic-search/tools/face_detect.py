#!/usr/bin/env python3
"""Face detection and clustering using face_recognition.

Reads JSON object from stdin: { "framePaths": [...], "outputDir": "..." }
Outputs FaceDetectionSummary JSON to stdout.
Saves face crop images to outputDir.
"""

import json
import os
import re
import sys

import face_recognition
import numpy as np
from PIL import Image


CLUSTER_THRESHOLD = 0.6
FACE_CROP_SIZE = 128
FACE_CROP_PADDING = 0.35  # 35% padding around face


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


def cluster_faces(encodings: list[np.ndarray]) -> list[int]:
    """Assign cluster IDs based on face encoding similarity."""
    if not encodings:
        return []

    clusters: list[list[int]] = []
    centroids: list[np.ndarray] = []
    assignments: list[int] = []

    for i, enc in enumerate(encodings):
        assigned = False
        for ci, centroid in enumerate(centroids):
            distance = np.linalg.norm(enc - centroid)
            if distance < CLUSTER_THRESHOLD:
                clusters[ci].append(i)
                centroids[ci] = (centroid * (len(clusters[ci]) - 1) + enc) / len(clusters[ci])
                assignments.append(ci)
                assigned = True
                break

        if not assigned:
            clusters.append([i])
            centroids.append(enc)
            assignments.append(len(clusters) - 1)

    return assignments


def save_face_crop(image_path: str, location: tuple, output_path: str):
    """Crop and save a face from an image with padding."""
    top, right, bottom, left = location
    img = Image.open(image_path)
    img_w, img_h = img.size

    # Add padding
    face_h = bottom - top
    face_w = right - left
    pad_h = int(face_h * FACE_CROP_PADDING)
    pad_w = int(face_w * FACE_CROP_PADDING)

    crop_top = max(0, top - pad_h)
    crop_bottom = min(img_h, bottom + pad_h)
    crop_left = max(0, left - pad_w)
    crop_right = min(img_w, right + pad_w)

    cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    cropped = cropped.resize((FACE_CROP_SIZE, FACE_CROP_SIZE), Image.LANCZOS)
    cropped.save(output_path, "JPEG", quality=85)


def main():
    # Emit JSON on stdout only; keep any library chatter off that channel.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    input_data = json.loads(sys.stdin.read())
    frame_paths = input_data["framePaths"]
    output_dir = input_data.get("outputDir", "")

    all_faces = []
    all_encodings = []
    all_locations = []
    all_frame_paths = []

    for path in frame_paths:
        frame_index, timestamp = extract_frame_info(path)

        image = face_recognition.load_image_file(path)
        locations = face_recognition.face_locations(image, model="hog")
        encodings = face_recognition.face_encodings(image, locations)

        for loc, enc in zip(locations, encodings):
            top, right, bottom, left = loc
            all_faces.append({
                "frameIndex": frame_index,
                "timestamp": timestamp,
                "confidence": 1.0,
                "bbox": [top, right, bottom, left],
            })
            all_encodings.append(enc)
            all_locations.append(loc)
            all_frame_paths.append(path)

    # Cluster faces
    cluster_ids = cluster_faces(all_encodings)

    # Save individual face crops and build faces output with cluster IDs
    faces_out = []
    frame_detection_counter: dict[int, int] = {}
    for i, face in enumerate(all_faces):
        fi = face["frameIndex"]
        det_idx = frame_detection_counter.get(fi, 0)
        frame_detection_counter[fi] = det_idx + 1

        crop_filename = ""
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            crop_filename = f"face_f{fi}_d{det_idx}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            try:
                save_face_crop(all_frame_paths[i], all_locations[i], crop_path)
            except Exception:
                crop_filename = ""

        faces_out.append({
            **face,
            "clusterId": cluster_ids[i] if i < len(cluster_ids) else 0,
            "cropFile": crop_filename,
        })

    unique_face_count = len(set(cluster_ids)) if cluster_ids else 0

    # Save representative face crop per cluster + build cluster info
    clusters_info = []
    if output_dir and cluster_ids:
        os.makedirs(output_dir, exist_ok=True)
        seen_clusters = set()

        for i, cid in enumerate(cluster_ids):
            if cid in seen_clusters:
                continue
            seen_clusters.add(cid)

            # Count appearances for this cluster
            appearances = sum(1 for c in cluster_ids if c == cid)
            # Get all timestamps for this cluster
            timestamps = [faces_out[j]["timestamp"] for j, c in enumerate(cluster_ids) if c == cid]

            crop_filename = f"face_cluster_{cid}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)

            try:
                save_face_crop(all_frame_paths[i], all_locations[i], crop_path)
            except Exception:
                crop_filename = ""

            clusters_info.append({
                "clusterId": cid,
                "appearances": appearances,
                "cropFile": crop_filename,
                "firstSeen": min(timestamps),
                "lastSeen": max(timestamps),
                "frameIndex": faces_out[i]["frameIndex"],
            })

    summary = {
        "faces": faces_out,
        "clusters": clusters_info,
        "uniqueFaceCount": unique_face_count,
        "totalFaceAppearances": len(faces_out),
    }

    json.dump(summary, real_stdout)


if __name__ == "__main__":
    main()
