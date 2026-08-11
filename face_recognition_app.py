"""Face recognition demo: enroll people by name, then recognize them in new
photos. Runs fully locally on the `face_recognition` (dlib) library already
used by tools/face_detect.py in assignment-semantic-search — same
detector/encoder, just matched against named embeddings instead of being
clustered anonymously.

Setup (only needs to be done once):
    pip install face_recognition gradio pillow numpy

Run:
    python face_recognition_app.py
"""

import json
import os

import face_recognition
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

GALLERY_PATH = os.path.join(os.path.dirname(__file__), "known_faces.json")
MATCH_TOLERANCE_DEFAULT = 0.6  # face_recognition's own default cutoff


def _load_gallery() -> list[dict]:
    if not os.path.exists(GALLERY_PATH):
        return []
    with open(GALLERY_PATH) as f:
        return json.load(f)


def _save_gallery(gallery: list[dict]) -> None:
    with open(GALLERY_PATH, "w") as f:
        json.dump(gallery, f)


def _gallery_summary(gallery: list[dict]) -> str:
    if not gallery:
        return "No one enrolled yet."
    counts: dict[str, int] = {}
    for entry in gallery:
        counts[entry["name"]] = counts.get(entry["name"], 0) + 1
    return "Enrolled: " + ", ".join(f"{name} ({n})" for name, n in sorted(counts.items()))


def enroll(image, name):
    gallery = _load_gallery()
    if image is None:
        return "Upload a photo first.", _gallery_summary(gallery)
    name = (name or "").strip()
    if not name:
        return "Enter a name before enrolling.", _gallery_summary(gallery)

    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image)
    locations = face_recognition.face_locations(arr)
    if not locations:
        return "No face found in that photo.", _gallery_summary(gallery)
    if len(locations) > 1:
        return (f"Found {len(locations)} faces — enroll a photo with just {name} in it.",
                _gallery_summary(gallery))

    encoding = face_recognition.face_encodings(arr, locations)[0]
    gallery.append({"name": name, "encoding": encoding.tolist()})
    _save_gallery(gallery)
    return f"Enrolled {name}.", _gallery_summary(gallery)


def reset_gallery():
    _save_gallery([])
    return "Cleared all enrolled faces.", _gallery_summary([])


def recognize(image, tolerance):
    if image is None:
        return None, "Upload a photo first."

    gallery = _load_gallery()
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image)
    locations = face_recognition.face_locations(arr)
    if not locations:
        return image, "No faces detected."

    encodings = face_recognition.face_encodings(arr, locations)
    known_encodings = np.array([e["encoding"] for e in gallery]) if gallery else None

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    lines = []
    for (top, right, bottom, left), enc in zip(locations, encodings):
        label = "Unknown"
        if known_encodings is not None and len(known_encodings):
            distances = np.linalg.norm(known_encodings - enc, axis=1)
            best = int(np.argmin(distances))
            if distances[best] <= tolerance:
                label = f"{gallery[best]['name']} ({distances[best]:.2f})"
        draw.rectangle([left, top, right, bottom], outline="#2ecc71", width=3)
        draw.text((left, max(0, top - 18)), label, fill="#2ecc71")
        lines.append(label)

    return annotated, "\n".join(lines)


with gr.Blocks(title="Face Recognition") as demo:
    gr.Markdown(
        "# Face Recognition\n"
        "Enroll people by name, then recognize them in new photos. Runs fully "
        "locally (no cloud, no API key) using the same `face_recognition` "
        "encoder as `assignment-semantic-search/tools/face_detect.py`."
    )

    with gr.Tab("Enroll"):
        with gr.Row():
            enroll_image = gr.Image(type="pil", label="Photo of one person")
            with gr.Column():
                enroll_name = gr.Textbox(label="Name")
                enroll_button = gr.Button("Enroll")
                enroll_status = gr.Textbox(label="Status", interactive=False)
                gallery_summary = gr.Textbox(
                    label="Known people", interactive=False,
                    value=_gallery_summary(_load_gallery()),
                )
                reset_button = gr.Button("Clear all enrolled faces", variant="stop")
        enroll_button.click(enroll, inputs=[enroll_image, enroll_name],
                             outputs=[enroll_status, gallery_summary])
        reset_button.click(reset_gallery, outputs=[enroll_status, gallery_summary])

    with gr.Tab("Recognize"):
        with gr.Row():
            recognize_image = gr.Image(type="pil", label="Photo to identify")
            with gr.Column():
                tolerance = gr.Slider(0.3, 0.9, value=MATCH_TOLERANCE_DEFAULT, step=0.05,
                                       label="Match tolerance (lower = stricter)")
                recognize_button = gr.Button("Recognize")
                recognize_output = gr.Image(label="Annotated photo")
                recognize_text = gr.Textbox(label="Detections", interactive=False)
        recognize_button.click(recognize, inputs=[recognize_image, tolerance],
                                outputs=[recognize_output, recognize_text])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
