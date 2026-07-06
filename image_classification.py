"""
Extract text from an image using your LOCAL Gemma 3 model via Ollama.
Works like OCR — upload a photo of text (a sign, a screenshot, a page
of a book, handwriting, etc.) and Gemma 3 reads it back to you.

Setup (only needs to be done once):
    ollama pull gemma3
    pip install gradio requests pillow

Run:
    python app.py
"""

import base64
import io

import gradio as gr
import requests
from PIL import Image

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3"

# This prompt is the whole trick here — it tells Gemma 3 to act like an
# OCR tool instead of describing the image or summarizing it.
OCR_PROMPT = (
    "Extract ALL text visible in this image, exactly as it appears. "
    "Preserve line breaks and formatting where possible. "
    "Do not describe the image, do not summarize, do not add commentary — "
    "output ONLY the text you can read. "
    "If there is no readable text in the image, say 'No text found.'"
)


def image_to_base64(image: Image.Image) -> str:
    """Ollama's API expects images as base64-encoded strings, not raw files."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_text(image):
    if image is None:
        return "No image provided."

    if image.mode != "RGB":
        image = image.convert("RGB")

    encoded_image = image_to_base64(image)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": OCR_PROMPT,
                "images": [encoded_image],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        return ("Couldn't reach Ollama. Is it running? Try: ollama serve\n"
                "Also make sure you've pulled the model: ollama pull gemma3")
    except Exception as e:
        return f"An error occurred: {str(e)}"


iface = gr.Interface(
    fn=extract_text,
    inputs=gr.Image(type="pil", label="Upload an image with text in it"),
    outputs=gr.Textbox(label="Extracted text", lines=15),
    title="OCR with Local Gemma 3 (via Ollama)",
    description=(
        "Upload a photo of a sign, screenshot, page, or handwriting, "
        "and Gemma 3 will read the text back to you. Runs fully locally."
    ),
)

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860)
