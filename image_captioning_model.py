# BLIP (Bootstrapped Language-Image Pretraining) model can generate captions for images
import os
import torch
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Optionally use a Hugging Face token to avoid anonymous rate limits.
HF_TOKEN = os.getenv("HF_TOKEN", "")
auth_args = {"token": HF_TOKEN} if HF_TOKEN else {}

try:
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base", **auth_args
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", **auth_args
    )
    model.eval()  # inference mode: disables dropout etc.
except Exception as e:
    raise RuntimeError(
        "Failed to load model. If you see a warning about unauthenticated requests, "
        "set HF_TOKEN or run `huggingface-cli login`. "
        f"Original error: {e}"
    )

# Use GPU if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_caption(image):
    """Generate a caption for a PIL Image and return it as a string."""
    if image is None:
        raise ValueError("No image provided")

    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption


def caption_image(image):
    """
    Takes a PIL Image input and returns a caption.
    """
    try:
        caption = generate_caption(image)
        return caption
    except Exception as e:
        return f"An error occurred: {str(e)}"


iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Captioning with BLIP",
    description="Upload an image to generate a caption.",
)

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860)
# BLIP (Bootstrapped Language-Image Pretraining) model can generate captions for images
import os
import torch
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Optionally use a Hugging Face token to avoid anonymous rate limits.
HF_TOKEN = os.getenv("HF_TOKEN", "")
auth_args = {"token": HF_TOKEN} if HF_TOKEN else {}

try:
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base", **auth_args
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", **auth_args
    )
    model.eval()  # inference mode: disables dropout etc.
except Exception as e:
    raise RuntimeError(
        "Failed to load model. If you see a warning about unauthenticated requests, "
        "set HF_TOKEN or run `huggingface-cli login`. "
        f"Original error: {e}"
    )

# Use GPU if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def generate_caption(image):
    """Generate a caption for a PIL Image and return it as a string."""
    if image is None:
        raise ValueError("No image provided")

    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption


def caption_image(image):
    """
    Takes a PIL Image input and returns a caption.
    """
    try:
        caption = generate_caption(image)
        return caption
    except Exception as e:
        return f"An error occurred: {str(e)}"


iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Captioning with BLIP",
    description="Upload an image to generate a caption.",
)

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860)