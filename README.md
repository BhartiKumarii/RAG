# RAG_APP

A small collection of Gradio demos for image and document workflows.

## What’s Included

- `gradio_intro.py` - a minimal Gradio greeting demo.
- `image_captioning_model.py` - image captioning with BLIP from Hugging Face.
- `image_classification.py` - OCR-style text extraction from images using a local Gemma 3 model through Ollama.
- `face_recognition_app.py` - enroll people by name from a photo, then recognize them in new photos. Fully local (`face_recognition`/dlib).
- `practice.py` - a PDF question-answering app using LangChain, Chroma, and Ollama.

## Requirements

- Python 3.10 or newer
- `gradio`
- `torch`
- `transformers`
- `pillow`
- `requests`
- `numpy`
- `face_recognition` (needs `dlib` — see note below)
- `langchain-community`
- `langchain-text-splitters`
- `langchain-ollama`
- `chromadb`
- Ollama installed locally for the OCR and PDF QA apps

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install gradio torch transformers pillow requests numpy face_recognition langchain-community langchain-text-splitters langchain-ollama chromadb
```

`face_recognition` depends on `dlib`, which compiles from source on some
platforms and needs CMake and a C++ toolchain installed first. If `pip
install dlib` fails, install your OS's build tools (e.g. `build-essential
cmake` on Debian/Ubuntu) and retry.

If you plan to use the Hugging Face BLIP captioning app, make sure your network access allows model downloads. If you use a Hugging Face token, prefer passing it through an environment variable instead of hardcoding it in the script.

For the local Ollama apps, start Ollama and pull the models first:

```bash
ollama serve
ollama pull gemma3
ollama pull nomic-embed-text
```

## Run the Apps

Launch any script directly with Python:

```bash
python gradio_intro.py
python image_captioning_model.py
python image_classification.py
python face_recognition_app.py
python practice.py
```

Each app opens a Gradio interface in the browser.

## Notes

- `image_classification.py` is actually an OCR-style text extraction demo, not an image classification app.
- `face_recognition_app.py` stores enrolled face embeddings in `known_faces.json` next to the script (created on first enroll, gitignored) — delete it to reset. Each enroll photo must show exactly one face.
- `practice.py` builds a local vector store from the uploaded PDF each time you ask a question.
- The apps bind to `127.0.0.1` and fixed local ports in the code. `face_recognition_app.py` uses `7861` since the others default to `7860`.

## Example Flow

1. Start Ollama if you are using the local-model apps.
2. Run the script you want.
3. Open the Gradio link shown in the terminal.
4. Upload an image or PDF and interact with the app.
