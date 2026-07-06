# RAG_APP

A small collection of Gradio demos for image and document workflows.

## What’s Included

- `gradio_intro.py` - a minimal Gradio greeting demo.
- `image_captioning_model.py` - image captioning with BLIP from Hugging Face.
- `image_classification.py` - OCR-style text extraction from images using a local Gemma 3 model through Ollama.
- `practice.py` - a PDF question-answering app using LangChain, Chroma, and Ollama.

## Requirements

- Python 3.10 or newer
- `gradio`
- `torch`
- `transformers`
- `pillow`
- `requests`
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
pip install gradio torch transformers pillow requests langchain-community langchain-text-splitters langchain-ollama chromadb
```

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
python practice.py
```

Each app opens a Gradio interface in the browser.

## Notes

- `image_classification.py` is actually an OCR-style text extraction demo, not an image classification app.
- `practice.py` builds a local vector store from the uploaded PDF each time you ask a question.
- The apps bind to `127.0.0.1` and fixed local ports in the code.

## Example Flow

1. Start Ollama if you are using the local-model apps.
2. Run the script you want.
3. Open the Gradio link shown in the terminal.
4. Upload an image or PDF and interact with the app.
