from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

import gradio as gr
import warnings

warnings.filterwarnings("ignore")


def get_llm():
    return OllamaLLM(model="gemma3")


def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")


def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)


def create_vector_db(chunks):
    embeddings = get_embeddings()
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


def create_retriever(file_path):
    documents = document_loader(file_path)
    chunks = split_text(documents)
    vector_db = create_vector_db(chunks)
    return vector_db


def ask_question(file_path, question):
    llm = get_llm()
    vector_db = create_retriever(file_path)

    relevant_documents = vector_db.similarity_search(question, k=4)
    context = "\n\n".join(document.page_content for document in relevant_documents)

    prompt = (
        "You are a helpful assistant answering questions from the provided PDF context. "
        "If the answer is not present in the context, say you could not find it in the document.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else result


app = gr.Interface(
    fn=ask_question,
    inputs=[
        gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath"),
        gr.Textbox(label="Ask a Question", lines=2, placeholder="Type your question..."),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="RAG Chatbot using Gemma 3 + Ollama",
    description="Upload a PDF and ask questions about it.",
)

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7862)
