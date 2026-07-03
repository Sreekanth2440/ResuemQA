import os
import gradio as gr

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# ----------------------------------------------------
# Configure OpenAI
# ----------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Please set it as an environment variable."
    )

Settings.llm = OpenAI(
    model="gpt-3.5-turbo",
    api_key=OPENAI_API_KEY,
)

Settings.embed_model = OpenAIEmbedding(
    api_key=OPENAI_API_KEY
)

# ----------------------------------------------------
# Global Variables
# ----------------------------------------------------

query_engine = None

DATA_DIR = "./data"
PERSIST_DIR = "./storage"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# ----------------------------------------------------
# Build Index
# ----------------------------------------------------


def process_resume(file_obj):
    global query_engine

    if file_obj is None:
        return "⚠️ Please upload a PDF resume."

    try:
        resume_path = os.path.join(DATA_DIR, "resume.pdf")

        # Save uploaded file
        with open(resume_path, "wb") as f:
            f.write(file_obj)

        # Load document
        documents = SimpleDirectoryReader(
            input_dir=DATA_DIR,
            required_exts=[".pdf"]
        ).load_data()

        # Create vector index
        index = VectorStoreIndex.from_documents(documents)

        # Save index
        index.storage_context.persist(persist_dir=PERSIST_DIR)

        # Create query engine
        query_engine = index.as_query_engine()

        return "✅ Resume indexed successfully! You can now ask questions."

    except Exception as e:
        return f"❌ Error: {e}"


# ----------------------------------------------------
# Chat Function
# ----------------------------------------------------


def chat_response(message, history):
    global query_engine

    if query_engine is None:

        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):

            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=PERSIST_DIR
                )

                index = load_index_from_storage(storage_context)

                query_engine = index.as_query_engine()

            except Exception:
                return "Please upload a resume first."

        else:
            return "Please upload a resume first."

    try:
        response = query_engine.query(message)
        return str(response)

    except Exception as e:
        return f"Error: {e}"


# ----------------------------------------------------
# UI
# ----------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 📄 Resume Q&A Chatbot")
    gr.Markdown(
        "Upload your resume (PDF) and ask questions about it."
    )

    with gr.Row():

        with gr.Column(scale=1):

            file_input = gr.File(
                label="Upload Resume",
                file_types=[".pdf"],
                type="binary",
            )

            upload_btn = gr.Button(
                "Build AI Index",
                variant="primary",
            )

            status = gr.Textbox(
                label="Status",
                interactive=False,
            )

            upload_btn.click(
                fn=process_resume,
                inputs=file_input,
                outputs=status,
            )

        with gr.Column(scale=3):

            gr.ChatInterface(
                fn=chat_response,
                examples=[
                    "Summarize this resume.",
                    "What technical skills does the candidate have?",
                    "What projects has the candidate worked on?",
                    "Does the candidate know Python?",
                    "What is the educational background?"
                ],
            )

# ----------------------------------------------------

if __name__ == "__main__":
    demo.launch()