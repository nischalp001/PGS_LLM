# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables from .env (optional)
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

# Allow CORS (customize origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== UTILITIES ==========

def extract_pdf_chunks(file_path: str, chunk_size=500, overlap=50):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def retrieve_chunks(query: str, chunks, top_k=3):
    # Simple keyword matching, improve as needed
    return sorted(chunks, key=lambda x: query.lower() in x.lower(), reverse=True)[:top_k]

# ========== GLOBALS ==========

PDF_PATH = "uploaded_pdfs\Dcumentation_PGS.pdf"  # fixed path, ensure file exists here
pdf_chunks = []

# Load and process PDF on startup
@app.on_event("startup")
def load_pdf_on_startup():
    global pdf_chunks
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF file not found at {PDF_PATH}. Please add it before starting.")
    pdf_chunks = extract_pdf_chunks(PDF_PATH)
    print(f"Loaded PDF and extracted {len(pdf_chunks)} chunks.")

# ========== REQUEST MODELS ==========

class Query(BaseModel):
    query: str

# ========== ROUTES ==========

@app.post("/rag")
def ask_question(q: Query):
    global pdf_chunks

    if not pdf_chunks:
        return {"error": "PDF data not loaded. Please check server logs."}

    top_chunks = retrieve_chunks(q.query, pdf_chunks)
    context = "\n\n".join(top_chunks)

    prompt = (
        f"You are an AI assistant talking as a human being tone answering based on this context and explain no more than 400 words, But never ever mention that you are acting like human and your mission is to explain words under 200.:\n\n{context}\n\n"
        f"Question: {q.query}\nAnswer:"
    )

    # Use genai.generate_text to get the answer
    model = genai.GenerativeModel('gemma-3n-e4b-it')
    response = model.generate_content(prompt)

    return {"answer": response.text.strip()}
