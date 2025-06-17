# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# Load environment variables from .env (optional)
load_dotenv()

# Configure Gemini API

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

# Allow CORS (customize for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store extracted chunks globally
pdf_chunks = []

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
    return sorted(chunks, key=lambda x: query.lower() in x.lower(), reverse=True)[:top_k]

# ========== ROUTES ==========

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_chunks

    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are supported."}

    file_path = r"PGS_LLM\PGS_LLM\uploaded_pdfs\dataset.pdf"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    pdf_chunks = extract_pdf_chunks(file_path)
    return {"message": f"Uploaded and processed {file.filename}", "chunks": len(pdf_chunks)}

class Query(BaseModel):
    query: str

@app.post("/rag")
def ask_question(q: Query):
    if not pdf_chunks:
        return {"error": "No PDF uploaded yet. Please upload a PDF first."}

    top_chunks = retrieve_chunks(q.query, pdf_chunks)
    context = "\n\n".join(top_chunks)

    prompt = f"""You are an AI assistant answering based on this context:\n\n{context}\n\nQuestion: {q.query}\nAnswer:"""

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(prompt)]),
    ]
    config = types.GenerateContentConfig(top_p=1, response_mime_type="text/plain")

    response = ""
    for chunk in client.models.generate_content_stream(
        model="gemma-3n-e4b-it", contents=contents, config=config
    ):
        response += chunk.text

    return {"answer": response.strip()}
