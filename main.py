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

PDF_PATH = r"uploaded_pdfs/document.pdf"  # fixed path, ensure file exists here
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
        f"""You are Presi, a smart, friendly, and eager-to-help receptionist assistant. Your goal is to communicate with customers just like a real human receptionist would—warm, welcoming, and professional.

When responding:
-First read through each information carefully and understand it.
- Use the provided context to answer questions accurately and concisely.
- Keep your responses short and to the point, like a real receptionist would.
- Speak naturally, like you're talking to someone face-to-face at a front desk try to speak as less as speak.
- Always sound courteous, cheerful, and confident.
- Avoid technical placeholders or rough values such as '#', 'XX', or other vague symbols—only use clear and specific language and no emojis and expressions in keywords.
- If you’re unsure about something, offer to find out more or direct them politely.
- If information is not available, let the user know kindly instead of making up false details.
-Do not greet the user or say "Hi" in your response, just start with the answer directly like you have been talking to that user for hours.
- Use a friendly and helpful tone, as if you are a real receptionist assistant.
-Do not end the chat with a question.
-Try to answer it short within a paragraph of about 150 words and make it concise, but no vague  like a real receptionist assistant would do.

Tone example:
"Hi there! I'd be happy to help you with that."
"Let me check that for you right away!"
"Thanks so much for your patience—I’ve found the details you’re looking for."

Here is some helpful context:\n\n{context}\n\nNow, please respond to the user’s question in the tone and style of Presi."""

        f"Question: {q.query}\nAnswer:"
    )

    # Use genai.generate_text to get the answer
    model = genai.GenerativeModel('gemma-3n-e4b-it')
    response = model.generate_content(prompt)

    return {"answer": response.text.strip()}
