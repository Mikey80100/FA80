import os
from fastapi import FastAPI, UploadFile, File, Form
from google.generativeai import GenerativeModel, embed_content
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import fitz  # PyMuPDF

app = FastAPI()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = "models/embedding-001"
LLM_MODEL = GenerativeModel("gemini-2.0-flash-exp")

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
client.recreate_collection("pdfs", vectors_config=VectorParams(size=768, distance=Distance.COSINE))

@app.post("/index_pdf")
async def index_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")
    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        for i in range(0, len(text), 800):
            chunk = text[i:i+800].strip()
            if len(chunk) > 100:
                result = genai.embed_content(EMBED_MODEL, chunk, task_type="RETRIEVAL_DOCUMENT")
                emb = result['embedding']
                point_id = len(chunks)
                chunks.append(PointStruct(id=point_id, vector=emb, payload={
                    "text": chunk, 
                    "page": page_num + 1, 
                    "pdf": file.filename
                }))
    client.upsert("pdfs", points=chunks)
    doc.close()
    return {"status": f"✅ Indexed {len(chunks)} chunks from {file.filename}"}

@app.post("/qa")
async def qa(question: str = Form(...)):
    result = genai.embed_content(EMBED_MODEL, question, task_type="RETRIEVAL_QUERY")
    emb = result['embedding']
    hits = client.search(collection_name="pdfs", query_vector=emb, limit=3)
    context = "\n".join([hit.payload["text"] for hit in hits])
    response = LLM_MODEL.generate_content(
        f"""Réponds précisément à la question en utilisant UNIQUEMENT ces extraits de documents internes :

CONTEXTE (sources PDF) :
{context}

QUESTION : {question}

Format réponse : 
- Réponse claire et concise
- Sources : Fichier page X à la fin"""
    )
    sources = [f"{hit.payload['pdf']} p.{hit.payload['page']}" for hit in hits]
    return {"answer": response.text, "sources": sources}
