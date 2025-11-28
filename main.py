from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import numpy as np
import json
import os

# -----------------------------
# LOAD ENVIRONMENT
# -----------------------------
load_dotenv()

MASTER_API_KEY = os.getenv("MASTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://127.0.0.1:5500,http://localhost:5500"
).split(",")

if not OPENAI_API_KEY:
    print("❌ ERROR: MISSING OPENAI API KEY IN .env")
if not MASTER_API_KEY:
    print("⚠️ WARNING: MASTER_API_KEY NOT SET — API IS NOT SECURE")

# -----------------------------
# FASTAPI INIT
# -----------------------------
app = FastAPI()

# Secure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# DB SETUP
# -----------------------------
from database.database import SessionLocal
from database.models import Memory

def get_db():
    """Thread-safe DB session handling."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------
# OPENAI CLIENT
# -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# REQUEST MODEL
# -----------------------------
class Message(BaseModel):
    business_id: str
    user: str
    message: str

# -----------------------------
# API KEY VALIDATION
# -----------------------------
def verify_api_key(x_api_key: str | None):
    if not MASTER_API_KEY:
        return  # dev mode
    if x_api_key != MASTER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# -----------------------------
# LOAD BUSINESS PROFILE
# -----------------------------
def load_business_profile(business_id: str):
    path = f"business/{business_id}.json"
    if not os.path.exists(path):
        return {"system_prompt": "You are a helpful assistant."}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------
# MEMORY FUNCTIONS
# -----------------------------
def load_memory(db: Session, user: str, business_id: str):
    rows = (
        db.query(Memory)
        .filter(Memory.user == user, Memory.business_id == business_id)
        .order_by(Memory.timestamp.asc())
        .all()
    )
    return [{"role": r.role, "content": r.content} for r in rows]

def save_memory(db: Session, user: str, business_id: str, role: str, content: str):
    entry = Memory(user=user, business_id=business_id, role=role, content=content)
    db.add(entry)
    db.commit()

# -----------------------------
# KNOWLEDGE BASE CACHE (RAG)
# -----------------------------
kb_cache = {}

def load_kb(business_id: str):
    if business_id in kb_cache:
        return kb_cache[business_id]

    path = f"knowledge/indexes/{business_id}_kb.json"
    if not os.path.exists(path):
        kb_cache[business_id] = []
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["embedding"] = np.array(item["embedding"], dtype=np.float32)

    kb_cache[business_id] = data
    return data

def embed_text(text: str):
    try:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
        return np.array(emb, dtype=np.float32)
    except Exception as e:
        print("❌ EMBEDDING ERROR:", e)
        return None

def rag_retrieve(business_id: str, query: str, top_k=4):
    kb = load_kb(business_id)
    if not kb:
        return []

    q_emb = embed_text(query)
    if q_emb is None:
        return []

    scored = []
    for item in kb:
        sim = float(np.dot(q_emb, item["embedding"]) /
                    (np.linalg.norm(q_emb) * np.linalg.norm(item["embedding"]) + 1e-8))
        scored.append((sim, item["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:top_k]]

# -----------------------------
# CHAT ENDPOINT
# -----------------------------
@app.post("/chat")
async def chat(
    data: Message,
    db: Session = Depends(get_db),
    x_api_key: str = Header(None)
):

    verify_api_key(x_api_key)

    profile = load_business_profile(data.business_id)
    system_prompt = profile.get("system_prompt", "You are a helpful assistant.")

    memory = load_memory(db, data.user, data.business_id)

    # Retrieve relevant knowledge pieces
    chunks = rag_retrieve(data.business_id, data.message)
    kb_context = "\n".join(chunks) if chunks else ""

    final_system_prompt = system_prompt
    if kb_context:
        final_system_prompt += "\nRelevant Business Info:\n" + kb_context

    messages = [{"role": "system", "content": final_system_prompt}]
    messages.extend(memory)
    messages.append({"role": "user", "content": data.message})

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
        )
        reply = completion.choices[0].message.content

    except Exception as e:
        print("❌ GPT ERROR:", e)
        raise HTTPException(status_code=500, detail="LLM processing error")

    save_memory(db, data.user, data.business_id, "user", data.message)
    save_memory(db, data.user, data.business_id, "assistant", reply)

    return {"reply": reply}
