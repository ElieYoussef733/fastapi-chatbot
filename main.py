from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import numpy as np
import json
import os

# -----------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# -----------------------------------------------------
load_dotenv()

MASTER_API_KEY = os.getenv("MASTER_API_KEY")
PUBLIC_DEMO_KEY = os.getenv("PUBLIC_DEMO_KEY")     # <-- Streamlit demo key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "*"
).split(",")

if not OPENAI_API_KEY:
    print("❌ ERROR: Missing OPENAI_API_KEY in .env")
if not MASTER_API_KEY:
    print("⚠️ WARNING: MASTER_API_KEY is missing (dev mode)")

# -----------------------------------------------------
# FASTAPI APP INITIALIZATION
# -----------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # allow Streamlit & widget access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# DATABASE SETUP
# -----------------------------------------------------
from database.database import SessionLocal
from database.models import Memory

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------------------------------
# OPENAI CLIENT
# -----------------------------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------
# REQUEST MODEL
# -----------------------------------------------------
class Message(BaseModel):
    business_id: str
    user: str
    message: str

# -----------------------------------------------------
# API KEY VALIDATION (supports MASTER & PUBLIC_DEMO)
# -----------------------------------------------------
def verify_api_key(x_api_key: str | None):
    # Dev mode (no master key defined)
    if not MASTER_API_KEY:
        return

    # Accept public demo key
    if PUBLIC_DEMO_KEY and x_api_key == PUBLIC_DEMO_KEY:
        return

    # Accept master key
    if x_api_key == MASTER_API_KEY:
        return

    # Reject everything else
    raise HTTPException(status_code=401, detail="Invalid or missing API key")

# -----------------------------------------------------
# LOAD BUSINESS PROFILE
# -----------------------------------------------------
def load_business_profile(business_id: str):
    path = f"business/{business_id}.json"
    if not os.path.exists(path):
        return {"system_prompt": "You are a helpful assistant."}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------------------------------
# MEMORY FUNCTIONS
# -----------------------------------------------------
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

# -----------------------------------------------------
# KNOWLEDGE BASE (RAG)
# -----------------------------------------------------
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

    # Convert embeddings to numpy
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
        print("❌ Embedding error:", e)
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

        # Filter low-quality matches
        if sim < 0.55:
            continue

        scored.append((sim, item["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:top_k]]

# -----------------------------------------------------
# CHAT ENDPOINT
# -----------------------------------------------------
@app.post("/chat")
async def chat(
    data: Message,
    db: Session = Depends(get_db),
    x_api_key: str = Header(None)
):

    verify_api_key(x_api_key)

    # Load business settings
    profile = load_business_profile(data.business_id)
    system_prompt = profile.get("system_prompt", "You are a helpful assistant.")

    # Load memory from DB
    memory = load_memory(db, data.user, data.business_id)

    # Retrieve relevant knowledge
    chunks = rag_retrieve(data.business_id, data.message)
    kb_context = "\n".join(chunks) if chunks else ""

    # Build final system prompt
    final_system_prompt = system_prompt
    if kb_context:
        final_system_prompt += "\nRelevant Business Info:\n" + kb_context

    # Build messages array
    messages = [{"role": "system", "content": final_system_prompt}]
    messages.extend(memory)
    messages.append({"role": "user", "content": data.message})

    # Call GPT
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
        )
        reply = completion.choices[0].message.content
    except Exception as e:
        print("❌ OpenAI Chat error:", e)
        raise HTTPException(status_code=500, detail="AI generation error")

    # Save memory
    save_memory(db, data.user, data.business_id, "user", data.message)
    save_memory(db, data.user, data.business_id, "assistant", reply)

    return {"reply": reply}
