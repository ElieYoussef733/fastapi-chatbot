import os
import json
import re
import time
import logging
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, OpenAIError
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.database import SessionLocal
from database.models import Memory

# -----------------------------------------------------
# ENV & LOGGING
# -----------------------------------------------------
load_dotenv()

logger = logging.getLogger("chatbot")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

MASTER_API_KEY = os.getenv("MASTER_API_KEY")
PUBLIC_DEMO_KEY = os.getenv("PUBLIC_DEMO_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

if not OPENAI_API_KEY:
    logger.error("❌ MISSING OPENAI_API_KEY")
if not MASTER_API_KEY:
    logger.warning("⚠️ MASTER_API_KEY not set – API is in dev mode")

# Per-business API keys (optionally set via env)
BUSINESS_API_KEYS: Dict[str, str] = {
    "phone_store_ali": os.getenv("API_KEY_PHONE_STORE_ALI", ""),
    "restaurant_mario": os.getenv("API_KEY_RESTAURANT_MARIO", ""),
    "gym_180fitness": os.getenv("API_KEY_GYM_180FITNESS", ""),
}
# Remove empties
BUSINESS_API_KEYS = {k: v for k, v in BUSINESS_API_KEYS.items() if v}

# Limits / tuning
MAX_MESSAGE_CHARS = 1500
MAX_HISTORY_MESSAGES = 20  # per user/business kept in prompt + DB
OPENAI_MAX_RETRIES = 3
OPENAI_RETRY_DELAY = 2  # seconds

# Simple in-memory rate limiting (per process)
# key -> deque[timestamps]
rate_store: Dict[str, Deque[float]] = defaultdict(deque)

# Windows & limits
RATE_LIMITS = {
    "ip_per_minute": (60.0, 100),      # 100 req / minute / IP
    "user_per_minute": (60.0, 80),     # 80 req / minute / user
    "business_per_minute": (60.0, 300) # 300 req / minute / business
}

# -----------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------
# SCHEMAS
# -----------------------------------------------------
class Message(BaseModel):
    business_id: str
    user: str
    message: str

# -----------------------------------------------------
# DB
# -----------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------------------------------
# HEALTH CHECK (for Render)
# -----------------------------------------------------
@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------------------------------
# SECURITY HELPERS
# -----------------------------------------------------
def _rate_limit_key(scope: str, value: str) -> str:
    return f"{scope}:{value}"


def enforce_rate_limit(scope: str, value: str):
    """Simple sliding-window rate limiter (in-memory, per process)."""
    if scope not in RATE_LIMITS:
        return

    window, max_calls = RATE_LIMITS[scope]
    key = _rate_limit_key(scope, value)
    now = time.time()
    dq = rate_store[key]

    # Drop old timestamps
    while dq and (now - dq[0]) > window:
        dq.popleft()

    if len(dq) >= max_calls:
        logger.warning(f"Rate limit exceeded: {scope}={value}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for {scope}"
        )

    dq.append(now)


def sanitize_message(text: str) -> str:
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="Message must be text")

    # Remove null bytes and control chars (except basic whitespace)
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F]", " ", text)

    # Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) == 0:
        raise HTTPException(status_code=400, detail="Empty message")

    if len(text) > MAX_MESSAGE_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Message too long (max {MAX_MESSAGE_CHARS} characters)"
        )

    return text


def verify_api_key(x_api_key: str | None, business_id: str):
    """
    Auth logic:
    - MASTER_API_KEY → full access to all businesses
    - PUBLIC_DEMO_KEY → demo / public frontend
    - BUSINESS_API_KEYS[business] → restricted to that business only
    """
    # Dev mode: no master key set
    if not MASTER_API_KEY:
        return

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    # Master
    if x_api_key == MASTER_API_KEY:
        return

    # Public demo
    if PUBLIC_DEMO_KEY and x_api_key == PUBLIC_DEMO_KEY:
        return

    # Per-business keys
    for biz_id, key in BUSINESS_API_KEYS.items():
        if key and x_api_key == key:
            if biz_id != business_id:
                logger.warning(
                    f"API key for {biz_id} tried to access {business_id}"
                )
                raise HTTPException(
                    status_code=403,
                    detail="API key not allowed for this business"
                )
            return

    raise HTTPException(status_code=401, detail="Invalid API key")


# -----------------------------------------------------
# BUSINESS PROFILE & MEMORY
# -----------------------------------------------------
def load_business_profile(business_id: str):
    path = f"business/{business_id}.json"
    if not os.path.exists(path):
        logger.warning(f"Business profile not found: {business_id}")
        return {"system_prompt": "You are a helpful assistant."}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_memory(db: Session, user: str, business_id: str):
    rows = (
        db.query(Memory)
        .filter(Memory.user == user, Memory.business_id == business_id)
        .order_by(Memory.timestamp.asc())
        .all()
    )
    # Keep only last MAX_HISTORY_MESSAGES in the prompt
    trimmed = rows[-MAX_HISTORY_MESSAGES:]
    return [{"role": r.role, "content": r.content} for r in trimmed]


def save_memory(db: Session, user: str, business_id: str, role: str, content: str):
    entry = Memory(user=user, business_id=business_id, role=role, content=content)
    db.add(entry)
    db.commit()
    prune_old_memory(db, user, business_id)


def prune_old_memory(db: Session, user: str, business_id: str):
    """Delete oldest messages if more than MAX_HISTORY_MESSAGES*2 exist."""
    from sqlalchemy import func

    total = (
        db.query(func.count(Memory.id))
        .filter(Memory.user == user, Memory.business_id == business_id)
        .scalar()
    )
    limit = MAX_HISTORY_MESSAGES * 2
    if total and total > limit:
        to_delete = total - limit
        subq = (
            db.query(Memory.id)
            .filter(Memory.user == user, Memory.business_id == business_id)
            .order_by(Memory.timestamp.asc())
            .limit(to_delete)
            .subquery()
        )
        db.query(Memory).filter(Memory.id.in_(subq)).delete(
            synchronize_session=False
        )
        db.commit()
        logger.info(
            f"Pruned {to_delete} old memory rows for user={user}, business={business_id}"
        )

# -----------------------------------------------------
# KNOWLEDGE BASE (RAG)
# -----------------------------------------------------
kb_cache: Dict[str, list] = {}

def load_kb(business_id: str):
    if business_id in kb_cache:
        return kb_cache[business_id]

    path = f"knowledge/indexes/{business_id}_kb.json"
    if not os.path.exists(path):
        logger.info(f"No KB index for business: {business_id}")
        kb_cache[business_id] = []
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["embedding"] = np.array(item["embedding"], dtype=np.float32)

    kb_cache[business_id] = data
    return data


def embed_text(text: str):
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            emb = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            ).data[0].embedding
            return np.array(emb, dtype=np.float32)
        except OpenAIError as e:
            logger.error(f"Embedding error (attempt {attempt+1}): {e}")
            time.sleep(OPENAI_RETRY_DELAY)
        except Exception as e:
            logger.error(f"Unexpected embedding error: {e}")
            break
    return None


def rag_retrieve(business_id: str, query: str, top_k: int = 4):
    kb = load_kb(business_id)
    if not kb:
        return []

    q_emb = embed_text(query)
    if q_emb is None:
        return []

    scored: list[Tuple[float, str]] = []
    for item in kb:
        v = item["embedding"]
        sim = float(np.dot(q_emb, v) /
                    (np.linalg.norm(q_emb) * np.linalg.norm(v) + 1e-8))
        if sim < 0.55:
            continue
        scored.append((sim, item["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:top_k]]


def chat_with_retry(messages: list[dict]) -> str:
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
            )
            return completion.choices[0].message.content
        except OpenAIError as e:
            logger.error(f"Chat error (attempt {attempt+1}): {e}")
            time.sleep(OPENAI_RETRY_DELAY)
        except Exception as e:
            logger.error(f"Unexpected chat error: {e}")
            break
    raise HTTPException(status_code=500, detail="AI generation error")

# -----------------------------------------------------
# CHAT ENDPOINT (SECURED + RATE LIMITED)
# -----------------------------------------------------
@app.post("/chat")
async def chat(
    data: Message,
    request: Request,
    db: Session = Depends(get_db),
    x_api_key: str = Header(None)
):

    # Sanitize & validate basic input
    business_id = data.business_id.strip()
    if not business_id:
        raise HTTPException(status_code=400, detail="Missing business_id")

    user_id = data.user.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user id")

    clean_message = sanitize_message(data.message)

    # API key + per-business auth
    verify_api_key(x_api_key, business_id)

    # Lightweight rate limits
    ip = request.client.host if request.client else "unknown"
    enforce_rate_limit("ip_per_minute", ip)
    enforce_rate_limit("user_per_minute", f"{business_id}:{user_id}")
    enforce_rate_limit("business_per_minute", business_id)

    # Load business config
    profile = load_business_profile(business_id)
    system_prompt = profile.get("system_prompt", "You are a helpful assistant.")

    # Load memory
    memory = load_memory(db, user_id, business_id)

    # Retrieve KB snippets
    kb_chunks = rag_retrieve(business_id, clean_message)
    kb_context = "\n".join(kb_chunks) if kb_chunks else ""

    final_system_prompt = system_prompt
    if kb_context:
        final_system_prompt += "\nRelevant Business Info:\n" + kb_context

    messages = [{"role": "system", "content": final_system_prompt}]
    messages.extend(memory)
    messages.append({"role": "user", "content": clean_message})

    reply = chat_with_retry(messages)

    # Save memory
    save_memory(db, user_id, business_id, "user", clean_message)
    save_memory(db, user_id, business_id, "assistant", reply)

    return {"reply": reply}
