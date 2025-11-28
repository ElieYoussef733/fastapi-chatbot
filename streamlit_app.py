import os
import json
import uuid
import requests
from dotenv import load_dotenv

import streamlit as st

# -----------------------------
# ENV & CONFIG
# -----------------------------
load_dotenv()

BACKEND_URL = "https://fastapi-chatbot-780g.onrender.com/chat"
MASTER_API_KEY = os.getenv("MASTER_API_KEY")  # same as FastAPI

BUSINESS_DIR = "business"


def list_businesses():
    """Return list of business_ids from /business folder (without .json)."""
    ids = []
    if os.path.isdir(BUSINESS_DIR):
        for fname in os.listdir(BUSINESS_DIR):
            if fname.endswith(".json"):
                ids.append(fname.replace(".json", ""))
    return sorted(ids)


def load_business_title(business_id: str) -> str:
    """
    Optional: show a nicer label in the UI
    (fallback to the raw id if anything fails).
    """
    path = os.path.join(BUSINESS_DIR, f"{business_id}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # try to extract something nice from system_prompt
        prompt = data.get("system_prompt", "")
        if "assistant for" in prompt:
            # e.g. "You are the AI assistant for Ali’s Phone Store."
            return prompt.split("assistant for", 1)[1].split(".")[0].strip()
    except Exception:
        pass
    return business_id


# -----------------------------
# STREAMLIT SETUP
# -----------------------------
st.set_page_config(page_title="Multi-Business Chatbot", page_icon="💬", layout="centered")

st.title("💬 Multi-Business Chatbot (Local)")
st.caption("FastAPI backend at `http://127.0.0.1:8000/chat`")

# --- Session state init ---
if "chat_history" not in st.session_state:
    # {business_id: [ {role, content}, ... ]}
    st.session_state.chat_history = {}

if "session_ids" not in st.session_state:
    # {business_id: uuid}
    st.session_state.session_ids = {}


# -----------------------------
# SIDEBAR: CONFIG
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # Business selector
    business_ids = list_businesses()
    if not business_ids:
        st.error("No business JSON files found in /business.")
        st.stop()

    selected_business = st.selectbox(
        "Business",
        business_ids,
        format_func=load_business_title,
    )

    # Ensure session id for this business
    if selected_business not in st.session_state.session_ids:
        st.session_state.session_ids[selected_business] = str(uuid.uuid4())

    user_id = st.session_state.session_ids[selected_business]
    st.text(f"User session: {user_id[:8]}...")

    # Reset conversation (front-end + new user id ⇒ new memory thread in backend)
    if st.button("🔄 Reset conversation"):
        st.session_state.chat_history[selected_business] = []
        st.session_state.session_ids[selected_business] = str(uuid.uuid4())
        st.success("Conversation reset for this business.")


# -----------------------------
# HELPERS
# -----------------------------
def get_history_for_current():
    return st.session_state.chat_history.setdefault(selected_business, [])


def add_message(role: str, content: str):
    history = get_history_for_current()
    history.append({"role": role, "content": content})


def call_backend(message: str) -> str:
    payload = {
        "business_id": selected_business,
        "user": st.session_state.session_ids[selected_business],
        "message": message,
    }

    headers = {"Content-Type": "application/json"}
    if MASTER_API_KEY:
        headers["x-api-key"] = MASTER_API_KEY  # FastAPI verify_api_key()

    resp = requests.post(BACKEND_URL, json=payload, headers=headers, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"Backend error {resp.status_code}: {resp.text}")

    data = resp.json()
    return data.get("reply", "")


# -----------------------------
# MAIN CHAT AREA
# -----------------------------
chat_container = st.container()

# Display existing history
with chat_container:
    history = get_history_for_current()
    for msg in history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

# Input area (Streamlit chat input)
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message immediately
    add_message("user", user_input)
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

    # Call backend & stream reply
    try:
        with st.spinner("Thinking..."):
            reply = call_backend(user_input)

        add_message("assistant", reply)
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(reply)

    except Exception as e:
        err_msg = f"Error contacting backend: {e}"
        add_message("assistant", err_msg)
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(f"❌ {err_msg}")
