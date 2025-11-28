import os
import json
import uuid
import requests
from dotenv import load_dotenv
import streamlit as st

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
load_dotenv()

# Your Render backend URL
BACKEND_URL = "https://fastapi-chatbot-780g.onrender.com/chat"

# Public demo API key (safe for Streamlit)
DEMO_KEY = os.getenv("PUBLIC_DEMO_KEY", "Demo367")

HEADERS = {"x-api-key": DEMO_KEY}

BUSINESS_DIR = "business"


# -----------------------------------------------------
# HELPERS
# -----------------------------------------------------
def list_businesses():
    """List business JSON files from /business."""
    ids = []
    if os.path.isdir(BUSINESS_DIR):
        for fname in os.listdir(BUSINESS_DIR):
            if fname.endswith(".json"):
                ids.append(fname[:-5])
    return sorted(ids)


def load_business_title(business_id: str):
    """Show nicer labels in dropdown."""
    path = os.path.join(BUSINESS_DIR, f"{business_id}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            prompt = data.get("system_prompt", "")
            if "assistant for" in prompt:
                return prompt.split("assistant for", 1)[1].split(".")[0].strip()
    except:
        pass
    return business_id


def call_backend(business_id: str, user_id: str, message: str):
    payload = {
        "business_id": business_id,
        "user": user_id,
        "message": message
    }

    resp = requests.post(BACKEND_URL, json=payload, headers=HEADERS, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"Backend error {resp.status_code}: {resp.text}")

    return resp.json().get("reply", "No reply")


# -----------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------
st.set_page_config(page_title="Multi-Business Chatbot", page_icon="💬", layout="centered")

st.title("💬 Multi-Business Chatbot (Live Demo)")

# Session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "session_ids" not in st.session_state:
    st.session_state.session_ids = {}


# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # Business selector
    business_ids = list_businesses()
    if not business_ids:
        st.error("❌ No business profiles found in /business folder.")
        st.stop()

    selected_business = st.selectbox(
        "Choose a Business",
        business_ids,
        format_func=load_business_title,
    )

    # Unique user session per business
    if selected_business not in st.session_state.session_ids:
        st.session_state.session_ids[selected_business] = str(uuid.uuid4())

    user_id = st.session_state.session_ids[selected_business]
    st.write(f"Session ID: `{user_id[:8]}...`")

    # Reset conversation
    if st.button("🔄 Reset Conversation"):
        st.session_state.chat_history[selected_business] = []
        st.session_state.session_ids[selected_business] = str(uuid.uuid4())
        st.success("Conversation reset.")


# -----------------------------------------------------
# LOAD OR INIT HISTORY
# -----------------------------------------------------
def get_history():
    return st.session_state.chat_history.setdefault(selected_business, [])


# -----------------------------------------------------
# CHAT DISPLAY
# -----------------------------------------------------
chat_container = st.container()

with chat_container:
    history = get_history()
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# -----------------------------------------------------
# INPUT AREA
# -----------------------------------------------------
user_input = st.chat_input("Type a message...")

if user_input:
    # Add to UI
    get_history().append({"role": "user", "content": user_input})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

    # Get reply
    try:
        with st.spinner("Thinking..."):
            reply = call_backend(selected_business, user_id, user_input)

        get_history().append({"role": "assistant", "content": reply})

        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(reply)

    except Exception as e:
        error_msg = f"❌ Error contacting backend: {e}"
        get_history().append({"role": "assistant", "content": error_msg})
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(error_msg)
