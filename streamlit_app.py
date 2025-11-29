import os
import json
import uuid
import requests
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------

# Backend URL (Render)
BACKEND_URL = "https://fastapi-chatbot-780g.onrender.com/chat"

# Demo key for public frontend
DEMO_KEY = "demo123"
HEADERS = {"x-api-key": DEMO_KEY}

BUSINESS_DIR = "business"


# -----------------------------
# HELPERS
# -----------------------------
def list_businesses():
    """List business JSON files from /business folder."""
    ids = []
    if os.path.isdir(BUSINESS_DIR):
        for fname in os.listdir(BUSINESS_DIR):
            if fname.endswith(".json"):
                ids.append(fname.replace(".json", ""))
    return sorted(ids)


def load_business_title(business_id: str):
    """Pretty label for dropdown."""
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
    """Send message to FastAPI backend."""
    payload = {
        "business_id": business_id,
        "user": user_id,
        "message": message
    }

    resp = requests.post(
        BACKEND_URL,
        json=payload,
        headers=HEADERS,
        timeout=60
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Backend error {resp.status_code}: {resp.text}")

    return resp.json().get("reply", "")


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(page_title="Multi-Business Chatbot", page_icon="💬")

st.title("💬 Multi-Business Chatbot (Live Demo)")

# Session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "session_ids" not in st.session_state:
    st.session_state.session_ids = {}


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    # List available businesses
    business_ids = list_businesses()

    # DEBUG PRINT — Verify Streamlit sees gym_180fitness
    st.write("Loaded businesses:", business_ids)

    if not business_ids:
        st.error("❌ No businesses found in /business folder.")
        st.stop()

    selected_business = st.selectbox(
        "Choose a Business:",
        business_ids,
        format_func=load_business_title
    )

    # Assign session ID per business
    if selected_business not in st.session_state.session_ids:
        st.session_state.session_ids[selected_business] = str(uuid.uuid4())

    user_id = st.session_state.session_ids[selected_business]
    st.write(f"Session ID: `{user_id[:8]}`")

    # Reset conversation
    if st.button("🔄 Reset Conversation"):
        st.session_state.chat_history[selected_business] = []
        st.success("Conversation reset!")


# Load history for selected business
history = st.session_state.chat_history.setdefault(selected_business, [])


# -----------------------------
# CHAT DISPLAY
# -----------------------------
chat_container = st.container()

with chat_container:
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# -----------------------------
# INPUT AREA
# -----------------------------
user_input = st.chat_input("Type a message...")

if user_input:
    # Add to UI
    history.append({"role": "user", "content": user_input})

    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

    # Call backend
    try:
        with st.spinner("Thinking..."):
            reply = call_backend(selected_business, user_id, user_input)

        history.append({"role": "assistant", "content": reply})

        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(reply)

    except Exception as e:
        err = f"❌ Error contacting backend: {e}"
        history.append({"role": "assistant", "content": err})
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(err)
