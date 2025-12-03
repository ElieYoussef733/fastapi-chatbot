import os
import uuid
import streamlit as st

from doc_ingest import process_file_for_business

BUSINESS_DIR = "business"
UPLOADS_ROOT = "knowledge/uploads"

def list_businesses():
    ids = []
    if os.path.isdir(BUSINESS_DIR):
        for fname in os.listdir(BUSINESS_DIR):
            if fname.endswith(".json"):
                ids.append(fname[:-5])
    return sorted(ids)

st.set_page_config(page_title="Admin – KB Uploader", page_icon="🛠")

st.title("🛠 Admin – Business Document Uploader")

businesses = list_businesses()
if not businesses:
    st.error("No businesses found in /business folder.")
    st.stop()

selected_business = st.selectbox("Choose business", businesses)

uploaded_files = st.file_uploader(
    "Upload client documents",
    accept_multiple_files=True,
    type=["pdf", "docx", "txt", "xlsx", "xlsm", "png", "jpg", "jpeg", "bmp", "tiff"]
)

if uploaded_files and st.button("Process uploaded files"):
    total_chunks = 0
    business_upload_dir = os.path.join(UPLOADS_ROOT, selected_business)
    os.makedirs(business_upload_dir, exist_ok=True)

    with st.spinner("Processing files..."):
        for uf in uploaded_files:
            # Save file to disk
            unique_name = f"{uuid.uuid4()}_{uf.name}"
            path = os.path.join(business_upload_dir, unique_name)
            with open(path, "wb") as f:
                f.write(uf.read())

            st.write(f"📂 Saved file: {path}")

            # Ingest
            chunks = process_file_for_business(selected_business, path)
            total_chunks += chunks
            st.write(f"✅ {chunks} chunks added from {uf.name}")

    st.success(f"🎉 Done! Total chunks added: {total_chunks}")
    st.info("Remember to git add/commit/push knowledge/indexes/*.json so Render gets the new KB.")
