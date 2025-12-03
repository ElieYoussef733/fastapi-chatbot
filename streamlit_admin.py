import os
import uuid
import time
import json
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from doc_ingest import (
    process_file_for_business,
    extract_text_from_file,
    split_into_chunks,
    embed_texts,
    load_existing_index,
    save_index,
)

load_dotenv()

BUSINESS_DIR = "business"
UPLOADS_ROOT = "knowledge/uploads"
INDEX_DIR = "knowledge/indexes"

# NEW: folders for review + approved text
REVIEW_ROOT = "knowledge/review"
APPROVED_ROOT = "knowledge/approved"

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")


# -----------------------------
# Helpers
# -----------------------------
def list_businesses():
    ids = []
    if os.path.isdir(BUSINESS_DIR):
        for fname in os.listdir(BUSINESS_DIR):
            if fname.endswith(".json"):
                ids.append(fname[:-5])
    return sorted(ids)


def get_business_upload_dir(business_id: str) -> str:
    path = os.path.join(UPLOADS_ROOT, business_id)
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


# NEW: review + approved folders
def get_business_review_dir(business_id: str) -> str:
    path = os.path.join(REVIEW_ROOT, business_id)
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_business_approved_dir(business_id: str) -> str:
    path = os.path.join(APPROVED_ROOT, business_id)
    path = os.path.normpath(path)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path


def list_uploaded_files(business_id: str):
    folder = get_business_upload_dir(business_id)
    files = []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            files.append((fname, fpath, size))
    return sorted(files, key=lambda x: x[0].lower())


def get_kb_stats(business_id: str):
    path = os.path.join(INDEX_DIR, f"{business_id}_kb.json")
    if not os.path.exists(path):
        return 0, None
    kb = load_existing_index(business_id)
    size = len(kb)
    mtime = os.path.getmtime(path)
    return size, datetime.fromtimestamp(mtime)


def rebuild_kb_from_uploads(business_id: str):
    """
    OLD BEHAVIOR – rebuild from ALL uploaded docs (without approval).
    Kept for compatibility.
    """
    upload_dir = get_business_upload_dir(business_id)
    all_files = [
        os.path.join(upload_dir, f)
        for f in os.listdir(upload_dir)
        if os.path.isfile(os.path.join(upload_dir, f))
    ]

    if not all_files:
        return 0

    all_chunks = []
    for fpath in all_files:
        try:
            text = extract_text_from_file(fpath)
            chunks = split_into_chunks(text)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[WARN] Failed to process {fpath}: {e}")

    if not all_chunks:
        return 0

    items = embed_texts(all_chunks)
    save_index(business_id, items)
    return len(all_chunks)


# NEW: load system prompt for business
def load_business_system_prompt(business_id: str) -> str:
    path = os.path.join(BUSINESS_DIR, f"{business_id}.json")
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("system_prompt", "") or ""
    except Exception:
        return ""


# NEW: rebuild KB **only from approved docs + system prompt**
def rebuild_kb_from_approved(business_id: str) -> int:
    approved_dir = get_business_approved_dir(business_id)

    approved_files = [
        os.path.join(approved_dir, f)
        for f in os.listdir(approved_dir)
        if os.path.isfile(os.path.join(approved_dir, f))
    ]

    if not approved_files:
        return 0

    all_chunks = []

    # 1) include business system prompt as context
    sys_prompt = load_business_system_prompt(business_id)
    if sys_prompt.strip():
        prompt_chunks = split_into_chunks(sys_prompt)
        all_chunks.extend(prompt_chunks)

    # 2) include approved docs
    for fpath in approved_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            chunks = split_into_chunks(text)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[WARN] Failed to read approved file {fpath}: {e}")

    if not all_chunks:
        return 0

    items = embed_texts(all_chunks)
    save_index(business_id, items)
    return len(all_chunks)


# -----------------------------
# Auth gate
# -----------------------------
st.set_page_config(page_title="Admin – KB Uploader", page_icon="🛠")

st.title("🛠 Admin – Business Document Control Panel")

if ADMIN_PASSWORD:
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if not st.session_state.auth_ok:
        pwd = st.text_input("Enter admin password", type="password")
        if st.button("Login"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.auth_ok = True
                st.success("✅ Authenticated as admin.")
            else:
                st.error("❌ Wrong password.")
        st.stop()
else:
    st.warning("⚠️ ADMIN_PASSWORD not set in .env – admin page is unprotected on this machine.")


# -----------------------------
# Sidebar – Global info
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    businesses = list_businesses()
    if not businesses:
        st.error("No businesses found in /business folder.")
        st.stop()

    selected_business = st.selectbox("Choose business", businesses)

    kb_size, kb_mtime = get_kb_stats(selected_business)
    st.write(f"📚 KB size: **{kb_size}** chunks")
    if kb_mtime:
        st.write(f"🕒 Last updated: {kb_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.write("🕒 Last updated: N/A")

    if st.button("🔄 Refresh stats"):
        st.experimental_rerun()


st.subheader(f"📁 Document Management for: `{selected_business}`")

# -----------------------------
# Upload section
# -----------------------------
st.markdown("### ⬆️ Upload new documents")

uploaded_files = st.file_uploader(
    "Drop client documents here",
    accept_multiple_files=True,
    type=["pdf", "docx", "txt", "xlsx", "xlsm", "png", "jpg", "jpeg", "bmp", "tiff"]
)

if uploaded_files and st.button("Save uploaded files"):
    upload_dir = get_business_upload_dir(selected_business)
    saved_files = []
    with st.spinner("Saving files..."):
        for uf in uploaded_files:
            unique_name = f"{int(time.time())}_{uuid.uuid4().hex}_{uf.name}"
            fpath = os.path.join(upload_dir, unique_name)
            with open(fpath, "wb") as f:
                f.write(uf.read())
            saved_files.append(unique_name)
    st.success(f"✅ Saved {len(saved_files)} file(s).")
    st.info("You can now review / approve and then rebuild the KB.")
    st.rerun()


# -----------------------------
# Existing files list + REVIEW / APPROVE
# -----------------------------
st.markdown("### 📂 Existing uploaded documents")

files = list_uploaded_files(selected_business)
if not files:
    st.info("No uploaded documents yet for this business.")
else:
    review_dir = get_business_review_dir(selected_business)
    approved_dir = get_business_approved_dir(selected_business)

    for fname, fpath, size in files:
        size_kb = round(size / 1024, 1)
        with st.expander(f"📄 {fname} ({size_kb} KB)"):

            # Preview + delete buttons (as before)
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button("👀 Quick preview", key=f"preview_{fname}"):
                    try:
                        text = extract_text_from_file(fpath)
                        st.text_area(
                            "Extracted text (first 1000 chars)",
                            text[:1000],
                            height=200,
                        )
                    except Exception as e:
                        st.error(f"Error extracting text: {e}")

            with col2:
                if st.button("🗑 Delete", key=f"delete_{fname}"):
                    try:
                        os.remove(fpath)
                        # also remove review / approved versions if they exist
                        review_path = os.path.join(review_dir, fname + ".txt")
                        approved_path = os.path.join(approved_dir, fname + ".txt")
                        if os.path.exists(review_path):
                            os.remove(review_path)
                        if os.path.exists(approved_path):
                            os.remove(approved_path)
                        st.success(f"Deleted {fname} (and related review/approved files)")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting file: {e}")

            with col3:
                st.write(f"Path: `{fpath}`")

            st.markdown("---")
            st.markdown("#### ✏️ Review & approve extracted text")

            # FORM: review + approve text for this file
            review_path = os.path.join(review_dir, fname + ".txt")
            approved_path = os.path.join(approved_dir, fname + ".txt")

            # load or create review text
            if os.path.exists(review_path):
                try:
                    with open(review_path, "r", encoding="utf-8", errors="ignore") as f:
                        existing_text = f.read()
                except Exception:
                    existing_text = ""
            else:
                # first time: extract from original file
                try:
                    existing_text = extract_text_from_file(fpath)
                except Exception as e:
                    st.error(f"Error extracting text for review: {e}")
                    existing_text = ""

            with st.form(f"review_form_{fname}"):
                editable_text = st.text_area(
                    "Extracted text",
                    value=existing_text,
                    height=250,
                )
                col_save, col_approve = st.columns(2)
                save_clicked = col_save.form_submit_button("💾 Save draft")
                approve_clicked = col_approve.form_submit_button("✅ Approve for KB")

                if save_clicked or approve_clicked:
                    # always save to review file
                    try:
                        with open(review_path, "w", encoding="utf-8") as f:
                            f.write(editable_text)
                        st.success("Draft saved to review folder.")
                    except Exception as e:
                        st.error(f"Error saving review draft: {e}")

                if approve_clicked:
                    # also copy to approved folder
                    try:
                        with open(approved_path, "w", encoding="utf-8") as f:
                            f.write(editable_text)
                        st.success("✅ Document approved and stored in APPROVED KB folder.")
                    except Exception as e:
                        st.error(f"Error approving document: {e}")


# -----------------------------
# KB Actions
# -----------------------------
st.markdown("### 🧠 Knowledge Base Actions")

col_ingest, col_rebuild_uploads, col_rebuild_approved = st.columns(3)

# 1) Legacy: append to KB from all uploads directly
with col_ingest:
    if st.button("➕ Ingest ALL uploads (append)"):
        upload_dir = get_business_upload_dir(selected_business)
        all_files = [
            os.path.join(upload_dir, f)
            for f in os.listdir(upload_dir)
            if os.path.isfile(os.path.join(upload_dir, f))
        ]
        if not all_files:
            st.warning("No files to ingest.")
        else:
            total_chunks = 0
            with st.spinner("Ingesting documents..."):
                for fpath in all_files:
                    try:
                        c = process_file_for_business(selected_business, fpath)
                        total_chunks += c
                    except Exception as e:
                        st.error(f"Error ingesting {fpath}: {e}")
            st.success(f"✅ Ingestion complete. Added ~{total_chunks} chunks.")
            st.info("Remember to git add/commit/push knowledge/indexes/*.json to update Render.")
            kb_size, kb_mtime = get_kb_stats(selected_business)
            st.write(f"📚 New KB size: **{kb_size}** chunks")
            if kb_mtime:
                st.write(f"🕒 Last updated: {kb_mtime.strftime('%Y-%m-%d %H:%M:%S')}")

# 2) Legacy: rebuild from all uploads (no approval)
with col_rebuild_uploads:
    if st.button("🧹 Rebuild KB from ALL uploads"):
        with st.spinner("Rebuilding KB from ALL uploaded docs..."):
            chunks_count = rebuild_kb_from_uploads(selected_business)
        st.success(f"✅ Rebuild complete. New KB has ~{chunks_count} chunks.")
        st.info("Remember to git add/commit/push knowledge/indexes/*.json to update Render.")
        kb_size, kb_mtime = get_kb_stats(selected_business)
        st.write(f"📚 New KB size: **{kb_size}** chunks")
        if kb_mtime:
            st.write(f"🕒 Last updated: {kb_mtime.strftime('%Y-%m-%d %H:%M:%S')}")

# 3) NEW: rebuild from APPROVED docs + system prompt
with col_rebuild_approved:
    if st.button("✅ Build KB from APPROVED docs + prompts"):
        with st.spinner("Building KB ONLY from APPROVED docs + system prompt..."):
            chunks_count = rebuild_kb_from_approved(selected_business)

        if chunks_count == 0:
            st.warning("No approved documents found for this business.")
        else:
            st.success(f"✅ Approved-KB rebuild complete. New KB has ~{chunks_count} chunks.")
            st.info("Now run git add/commit/push for knowledge/indexes/*.json to update Render.")

            kb_size, kb_mtime = get_kb_stats(selected_business)
            st.write(f"📚 New KB size: **{kb_size}** chunks")
            if kb_mtime:
                st.write(f"🕒 Last updated: {kb_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
