import os
import json
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

print("DEBUG: Starting build_kb.py")

try:
    client = OpenAI()
    print("DEBUG: OpenAI client initialized.")
except Exception as e:
    print("ERROR: Failed to initialize OpenAI client:", e)
    raise

def read_text_from_folder(folder_path: str) -> str:
    print("DEBUG: Reading folder:", folder_path)

    if not os.path.isdir(folder_path):
        print("ERROR: Folder does not exist:", folder_path)
        raise RuntimeError(f"Folder not found: {folder_path}")

    chunks = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        print("DEBUG: Found file:", filename)

        if filename.lower().endswith(".txt"):
            print("DEBUG: Reading TXT:", filename)
            with open(path, "r", encoding="utf-8") as f:
                chunks.append(f.read())

        elif filename.lower().endswith(".pdf"):
            print("DEBUG: Reading PDF:", filename)
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            chunks.append(text)

    print("DEBUG: Finished reading documents.")
    return "\n".join(chunks)

def split_into_chunks(text: str, max_chars=700):
    print("DEBUG: Splitting into chunks")
    lines = text.splitlines()
    result = []
    current = []
    length = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if length + len(line) > max_chars:
            result.append(" ".join(current))
            current = [line]
            length = len(line)
        else:
            current.append(line)
            length += len(line)

    if current:
        result.append(" ".join(current))

    print("DEBUG: Total chunks:", len(result))
    return result

def build_kb_for_business(business_id: str):
    print(f"DEBUG: Building KB for {business_id}")

    folder = os.path.join("knowledge", business_id)
    print("DEBUG: Folder path:", folder)

    full_text = read_text_from_folder(folder)

    chunks = split_into_chunks(full_text)

    print("DEBUG: Generating embeddings...")

    kb_entries = []
    for chunk in chunks:
        print("DEBUG: Embedding chunk...")
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding

        kb_entries.append({
            "text": chunk,
            "embedding": emb
        })

    os.makedirs("knowledge/indexes", exist_ok=True)
    out_path = f"knowledge/indexes/{business_id}_kb.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kb_entries, f)

    print("SUCCESS: Saved index to:", out_path)

if __name__ == "__main__":
    build_kb_for_business("phone_store_ali")
