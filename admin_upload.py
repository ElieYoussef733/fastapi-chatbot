import os
import argparse

from doc_ingest import process_file_for_business

def main():
    parser = argparse.ArgumentParser(description="Admin file ingestion for business KB.")
    parser.add_argument(
        "--business",
        required=True,
        help="Business ID (e.g., gym_180fitness, restaurant_mario)"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the file to ingest (pdf, docx, txt, xlsx, image)"
    )

    args = parser.parse_args()
    business_id = args.business
    file_path = args.file

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"📂 Processing file '{file_path}' for business '{business_id}'...")
    try:
        chunks_added = process_file_for_business(business_id, file_path)
        print(f"✅ Done. Added {chunks_added} chunks to {business_id}_kb.json.")
    except Exception as e:
        print(f"❌ Error during processing: {e}")

if __name__ == "__main__":
    main()
