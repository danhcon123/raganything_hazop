import json
from pathlib import Path

WORKING_DIR = Path("rag_storage_table_test_2")
CHUNKS_FILE = WORKING_DIR / "kv_store_text_chunks.json"

SEARCH_TERMS = [
    "500",
    "NL/h",
    "35 barg",
    "420",
    "2.4",
    "3 kW",
    "45",
    "42 kg",
]


def main():
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Total chunks: {len(chunks)}")

    for chunk_id, chunk in chunks.items():
        content = chunk.get("content", "")

        print("\n" + "=" * 100)
        print("CHUNK:", chunk_id)
        print("TOKENS:", chunk.get("tokens"))
        print("FILE:", chunk.get("file_path"))
        print("IS MULTIMODAL:", chunk.get("is_multimodal"))
        print("ORIGINAL TYPE:", chunk.get("original_type"))
        print("\nCONTENT:")
        print(content)

        matches = [
            term for term in SEARCH_TERMS
            if term.lower() in content.lower()
        ]

        if matches:
            print("\n>>> MATCHED TABLE VALUES:", matches)


if __name__ == "__main__":
    main()