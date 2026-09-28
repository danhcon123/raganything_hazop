import json
from pathlib import Path

path = Path("rag_storage_phase2/kv_store_doc_status.json")

with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

for doc_id, status in data.items():
    print("=" * 100)
    print("DOC ID:", doc_id)

    for key in [
        "status",
        "multimodal_processed",
        "chunks_count",
        "chunks_list",
        "error_msg",
        "file_path",
    ]:
        print(f"{key}: {status.get(key)}")