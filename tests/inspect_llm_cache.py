import json
from pathlib import Path


CACHE_PATH = Path(
    "rag_storage_table_test_4/kv_store_llm_response_cache.json"
)


def main():
    if not CACHE_PATH.is_file():
        print(f"Cache file not found: {CACHE_PATH}")
        return

    with CACHE_PATH.open("r", encoding="utf-8") as f:
        cache = json.load(f)

    print("Cache:", CACHE_PATH)
    print("Total records:", len(cache))

    found = 0

    for key, value in cache.items():
        if ":extract:" not in key:
            continue

        found += 1
        print("\n" + "=" * 100)
        print("KEY:", key)

        if isinstance(value, dict):
            for field, content in value.items():
                print(f"\n--- {field} ---")
                print(content)
        else:
            print(value)

    print(f"\nExtraction records found: {found}")

    if found == 0:
        print("Available keys:")
        for key in cache:
            print(key)


if __name__ == "__main__":
    main()