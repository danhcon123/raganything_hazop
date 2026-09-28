import json
from pathlib import Path


CACHE_PATH = Path(
    "rag_storage_table_test_2/kv_store_llm_response_cache.json"
)

TARGET_KEYS = [
    "default:extract:199e04e1f2faed58983a42a6e2ac7b89",
    "default:extract:ad806559f59e0f67f30225b03511fbf7",
]


def main():
    with CACHE_PATH.open("r", encoding="utf-8") as f:
        cache = json.load(f)

    for key in TARGET_KEYS:
        print("\n" + "=" * 120)
        print("KEY:", key)

        value = cache.get(key)

        if value is None:
            print("NOT FOUND")
            continue

        if isinstance(value, dict):
            for k, v in value.items():
                print(f"\n--- {k} ---")
                print(v)
        else:
            print(value)


if __name__ == "__main__":
    main()