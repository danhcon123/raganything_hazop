from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "documents"

def load_documents():
    documents = []

    for file in DATA_PATH.glob("*.txt"):
        text = file.read_text(encoding="utf-8")

        documents.append({
            "name": file.name,
            "content": text,
        })
        
    return documents

def main():
    docs = load_documents()

    print("Looking in:")
    print(DATA_PATH)

    print("\nLoaded documents:")

    for doc in docs:
        print(doc["name"])

if __name__ == "__main__":
    main()