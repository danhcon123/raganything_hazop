from pathlib import Path

PDF_PATH = (
    Path(__file__)
        .resolve()
        .parents[3]
        / "data"
        / "documents"
        / "Enapter_Datasheet_EL40_EN.pdf"
)

def ingest_pdf():
    print("Processing:")
    print(PDF_PATH)

if __name__ == "__main__":
    ingest_pdf()