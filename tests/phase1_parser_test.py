import asyncio

from pathlib import Path
from raganything import RAGAnything, RAGAnythingConfig


async def main():

    pdf_path = Path(
        "data/documents/Enapter_Datasheet_EL40_EN.pdf",
    )

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    config = RAGAnythingConfig(
        parser="mineru",
        parser_output_dir="./output",
        working_dir="./rag_storage",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    rag = RAGAnything(config=config)
    rag.check_parser_installation()

    print("Starting document parsing...")
    result = await rag.parse_document(
        file_path=str(pdf_path)
    )

    print("\n=== Parsing Result ====")
    print(result)

if __name__=="__main__":
    asyncio.run(main())