import asyncio
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc


OLLAMA_HOST = "http://localhost:11434"

LLM_MODEL = "qwen3.8:latest"
EMBEDDING_MODEL = "qwen3-embedding:8b"
EMBEDDING_DIM = 4096




async def main():
    pdf_path = Path(
        "data/documents/Enapter_Datasheet_EL40_EN.pdf"
    )

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    config = RAGAnythingConfig(
        parser="mineru",
        parser_output_dir="./output",
        working_dir="./rag_storage_phase2",

        # First KG test:
        enable_image_processing=False,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=8192,
        func=partial(
            ollama_embed.func,
            embed_model=EMBEDDING_MODEL,
            host=OLLAMA_HOST,
        ),
        model_name=EMBEDDING_MODEL,
    )

    rag = RAGAnything(
        config=config,
        # Native LightRAG -> Ollama integration
        llm_model_func=ollama_model_complete,
        embedding_func=embedding_func,
        lightrag_kwargs={
            "llm_model_name": LLM_MODEL,
            "llm_model_kwargs": {
                "host": OLLAMA_HOST,
                "options":{
                    "num_ctx": 8192,
                },
                "timeout": 300,
            },
        },
    )

    try:
        print("Starting complete RAG-Anything processing...")

        await rag.process_document_complete(
            file_path=str(pdf_path)
        )

        print("Processing finished.")

    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())