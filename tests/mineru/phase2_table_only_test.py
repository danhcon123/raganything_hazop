import asyncio
import json
from functools import partial
from pathlib import Path

from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig

from raganything_hazop.extraction_prompt import (
    ENGINEERING_ENTITY_TYPES,
    configure_engineering_extraction_prompt,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PARSE_CACHE = Path(
    "output_stores/rag_storage_phase2/kv_store_parse_cache.json"
)

# IMPORTANT:
# Use a NEW directory so the result is not mixed with the previous
# 18-node / 27-edge baseline.
TEST_STORAGE = "./output_stores/rag_storage_dexpi_llm_engineering_test_1"


# ---------------------------------------------------------------------
# Load known-good table from MinerU parse cache
# ---------------------------------------------------------------------

def load_table_from_cache():
    with PARSE_CACHE.open("r", encoding="utf-8") as f:
        cache = json.load(f)

    cache_entry = next(iter(cache.values()))

    tables = [
        item
        for item in cache_entry["content_list"]
        if item.get("type") == "table"
    ]

    if len(tables) != 1:
        raise RuntimeError(
            f"Expected exactly 1 table, found {len(tables)}"
        )

    return tables[0]


import logging

logger = logging.getLogger(__name__)


async def debug_llm(
    prompt,
    system_prompt=None,
    history_messages=None,
    **kwargs,
):
    try:
        return await ollama_model_complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=(
                history_messages if history_messages is not None else []
            ),
            **kwargs,
        )
    except asyncio.CancelledError:
        logger.exception("LLM call cancelled, possibly by an outer timeout")
        raise
    except Exception as exc:
        logger.exception(
            "LLM call failed: type=%s, repr=%r",
            type(exc).__name__,
            exc,
        )
        raise

# ---------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------

async def main():

    # Must happen BEFORE RAGAnything / LightRAG initialization.
    configure_engineering_extraction_prompt()

    print("=== ENGINEERING EXTRACTION CONFIGURATION ===")
    print("Entity types:")
    for entity_type in ENGINEERING_ENTITY_TYPES:
        print(f"  - {entity_type}")

    PDF_DOCUMENTS = [ 
        Path("data/documents/Enapter_Datasheet_EL40_EN.pdf"),
        Path("data/documents/Hydrogen_compressed_Datasheet.pdf"),
        Path("data/documents/SITRANS-P-EN.pdf"),
    ]

    DEXPI_DOCUMENTS = [
        Path("data/others/C01V04-VER.EX01.xml")
    ]

    for document in PDF_DOCUMENTS + DEXPI_DOCUMENTS:
        if not document.is_file():
            raise FileNotFoundError(document.resolve())


    config = RAGAnythingConfig(
        parser="mineru",
        parser_output_dir="./output_stores/output",

        # Fresh KG for this experiment.
        working_dir=TEST_STORAGE,

        # Process the full document's text, tables and equations.
        enable_image_processing=False,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    embedding_func = EmbeddingFunc(
        embedding_dim=4096,
        max_token_size=16384,
        func=partial(
            ollama_embed.func,
            embed_model="qwen3-embedding:8b",
            host="http://localhost:11434",
        ),
        model_name="qwen3-embedding:8b",
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=ollama_model_complete,
        embedding_func=embedding_func,
        lightrag_kwargs={
            "llm_model_name": "deepseek-v4.1-flash:cloud",

            # LightRAG timeout and cloud request concurrency.
            "default_llm_timeout": 600,
            "llm_model_max_async": 1,

            # Local Ollama forwards this model's requests to Ollama Cloud.
            "llm_model_kwargs": {
                "host": "http://localhost:11434",
                "timeout": 600,
            },

            "addon_params": {
                "language": "English",
                "entity_types": ENGINEERING_ENTITY_TYPES,
                "example_number": 1,
            },
        },
    )

    try:
        for document in PDF_DOCUMENTS:
            print(f"\nProcessing: {document.name}")

            await rag.process_document_complete(
                file_path=str(document),
                output_dir="./output_stores/output",
                parse_method="auto",
                display_stats=True,
            )

        for document in DEXPI_DOCUMENTS:
            xml_text = document.read_text(encoding="utf-8")

            content_list = [
                {
                    "type": "dexpi",
                    "content": xml_text,
                    "page_idx": 0,
                }
            ]

            await rag.insert_content_list(
                content_list=content_list,
                file_path=str(document),
            )

    finally:
        await rag.finalize_storages()

    print("\n=== HAZOP INGESTION TEST COMPLETE ===")
    print(f"Storage: {TEST_STORAGE}")


if __name__ == "__main__":
    asyncio.run(main())