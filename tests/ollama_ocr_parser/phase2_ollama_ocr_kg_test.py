"""
Phase 2 (Jetson workaround): build the KG + vector index from Ollama-OCR output.

    PDF --ollama_ocr parser--> content_list --RAG-Anything--> chunks + KG

Run from the project root:
    uv run python -m tests.ollama_ocr_parser.phase2_ollama_ocr_kg_test

Stages are checked separately:
    1. chunks   (kv_store_text_chunks.json)      -> did the values reach storage?
    2. KG       (graph_chunk_entity_relation.graphml) -> did the LLM extract them?
Retrieval is phase 3:
    uv run python -m tests.phase3_retrieval_test --storage output_stores/rag_storage_ollama_ocr_3docs
"""

import asyncio
import json
import re
import shutil
import sys
from functools import partial
from pathlib import Path

import networkx as nx
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig

from raganything_hazop.extraction_prompt import (
    ENGINEERING_ENTITY_TYPES,
    configure_engineering_extraction_prompt,
)
from raganything_hazop.llm import make_ollama_llm_func
from raganything_hazop.parsers import PARSER_NAME, register_ollama_ocr_parser


STORAGE = Path("output_stores/rag_storage_ollama_ocr_3docs")
OUTPUT_DIR = "./output_stores/output"

OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "deepseek-v4.1-flash:cloud"
EMBEDDING_MODEL = "qwen3-embedding:8b"
EMBEDDING_DIM = 4096

# Document -> key values that must survive OCR -> chunks -> KG.
DOCUMENTS = {
    # Scanned (no text layer). Verified against the rendered page image.
    Path("data/documents/Enapter_Datasheet_EL40_EN.pdf"): [
        "500 NL/h", "35 barg", "99.9%", "98.8%", "4.8 kWh", "2.4 kW",
        "3 kW", "0.3 kW", "420 mL/h", "1-4 barg", "42 kg",
    ],
    # Has a text layer. Verified against the PDF text (comparison table, page_idx 9).
    Path("data/documents/SITRANS-P-EN.pdf"): [
        "0.04%", "0.065%", "0.075%", "0.2%", "1 mbar to 700 bar",
        "0 bar to 40 bar", "+100 °C", "+200 °C", "in preparation",
    ],
    # Scanned (no text layer). Values from the phase 3 expected facts.
    Path("data/documents/Hydrogen_compressed_Datasheet.pdf"): [
        "1333-74-0", "H220", "H280", "99.8%",
    ],
}
KEY_VALUES = [v for values in DOCUMENTS.values() for v in values]


def normalize(text: str) -> str:
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s*-\s*", "-", text)  # "1 – 4 barg" == "1-4 barg"
    return re.sub(r"\s+", " ", text).lower()


def contains(text: str, value: str) -> bool:
    return normalize(value) in normalize(text)


async def build_index():
    configure_engineering_extraction_prompt()
    register_ollama_ocr_parser()

    config = RAGAnythingConfig(
        parser=PARSER_NAME,
        parser_output_dir=OUTPUT_DIR,
        working_dir=str(STORAGE),
        enable_image_processing=False,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=16384,
        func=partial(ollama_embed.func, embed_model=EMBEDDING_MODEL, host=OLLAMA_HOST),
        model_name=EMBEDDING_MODEL,
    )

    rag = RAGAnything(
        config=config,
        # Not ollama_model_complete: it needs hashing_kv, which RAG-Anything's
        # table processor does not pass (-> KeyError 'hashing_kv').
        llm_model_func=make_ollama_llm_func(LLM_MODEL, host=OLLAMA_HOST, timeout=600),
        embedding_func=embedding_func,
        lightrag_kwargs={
            "llm_model_name": LLM_MODEL,
            "default_llm_timeout": 600,
            "llm_model_max_async": 1,
            "addon_params": {
                "language": "English",
                "entity_types": ENGINEERING_ENTITY_TYPES,
                "example_number": 1,
            },
        },
    )

    try:
        for pdf_path in DOCUMENTS:
            print(f"\n>>> Processing: {pdf_path.name}")
            await rag.process_document_complete(
                file_path=str(pdf_path),
                output_dir=OUTPUT_DIR,
                parse_method="auto",
                display_stats=True,
            )
    finally:
        await rag.finalize_storages()


def check_chunks():
    print("\n" + "=" * 80 + "\nSTAGE 1: CHUNKS (what was embedded into the vector DB)\n" + "=" * 80)

    chunks = json.loads((STORAGE / "kv_store_text_chunks.json").read_text(encoding="utf-8"))
    print(f"Total chunks: {len(chunks)}")

    for chunk_id, chunk in chunks.items():
        content = chunk.get("content", "")
        print(f"\n--- {chunk_id} | tokens={chunk.get('tokens')} | "
              f"multimodal={chunk.get('is_multimodal')} | type={chunk.get('original_type')}")
        print(f"file={chunk.get('file_path')}")
        print(content if len(content) <= 600 else content[:600] + "\n...")

    # Each value must be found in a chunk of *its own* document.
    print("\nValue -> number of chunks (of its own document) containing it:")
    missing = []
    for pdf_path, values in DOCUMENTS.items():
        doc_chunks = [c for c in chunks.values() if pdf_path.name in str(c.get("file_path", ""))]
        print(f"  {pdf_path.name}: {len(doc_chunks)} chunks")
        for value in values:
            n = sum(contains(c.get("content", ""), value) for c in doc_chunks)
            print(f"    {value:<20} {n}")
            if n == 0:
                missing.append(f"{pdf_path.name}: {value}")

    assert not missing, f"values missing from chunks: {missing}"

    # If the table description LLM call fails, RAG-Anything falls back to
    # repeating the raw table as the "Analysis" -> the table appears twice.
    table_chunks = [c for c in chunks.values() if c.get("original_type") == "table"]
    assert table_chunks, "no table chunk stored"
    for chunk in table_chunks:
        n_tables = chunk["content"].count("<table")
        assert n_tables == 1, (
            f"table appears {n_tables}x in its chunk -> table description failed (fallback used)"
        )
    print("[ok] chunks")


def check_kg():
    print("\n" + "=" * 80 + "\nSTAGE 2: KNOWLEDGE GRAPH (what the LLM extracted)\n" + "=" * 80)

    graph = nx.read_graphml(STORAGE / "graph_chunk_entity_relation.graphml")
    print(f"Nodes: {graph.number_of_nodes()}  Edges: {graph.number_of_edges()}")

    print("\nNodes:")
    for node_id, data in graph.nodes(data=True):
        print(f"  [{data.get('entity_type', '?')}] {node_id}")

    print("\nEdges:")
    for source, target, data in graph.edges(data=True):
        print(f"  {source} --[{data.get('keywords', '')}]--> {target}")

    print("\nValue -> where it appears in the KG:")
    missing = []
    for value in KEY_VALUES:
        hits = [n for n, d in graph.nodes(data=True) if contains(f"{n} {d}", value)]
        hits += [f"{s} -> {t}" for s, t, d in graph.edges(data=True) if contains(f"{s} {t} {d}", value)]
        print(f"  {value:<20} {len(hits)} hit(s) {hits[:3]}")
        if not hits:
            missing.append(value)

    assert not missing, f"values missing from KG: {missing}"

    # The spec values must belong to the EL 4.0, not to an invented equipment name.
    print("\nParameter nodes owned by EL 4.0:")
    not_owned = []
    for value in ["35 barg", "2.4 kW", "500 NL/h", "1-4 barg"]:
        owners = [n for n, d in graph.nodes(data=True)
                  if "el 4.0" in n.lower() and contains(f"{n} {d}", value)]
        print(f"  {value:<10} {owners[:3]}")
        if not owners:
            not_owned.append(value)

    assert not not_owned, f"values not attached to an 'EL 4.0' node: {not_owned}"
    print("[ok] KG")


def main():
    for pdf_path in DOCUMENTS:
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

    # --check-only: inspect the existing storage without rebuilding it.
    if "--check-only" not in sys.argv:
        # Fresh index: results must come only from this run.
        if STORAGE.exists():
            shutil.rmtree(STORAGE)
        asyncio.run(build_index())

    check_chunks()
    check_kg()

    print(f"\nAll phase 2 checks passed. Storage: {STORAGE}")


if __name__ == "__main__":
    main()
