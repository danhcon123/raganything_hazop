"""
Phase 1 (Jetson workaround): parse documents with an Ollama OCR model instead of MinerU.

Checks only the parser stage: file -> RAG-Anything content_list.
No LLM extraction, no KG, no embeddings.

Expected values below were verified by eye against the datasheet page image.
"""

import re
import tempfile
from pathlib import Path

import pymupdf
from raganything import RAGAnything, RAGAnythingConfig
from raganything.parser import get_parser

from raganything_hazop.parsers.ollama_ocr import (
    PARSER_NAME,
    OllamaOCRParser,
    register_ollama_ocr_parser,
)


PDF_PATH = Path("data/documents/Enapter_Datasheet_EL40_EN.pdf")

# Specification values from page 2 of the datasheet (page_idx=1).
EXPECTED_TABLE_VALUES = [
    "500 NL/h",
    "1.0785 kg/24 h",
    "Up to 35 barg",
    "4.8 kWh/Nm³",
    "2.4 kW",
    "Peak power consumption",
    "0.3 kW",
    "220 - 240 V (AC), 50/60 Hz",
    "~ 420 mL/h",
    "1 - 4 barg",
    "5 °C - 45 °C",
    "IP 20",
    "42 kg",
]

VALID_BLOCK_TYPES = {"text", "table", "image", "equation"}


def normalize(text: str) -> str:
    """Make comparisons robust to dash variants and whitespace."""
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text)


def print_blocks(content_list):
    for i, block in enumerate(content_list):
        body = block.get("text") or block.get("table_body") or block.get("img_path", "")
        preview = body if len(body) <= 300 else body[:300] + " ..."
        print(f"  [{i}] page={block['page_idx']} type={block['type']}: {preview}")


def check_block_schema(content_list):
    assert isinstance(content_list, list) and content_list, "content_list is empty"

    for block in content_list:
        assert block.get("type") in VALID_BLOCK_TYPES, f"bad type: {block}"
        assert isinstance(block.get("page_idx"), int), f"missing page_idx: {block}"

        if block["type"] == "text":
            assert block.get("text", "").strip(), f"empty text block: {block}"
        if block["type"] == "table":
            assert "<table" in block.get("table_body", ""), f"table without <table>: {block}"
            assert isinstance(block.get("table_caption", []), list)
            assert isinstance(block.get("table_footnote", []), list)


def test_registration():
    register_ollama_ocr_parser()
    parser = get_parser(PARSER_NAME)

    assert isinstance(parser, OllamaOCRParser)
    assert parser.check_installation(), "Ollama not reachable or OCR model not pulled"

    # RAG-Anything itself must accept the parser name.
    rag = RAGAnything(config=RAGAnythingConfig(parser=PARSER_NAME))
    assert rag.check_parser_installation()

    print("[ok] registration")


def test_parse_pdf():
    parser = get_parser(PARSER_NAME)
    content_list = parser.parse_document(PDF_PATH, output_dir="./output")
    print_blocks(content_list)

    check_block_schema(content_list)

    pages = {block["page_idx"] for block in content_list}
    assert pages == {0, 1}, f"expected pages 0 and 1, got {pages}"

    # Cover page: plain text, no table.
    page0_text = normalize(" ".join(
        b["text"] for b in content_list if b["page_idx"] == 0 and b["type"] == "text"
    ))
    assert "AEM Electrolyser EL 4.0" in page0_text
    assert "anion exchange membrane" in page0_text
    assert not [b for b in content_list if b["page_idx"] == 0 and b["type"] == "table"], (
        "page 0 has no table, but a table block was produced"
    )

    # Specification page: values must land in a *table* block, so they reach
    # RAG-Anything's table processor (the path that builds the engineering KG).
    tables = [b for b in content_list if b["page_idx"] == 1 and b["type"] == "table"]
    assert len(tables) >= 1, "no table block on the specification page"
    table_text = normalize(" ".join(b["table_body"] for b in tables))

    missing = [v for v in EXPECTED_TABLE_VALUES if normalize(v) not in table_text]
    assert not missing, f"values missing from table blocks: {missing}"

    # Multi-line cells must not be glued together ("at 25 °C8 barg").
    assert "°C8barg" not in table_text.replace(" ", ""), "multi-line cell lines were merged"

    # No duplicates: table values must not also appear in text blocks,
    # otherwise they enter the KG/vector DB twice (text path + table path).
    page1_text = normalize(" ".join(
        b["text"] for b in content_list if b["page_idx"] == 1 and b["type"] == "text"
    ))
    duplicated = [v for v in EXPECTED_TABLE_VALUES if normalize(v) in page1_text]
    assert not duplicated, f"table values duplicated in text blocks: {duplicated}"

    print(f"[ok] parse_pdf: {len(content_list)} blocks, {len(tables)} table(s)")


def test_parse_image():
    # Image input must be handled by our parser; otherwise RAG-Anything
    # silently falls back to MinerU for images.
    parser = get_parser(PARSER_NAME)

    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "cover.png"
        pymupdf.open(PDF_PATH)[0].get_pixmap(dpi=150).save(png)

        content_list = parser.parse_document(png, output_dir=tmp)
    print_blocks(content_list)

    check_block_schema(content_list)
    text = normalize(" ".join(b.get("text", "") for b in content_list))
    assert "AEM Electrolyser EL 4.0" in text

    print(f"[ok] parse_image: {len(content_list)} blocks")


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)

    test_registration()
    test_parse_pdf()
    test_parse_image()

    print("\nAll Ollama OCR parser tests passed.")


if __name__ == "__main__":
    main()
