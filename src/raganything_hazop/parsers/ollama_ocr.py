"""
MinerU workaround for Jetson: parse documents with an Ollama OCR model (default: glm-ocr).

Each page is rendered to PNG and sent to the model twice:
  - "Text Recognition:"  -> all text on the page (tables flattened to lines)
  - "Table Recognition:" -> tables as HTML (or prose if the page has no table)

The result is a RAG-Anything content_list:
  {"type": "text",  "text": ...,       "page_idx": n}
  {"type": "table", "table_body": ..., "table_caption": [], "table_footnote": [], "page_idx": n}

Table lines are removed from the text output so each value enters the KG only once.

Usage:
    register_ollama_ocr_parser()
    config = RAGAnythingConfig(parser="ollama_ocr", ...)
"""

import html
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ollama
import pymupdf
from raganything.parser import Parser, register_parser


PARSER_NAME = "ollama_ocr"

TEXT_PROMPT = "Text Recognition:"
# Plain "Table Recognition:" glues multi-line cells ("at 25 °C8 barg"); asking for <br> fixes it.
TABLE_PROMPT = "Table Recognition: Keep line breaks inside cells as <br>."

TABLE_RE = re.compile(r"<table.*?</table>", re.DOTALL | re.IGNORECASE)
CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")

# Formats Ollama accepts directly; others are converted to PNG first.
OLLAMA_IMAGE_FORMATS = {".png", ".jpg", ".jpeg"}


def _normalize(text: str) -> str:
    text = html.unescape(text).replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def _table_cell_lines(table_html: str) -> set:
    """All lines inside table cells, normalized (cells split on <br>)."""
    lines = set()
    for cell in CELL_RE.findall(table_html):
        for part in BR_RE.split(cell):
            part = _normalize(TAG_RE.sub("", part))
            if part:
                lines.add(part)
    return lines


def _paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


class OllamaOCRParser(Parser):
    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        dpi: int = 150,
        num_ctx: int = 16384,
    ) -> None:
        super().__init__()
        self.model = model or os.getenv("OLLAMA_OCR_MODEL", "glm-ocr")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.dpi = dpi
        self.num_ctx = num_ctx
        self.client = ollama.Client(host=self.host)

    # ---------- model calls ----------

    def _ocr(self, image_path: Path, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [str(image_path)]}],
            options={"temperature": 0, "num_ctx": self.num_ctx},
        )
        return response["message"]["content"]

    def _ocr_page(self, image_path: Path, page_idx: int) -> List[Dict[str, Any]]:
        self.logger.info(f"[{PARSER_NAME}] OCR page {page_idx}: {image_path.name}")
        text_out = self._ocr(image_path, TEXT_PROMPT)
        tables = TABLE_RE.findall(self._ocr(image_path, TABLE_PROMPT))

        if not tables:
            return [
                {"type": "text", "text": p, "page_idx": page_idx}
                for p in _paragraphs(text_out)
            ]

        # Drop text lines that are already inside a table; remember where the
        # table started so it keeps its reading position between the text blocks.
        cell_lines = set().union(*(_table_cell_lines(t) for t in tables))
        before, after = [], []
        table_seen = False
        for line in text_out.splitlines():
            if _normalize(line) in cell_lines:
                table_seen = True
                continue
            (after if table_seen else before).append(line)

        blocks = [
            {"type": "text", "text": p, "page_idx": page_idx}
            for p in _paragraphs("\n".join(before))
        ]
        blocks += [
            {
                "type": "table",
                "table_body": table,
                "table_caption": [],
                "table_footnote": [],
                "page_idx": page_idx,
            }
            for table in tables
        ]
        blocks += [
            {"type": "text", "text": p, "page_idx": page_idx}
            for p in _paragraphs("\n".join(after))
        ]
        return blocks

    # ---------- helpers ----------

    def _work_dir(self, file_path: Path, output_dir: Optional[str]) -> Path:
        work_dir = Path(output_dir or "./output") / file_path.stem / PARSER_NAME
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def _save_content_list(self, work_dir: Path, file_path: Path, content_list) -> None:
        out = work_dir / f"{file_path.stem}_content_list.json"
        out.write_text(json.dumps(content_list, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info(f"[{PARSER_NAME}] wrote {out}")

    # ---------- Parser API ----------

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

        work_dir = self._work_dir(pdf_path, output_dir)
        content_list: List[Dict[str, Any]] = []
        with pymupdf.open(pdf_path) as doc:
            for page_idx, page in enumerate(doc):
                png = work_dir / f"page_{page_idx}.png"
                page.get_pixmap(dpi=self.dpi).save(png)
                content_list += self._ocr_page(png, page_idx)

        self._save_content_list(work_dir, pdf_path, content_list)
        return content_list

    def parse_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        ext = image_path.suffix.lower()
        if ext not in self.IMAGE_FORMATS:
            raise ValueError(f"Unsupported image format: {ext}")

        work_dir = self._work_dir(image_path, output_dir)
        if ext not in OLLAMA_IMAGE_FORMATS:
            png = work_dir / f"{image_path.stem}.png"
            pymupdf.Pixmap(str(image_path)).save(png)
            image_path = png

        content_list = self._ocr_page(image_path, int(kwargs.get("page_idx", 0)))
        self._save_content_list(work_dir, image_path, content_list)
        return content_list

    def parse_office_doc(
        self,
        doc_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        pdf_path = self.convert_office_to_pdf(doc_path, output_dir)
        return self.parse_pdf(pdf_path, output_dir, lang=lang, **kwargs)

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return self.parse_pdf(file_path, output_dir, method, lang=lang, **kwargs)
        if ext in self.IMAGE_FORMATS:
            return self.parse_image(file_path, output_dir, lang=lang, **kwargs)
        if ext in self.OFFICE_FORMATS:
            return self.parse_office_doc(file_path, output_dir, lang=lang, **kwargs)
        if ext in self.TEXT_FORMATS:
            return self.parse_text_file(file_path, output_dir, lang=lang, **kwargs)
        raise ValueError(f"Unsupported file format: {ext}")

    def check_installation(self) -> bool:
        try:
            self.client.show(self.model)
            return True
        except Exception as e:
            self.logger.warning(f"[{PARSER_NAME}] model '{self.model}' not available at {self.host}: {e}")
            return False


def register_ollama_ocr_parser() -> None:
    """Make parser="ollama_ocr" available to RAGAnythingConfig."""
    register_parser(PARSER_NAME, OllamaOCRParser)
