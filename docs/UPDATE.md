# UPDATE Working step

# 19.10.2026


# 15.10.2026
### What we did
- **MinerU workaround on Jetson:** MinerU needs PyTorch with CUDA, which doesn't work on Jetson (the `.venv` has x86 cu128 wheels, so `cuda.is_available()` is False). Replaced it with an Ollama OCR parser.
  - `src/raganything_hazop/parsers/ollama_ocr.py`: `OllamaOCRParser`, registered as `parser="ollama_ocr"`.
  - Each PDF page is rendered with PyMuPDF (150 dpi) and sent to `glm-ocr` twice: `Text Recognition:` and `Table Recognition: Keep line breaks inside cells as <br>.`
  - Output is a normal RAG-Anything `content_list`. Text lines that are already in a table are removed (no duplicates), and the table keeps its reading position.
  - Env vars: `OLLAMA_OCR_MODEL`, `OLLAMA_HOST`.
- **`hashing_kv` bug fixed** (`src/raganything_hazop/llm.py`, `make_ollama_llm_func`): binds the model, host and timeout up front, so RAG-Anything's table processor gets a real table description instead of the fallback.
- **Extraction prompt moved** to `src/raganything_hazop/extraction_prompt.py` (20 engineering entity types plus rules).
- **Tests reorganised:** `tests/mineru/` (old MinerU tests) and `tests/ollama_ocr_parser/` (phase 1 parser test, phase 2 KG test). `phase3_retrieval_test.py` gained a `--cases` filter.

### Result
- **Phase 1 (parser):** passed (about 2 min on EL40).
- **Phase 2 (KG, 3 PDFs: EL40, SITRANS P, Hydrogen SDS):** passed. 15 chunks, 631 nodes / 1,228 edges, `output_stores/rag_storage_ollama_ocr_3docs`. Took about 70 min (≈25 OCR + 50 KG); deepseek-v4.1-flash:cloud with `max_async=1`.
- **Phase 3 (retrieval):** **29/30 correct** (naive + local), 5.5 min. Report: `reports/retrieval/retrieval_20261015T110322_239166Z.json`.
  - The only failure was `cross_document_purity` in **local** mode: it missed the SDS purity of ≥99.8% even though that value is in the KG. Likely cause: top-k 5 was filled by EL 4.0 nodes.
  - Trick (missing-value) questions: no hallucinated values.

### Problems and solutions
- **Multi-line cells glued together** ("25 °C8 barg") → ask glm-ocr to keep `pa<br>` inside cells.
- **Table values duplicated in text** → drop text lines that match table cells.
- **`KeyError: 'hashing_kv'`** made the table description fall back to raw HTML, and the LLM invented a "Hydrogen Generator" → custom LLM wrapper.
- **Image input fell back to MinerU** → implemented `parse_image`.

### Open points
- Duplicate entities (`SITRANS P Compact` / `P COMPACT`; Accuracy / Measuring Accuracy) → entity merging.
- Local mode on cross-document questions → try `hybrid`/`mix` or a higher top-k.
- Speed: skip OCR on pages with a text layer; `max_async` > 1; don't run other GPU jobs in parallel (they slowed the build about 3×).
- Unload glm-ocr after parsing (`keep_alive=0`); set `num_ctx` for local LLMs (gemma ran with a 262k context, 110–260 s per answer).
- Deferred: deepseek vs gemma4:12b, with/without the engineering prompt. Next big step: DEXPI.

### Lessons learned
- Scanned PDFs (EL40, Hydrogen SDS) have no text layer, so the SITRANS text layer is useful for checking OCR values.
- Phase 3 results are only valid on a **fully built** index (earlier reports from a partial index are invalid).
- Saying "not in the evidence" is better than inventing a value, but it still points to a retrieval gap.

# 29.08.2026
### What we did

- Verified that MinerU parsed the datasheet’s technical table.
- Inserted the table directly and used engineering entity types and extraction instructions.
- Built a KG with **36 nodes and 72 edges**, including equipment parameters, limits, and table membership links.
- Confirmed that **EL 4.0 and EL40 share one entity**.
- Tested vector retrieval (`naive`) and KG retrieval (`local`), then answer generation. Both modes correctly answered pressure, power consumption, and water input pressure questions.

### Problems and solutions

- **Original ingestion missed table evidence:** the parsed table existed, but the initial LightRAG index contained only text. Direct table insertion isolated and bypassed this issue.
- **Extraction was incomplete and local inference failed:** relationships were missing in earlier experiments, and later local runs encountered LLM errors. We adjusted generation/context settings and timeouts, then switched to **Ollama Cloud for the LLM**, keeping **local embeddings**. The cloud run successfully stored engineering relationships.
- **Table processing raised `'hashing_kv'`:** the successful run used fallback table content. This bug remains unresolved; the fallback also duplicates table HTML.

### Current state and next step

The **isolated table → KG/vector storage → retrieval → answer** pipeline works for the tested questions. Next, validate **full-document ingestion in a fresh directory**, then expand to multiple documents.

### Lessons learned

- Check each stage separately: **parsed content → stored chunks → KG facts → retrieved evidence → answers**.
- A “test complete” message does not prove extraction succeeded.
- Correct answers on one small table establish basic functionality, not retrieval quality at scale.
- Distinguish a successful workaround from a resolved root cause.

# 28.08.2026

# 27.09.2026
Running the whole pipeline from .pdf to KG:  
- KG created successfully
- Run the Pipeline with Qwen embedding and qwen3.8  
- Deactivate Image -> no values got extracted into the KG  

For tomorrow, the key resume point is:  
Semantic text KG works. Next: fix table processing so the engineering specifications (35 barg, 500 NL/h, 2.4 kW, etc.) enter the KG. Then move to VLM/image processing, component hierarchy, and DEXPI topology.

After that:

Table KG
Formula/equation KG
Image + related text using qwen3-vl:8b
Component hierarchy KG
P&ID image / DEXPI topology
Hybrid retrieval across the KGs
HAZOP-specific query evaluation

That is the logical progression.

## 25.09.2026
Start with embeddings

## 23.09.2026

Start with phase 1

Phase 1 Goal: 
```
Can RAG-Anything correctly transform engineering documents into structured multimodal knowledge?
```

The last confirmed achievement on the x86-64 architecture device was that we successfully completed the first real document parsing pipeline with MinerU and produced usable Markdown output, meaning the parser stage was working.

Current state:
✅ Completed on x86-64 machine

Hardware / environment  
    - Architecture: x86-64 Windows desktop  
    - GPU: RTX 4070 Ti SUPER  
    - Python: 3.11.9  
    - PyTorch: 2.11.0 + CUDA 12.8  
    - MinerU: 4.0.  

MinerU pipeline  
    - Successfully parsed a 4-page PDF  
    - Output generated as Markdown  


The parser successfully handled:  
    - text extraction  
    - document structure  
    - table extraction  
    - image blocks  