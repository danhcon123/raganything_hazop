# UPDATE Working step

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