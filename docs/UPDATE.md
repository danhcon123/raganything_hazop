# UPDATE Working step

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