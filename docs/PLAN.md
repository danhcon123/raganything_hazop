# Working Plan
## Local-first architecture
- LLM → Ollama
- Embeddings → local model
- No OpenAI/API dependency for the core pipeline
# Validation phases
1. Parser + ingestion
2. Embedding generation
3. Vector retrieval
4. Knowledge Graph extraction/retrieval
5. Hybrid retrieval
6. Reranking
7. Final answer generation
8. HAZOP evaluation benchmark
# Initial preferred stack
- MinerU → document understanding
-  BGE-M3 → first embedding candidate
-  bge-reranker → reranking
- Ollama → local LLM serving
- RAG-Anything + LightRAG storage pipeline

