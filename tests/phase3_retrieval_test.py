"""Compare vector and local-KG retrieval from an existing HAZOP index.

From the project root:
    uv run python -m tests.phase3_retrieval_test
    uv run python -m tests.phase3_retrieval_test --answers

No documents are inserted. LightRAG may persist query/LLM caches.
Local KG retrieval can call the cloud LLM for keyword extraction even when
--answers is omitted. Embeddings must match those used to build the index.
"""

import argparse
import asyncio
import inspect
import json
import logging
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc


CASES = [
    {
        "id": "output_pressure",
        "question": "What is the EL 4.0 electrolyser output pressure?",
        "expected_fact": "Up to 35 barg.",
        "expected_entity": "Output Pressure",
        "expected_relationship_keyword": "has_limit",
    },
    {
        "id": "operative_power",
        "question": "What is the EL 4.0 electrolyser operative power consumption?",
        "expected_fact": "2.4 kW, beginning of life.",
        "expected_entity": "Operative Power Consumption",
        "expected_relationship_keyword": "has_parameter",
    },
    {
        "id": "water_input_pressure",
        "question": "What water input pressure range does the EL 4.0 electrolyser require?",
        "expected_fact": "1–4 barg.",
        "expected_entity": "Water Input Pressure Range",
        "expected_relationship_keyword": "has_limit",
    },
]

ANSWER_SYSTEM_PROMPT = """Answer the question using only the supplied retrieved
evidence. Preserve exact values, units, ranges, and qualifiers such as 'up to'
and 'beginning of life'. If the evidence is insufficient, say so. Cite document
names or source identifiers when present in the evidence. Do not invent facts
or citations. Give a short answer."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, default=Path("rag_storage_table_test_4"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/retrieval"))
    parser.add_argument("--model", default="deepseek-v4.1-flash:cloud")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--answers", action="store_true",
                        help="Generate answers from the exact captured contexts (extra LLM calls).")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    return args


def validate_storage(storage):
    # Fail before LightRAG can create an empty index at a mistyped path.
    required = [
        "graph_chunk_entity_relation.graphml",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
    ]
    missing = [name for name in required if not (storage / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing existing-index files in {storage.resolve()}: {missing}. "
            "Use the directory from the successful ingestion run."
        )
    with (storage / "kv_store_text_chunks.json").open(encoding="utf-8") as handle:
        chunks = json.load(handle)
    if not chunks:
        raise ValueError("The saved text-chunk store is empty; retrieval cannot be evaluated.")
    return len(chunks)


def retrieval_params(mode, top_k):
    fields = inspect.signature(QueryParam).parameters
    for required in ("mode", "only_need_context", "top_k"):
        if required not in fields:
            raise RuntimeError(f"Installed QueryParam lacks {required!r}; inspect the installed API.")
    kwargs = {"mode": mode, "only_need_context": True, "top_k": top_k}
    if "chunk_top_k" in fields:
        kwargs["chunk_top_k"] = top_k
    if "enable_rerank" in fields:
        kwargs["enable_rerank"] = False
    return QueryParam(**kwargs)


def save_report(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


async def main(args):
    storage = args.storage.resolve()
    chunk_count = validate_storage(storage)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    report_path = args.output_dir.resolve() / f"retrieval_{stamp}.json"
    report = {
        "created_at_utc": stamp,
        "storage": str(storage),
        "lightrag_version": version("lightrag-hku"),
        "llm_model": args.model,
        "embedding_model": "qwen3-embedding:8b",
        "embedding_dim": 4096,
        "saved_chunk_count": chunk_count,
        "top_k": args.top_k,
        "reranking": False,
        "answers_requested": args.answers,
        "note": "Expected facts are review references, not automatic accuracy scores. "
                "The same small table may supply all answers through vector retrieval.",
        "results": [],
    }

    embedding_func = EmbeddingFunc(
        embedding_dim=4096,
        max_token_size=16384,
        func=partial(ollama_embed.func, embed_model="qwen3-embedding:8b", host=args.host),
        model_name="qwen3-embedding:8b",
    )
    rag = LightRAG(
        working_dir=str(storage),
        llm_model_func=ollama_model_complete,
        llm_model_name=args.model,
        llm_model_kwargs={"host": args.host, "timeout": 600},
        default_llm_timeout=600,
        llm_model_max_async=1,
        embedding_func=embedding_func,
        addon_params={"language": "English"},
    )

    failures = 0
    save_report(report_path, report)
    try:
        await rag.initialize_storages()
        print(f"Existing storage: {storage}")
        print(f"Saved chunks: {chunk_count}; report: {report_path}")

        # Sequential calls suit the current Ollama Cloud concurrency limit.
        for case in CASES:
            for mode in ("naive", "local"):
                result = {**case, "mode": mode}
                report["results"].append(result)
                print(f"\n{'=' * 80}\n{case['id']} | {mode}\n{case['question']}")
                print(f"Expected fact (manual review): {case['expected_fact']}")
                start = perf_counter()
                try:
                    context = await rag.aquery(
                        case["question"], param=retrieval_params(mode, args.top_k)
                    )
                    result["retrieval_seconds"] = round(perf_counter() - start, 3)
                    if not isinstance(context, str):
                        raise TypeError(f"Expected string context; got {type(context).__name__}")
                    result["context"] = context
                    print(f"\nRETRIEVED CONTEXT ({result['retrieval_seconds']}s):\n{context}")
                    save_report(report_path, report)

                    if args.answers and context.strip():
                        start = perf_counter()
                        # Reuse exactly this context rather than retrieve a second time.
                        answer = await rag.llm_model_func(
                            f"Question:\n{case['question']}\n\nRetrieved evidence:\n{context}",
                            system_prompt=ANSWER_SYSTEM_PROMPT,
                        )
                        result["answer_seconds"] = round(perf_counter() - start, 3)
                        if not isinstance(answer, str):
                            raise TypeError(f"Expected string answer; got {type(answer).__name__}")
                        result["answer"] = answer
                        print(f"\nANSWER FROM THIS CONTEXT:\n{answer}")
                except Exception as exc:
                    failures += 1
                    result["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    logging.exception("Failed case=%s mode=%s", case["id"], mode)
                finally:
                    save_report(report_path, report)
    finally:
        try:
            await rag.finalize_storages()
        finally:
            save_report(report_path, report)

    print(f"\nReport saved: {report_path}")
    print(f"Completed cases: {len(report['results'])}; failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(asyncio.run(main(parse_args())))
