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

# Questions and expected answers are based on the EL 4.0 electrolyser HAZOP table.
CASES = [
    {
        "id": "el40_pressure_parameters",
        "question": (
            "What is the EL 4.0 maximum output pressure, and what "
            "water input pressure range does it require?"
        ),
        "expected_fact": (
            "Output pressure: up to 35 barg. "
            "Water input pressure: 1–4 barg. "
            "These are different parameters."
        ),
        "expected_entity": "EL 4.0 :: Output Pressure",
        "expected_relationship_keyword": "has_limit",
    },
    {
        "id": "el40_power_consumption",
        "question": (
            "What are the EL 4.0 operative and peak power consumptions?"
        ),
        "expected_fact": (
            "Operative: 2.4 kW, beginning of life. Peak: 3 kW."
        ),
        "expected_entity": "EL 4.0 :: Operative Power Consumption",
        "expected_relationship_keyword": "has_parameter",
    },
    {
        "id": "el40_hydrogen_purity",
        "question": (
            "What hydrogen purity does the EL 4.0 provide "
            "at 35 barg and 8 barg?"
        ),
        "expected_fact": (
            "At 25 °C: 99.9% at 35 barg and 98.8% at 8 barg."
        ),
        "expected_entity": "EL 4.0 :: Hydrogen Output Purity",
        "expected_relationship_keyword": "has_parameter",
    },
    {
        "id": "el40_standby_footnote",
        "question": (
            "What does 'standby' mean in the EL 4.0 datasheet, "
            "and what is its standby power consumption?"
        ),
        "expected_fact": (
            "Standby means no hydrogen is being produced and the "
            "auxiliary components are not powered. "
            "Standby power consumption: 0.3 kW."
        ),
        "expected_entity": "EL 4.0 :: Standby Power Consumption",
        "expected_relationship_keyword": "has_parameter",
    },
    {
        "id": "sitrans_accuracy_comparison",
        "question": (
            "What accuracies are specified for SITRANS P420, P320, "
            "P300 and P Compact?"
        ),
        "expected_fact": (
            "P420: 0.04%; P320: 0.065%; P300: 0.075%; "
            "P Compact: 0.2%. The accuracy footnote specifies "
            "conformity error according to EN IEC 62828-1."
        ),
        "expected_entity": "SITRANS P420 :: Accuracy",
        "expected_relationship_keyword": "has_parameter",
    },
    {
        "id": "sitrans_measuring_spans",
        "question": (
            "What measuring spans are listed for "
            "SITRANS P420 and P Compact?"
        ),
        "expected_fact": (
            "P420: 1 mbar to 700 bar. P Compact: 0 bar to 40 bar."
        ),
        "expected_entity": "SITRANS P420 :: Measuring Span",
        "expected_relationship_keyword": "has_limit",
    },
    {
        "id": "sitrans_medium_temperature",
        "question": (
            "What measured-medium temperature ranges are listed "
            "for SITRANS P420 and P300?"
        ),
        "expected_fact": (
            "P420: −40 °C to +100 °C. P300: −40 °C to +200 °C."
        ),
        "expected_entity": "SITRANS P420 :: Measured Medium Temperature",
        "expected_relationship_keyword": "has_limit",
    },
    {
        "id": "sitrans_fieldbus_qualifier",
        "question": (
            "What qualification accompanies FOUNDATION Fieldbus "
            "for SITRANS P420 and P320?"
        ),
        "expected_fact": (
            "The technical table lists FOUNDATION Fieldbus as "
            "'in preparation' for both P420 and P320."
        ),
        "expected_entity": "SITRANS P420",
        "expected_relationship_keyword": "communicates_via",
    },
    {
        "id": "hydrogen_identity",
        "question": (
            "What are the product name, chemical formula and "
            "CAS number in the hydrogen SDS?"
        ),
        "expected_fact": (
            "Product name: Hydrogen, compressed. "
            "Chemical formula: H₂. CAS number: 1333-74-0."
        ),
        "expected_entity": "Hydrogen",
        "expected_relationship_keyword": None,
    },
    {
        "id": "hydrogen_hazard_classification",
        "question": (
            "Which hazard codes and signal word are listed "
            "in section 2 of the hydrogen SDS?"
        ),
        "expected_fact": (
            "H220: Extremely flammable gas. "
            "H280: Contains gas under pressure; may explode if heated. "
            "Signal word: Danger."
        ),
        "expected_entity": "Hydrogen",
        "expected_relationship_keyword": None,
    },
    {
        "id": "hydrogen_sds_purity",
        "question": (
            "What hydrogen purity is specified "
            "in the SDS composition table?"
        ),
        "expected_fact": (
            "≥99.8%, for the product described in the Linde SDS. "
            "This is not the EL 4.0 output specification."
        ),
        "expected_entity": "Hydrogen",
        "expected_relationship_keyword": None,
    },
    {
        "id": "cross_document_purity",
        "question": (
            "Compare the EL 4.0 output purity at 35 barg with "
            "the hydrogen purity in the Linde SDS."
        ),
        "expected_fact": (
            "EL 4.0: 99.9% at 35 barg and 25 °C. "
            "Linde hydrogen SDS: ≥99.8%. "
            "Cite both documents and distinguish their contexts."
        ),
        "expected_entity": "EL 4.0 :: Hydrogen Output Purity",
        "expected_relationship_keyword": "has_parameter",
    },
    {
        "id": "cross_document_pressure_compatibility",
        "question": (
            "State the EL 4.0 maximum output pressure and the "
            "SITRANS P420 measuring span. Do these specifications "
            "alone prove the transmitter is suitable for "
            "installation on the electrolyser?"
        ),
        "expected_fact": (
            "EL 4.0 output pressure: up to 35 barg. "
            "SITRANS P420 measuring span: 1 mbar to 700 bar. "
            "These specifications alone do not establish "
            "compatibility or suitability of a particular "
            "transmitter configuration. Cite both documents."
        ),
        "expected_entity": "SITRANS P420 :: Measuring Span",
        "expected_relationship_keyword": "has_limit",
    },
    {
        "id": "missing_shutdown_setpoint",
        "question": (
            "What high-pressure shutdown setpoint is configured "
            "for the EL 4.0 installation?"
        ),
        "expected_fact": (
            "Not specified in the supplied documents. "
            "Do not interpret the maximum output pressure "
            "of 35 barg as a configured shutdown setpoint."
        ),
        "expected_entity": None,
        "expected_relationship_keyword": None,
    },
    {
        "id": "missing_installed_transmitter",
        "question": (
            "Which SITRANS transmitter is actually installed "
            "on the EL 4.0 in this system?"
        ),
        "expected_fact": (
            "The supplied documents do not identify an installed "
            "transmitter. Do not invent an installation relationship "
            "between a SITRANS model and the EL 4.0."
        ),
        "expected_entity": None,
        "expected_relationship_keyword": None,
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
    parser.add_argument("--top-k", type=int, default=5) # Retrieval limit
    parser.add_argument("--answers", action="store_true", # Enable answer generation
                        help="Generate answers from the exact captured contexts (extra LLM calls).") 
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    return args



"""
Checks for the graph, chunk store, and three vector files.
Checks that the chunk store contains data. This prevents accidentally testing an empty index because the folder path was mistyped.

Returns the number of stored chunks.
"""
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

"""
Configure each search
"""
def retrieval_params(mode, top_k):
    fields = inspect.signature(QueryParam).parameters
    for required in ("mode", "only_need_context", "top_k"): # mode "naive" or "local" (KG); only_need_context True -> return only evidence instead of final answer
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

    # Connect the models and storage
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
        llm_model_max_async=1, # limits simultaneous LLM calls
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
        for case in CASES: # Run retrieval test through question loops
            for mode in ("naive", "local"):
                result = {**case, "mode": mode}
                report["results"].append(result)
                print(f"\n{'=' * 80}\n{case['id']} | {mode}\n{case['question']}")
                print(f"Expected fact (manual review): {case['expected_fact']}")
                start = perf_counter()
                try:
                    context = await rag.aquery( # Retrieve evidences
                        case["question"], param=retrieval_params(mode, args.top_k)
                    )
                    result["retrieval_seconds"] = round(perf_counter() - start, 3)
                    if not isinstance(context, str):
                        raise TypeError(f"Expected string context; got {type(context).__name__}")
                    result["context"] = context
                    print(f"\nRETRIEVED CONTEXT ({result['retrieval_seconds']}s):\n{context}")
                    save_report(report_path, report)

                    if args.answers and context.strip(): # if "--answer", then put the evidences and prompt into LLM to generate an answer
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
    raise SystemExit(asyncio.run(main(parse_args()))) # Start asynchronous program
