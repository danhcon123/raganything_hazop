import asyncio
import json
from functools import partial
from pathlib import Path

from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.prompt import PROMPTS
from lightrag.utils import EmbeddingFunc

from raganything import RAGAnything, RAGAnythingConfig


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
# Engineering / HAZOP entity schema
# ---------------------------------------------------------------------

ENGINEERING_ENTITY_TYPES = [
    "Equipment",
    "Component",

    # DEXPI / P&ID
    "Pipe",
    "Pipeline",
    "Valve",
    "Instrument",
    "Nozzle",
    "Connection",
    "PlantArea",

    # Process / properties
    "Substance",
    "Process",
    "OperatingParameter",
    "OperatingLimit",
    "Utility",

    # Control / safety
    "ControlSystem",
    "ProtectionSystem",
    "SafetyRequirement",

    # Reference
    "Standard",
    "Organization",
    "Document",
]


# ---------------------------------------------------------------------
# Engineering-specific extraction instructions
# ---------------------------------------------------------------------

ENGINEERING_EXTRACTION_RULES = """
### Engineering / HAZOP extraction requirements

The input may contain engineering datasheets, technical tables,
equipment specifications, operating conditions, and HAZOP-relevant
information.

Follow these additional rules when extracting entities and relationships.

#### 1. Engineering parameters are important entities

Preserve engineering parameters whenever the source contains a
numerical value, range, unit, limit, nominal value, maximum value,
minimum value, operating value, capacity, consumption, or environmental
condition.

Examples include:

- pressure
- temperature
- flow rate
- production rate
- purity
- power consumption
- voltage
- frequency
- water consumption
- conductivity
- humidity
- dimensions
- mass / weight
- capacity
- IP rating

Do not omit an engineering parameter merely because its value is
numerical.

Whenever practical, represent each important parameter as its own
OperatingParameter or OperatingLimit entity instead of hiding all
technical values only inside the equipment description.


#### 2. Preserve exact values and units

Preserve each parameter's value, unit, range and qualifiers
exactly as provided in the source.

Preserve qualifiers such as:

- up to
- approximately
- minimum
- maximum
- nominal
- operative
- peak
- beginning of life
- end of life

Preserve associated conditions, such as temperature, pressure
and equipment operating state.

Include these facts in the entity and relationship descriptions.
Do not substitute example values or values from other equipment.


#### 3. Canonical equipment and parameter naming

Choose the equipment name from the source being processed.
Preserve model identifiers and equipment tags.

Merge aliases only when the source clearly identifies the same
equipment. Do not merge different models or equipment tags.

For equipment-specific parameters and limits, include the owning
equipment name in the entity name, for example:
"<equipment name> :: Output Pressure".

Use the same name consistently for repeated references to that
equipment and parameter.


#### 4. Engineering relationships

Create meaningful engineering relationships between extracted
entities.

Use concise relationship keywords such as:

- has_parameter
- has_limit
- produces
- consumes
- requires
- uses
- controlled_by
- monitored_by
- conforms_to
- communicates_via
- protected_by
- located_in
- connected_to
- up_stream_of
- down_stream_of
- contains
- part_of
- measures
- controls
- feeds
- discharges_to

For equipment parameters and limit, connect the parameter or limit to its owning equipment whenever
this relationship is explicitly supported by the source.

For example:

EL 4.0
and
Operative Power Consumption

should have relationship keyword:

has_parameter

and the relationship description should explain that the EL 4.0 has
an operative power consumption of 2.4 kW at beginning of life.

For process topology, use connected_to when a physical connection is explicitly represented.

Use upstream_of or downstream_of only when the source explicitly provides
flow direction or source/target semantics, or other information that clearly establishs directionality.

Do not infer process from graphical order, naming, or proximity alone.

#### 5. Parameter ownership

Every extracted operating parameter or operating limit should be
connected to the equipment or component to which it applies whenever
that ownership is clear from the source.

Avoid isolated parameter entities.


#### 6. Safety and HAZOP relevance

Prioritize facts useful for understanding:

- normal operating conditions
- allowable operating ranges
- maximum and minimum limits
- process inputs and outputs
- utilities
- pressure
- temperature
- flow
- composition / purity
- energy consumption
- control and monitoring
- protective functions
- standards and safety requirements


#### 7. Do not invent information

Only extract relationships and values supported by the input.

Do not infer:

- design pressure from operating pressure
- safety limits from normal operating values
- failure modes that are not stated
- causes or consequences that are not stated

The KG extraction stage should preserve source facts, not perform
HAZOP inference.


#### 8. Follow the original LightRAG output format exactly

These engineering requirements modify WHAT information should be
extracted.

They do NOT modify LightRAG's required entity and relationship output
format, delimiters, field count, or completion delimiter.
"""


# ---------------------------------------------------------------------
# Patch LightRAG extraction prompt
# ---------------------------------------------------------------------

def configure_engineering_extraction_prompt():
    """
    Extend LightRAG's existing extraction prompt without replacing
    its required formatting instructions.
    """

    marker = "### Engineering / HAZOP extraction requirements"

    base_prompt = PROMPTS["entity_extraction_system_prompt"]

    # Prevent accidental double-patching if this function is called twice.
    if marker in base_prompt:
        return

    instruction_marker = "---Instructions---"

    if instruction_marker not in base_prompt:
        raise RuntimeError(
            "Could not find LightRAG instruction marker in "
            "entity_extraction_system_prompt"
        )

    PROMPTS["entity_extraction_system_prompt"] = base_prompt.replace(
        instruction_marker,
        instruction_marker + "\n\n" + ENGINEERING_EXTRACTION_RULES,
        1,
    )


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