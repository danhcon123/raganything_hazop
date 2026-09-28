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
    "rag_storage_phase2/kv_store_parse_cache.json"
)

# IMPORTANT:
# Use a NEW directory so the result is not mixed with the previous
# 18-node / 27-edge baseline.
TEST_STORAGE = "./rag_storage_table_test_2"


# ---------------------------------------------------------------------
# Engineering / HAZOP entity schema
# ---------------------------------------------------------------------

ENGINEERING_ENTITY_TYPES = [
    "Equipment",
    "Component",
    "Substance",
    "Process",
    "OperatingParameter",
    "OperatingLimit",
    "Utility",
    "ControlSystem",
    "ProtectionSystem",
    "SafetyRequirement",
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

Preserve values and units exactly as stated in the source.

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

Examples:

"Operative power consumption | 2.4 kW, beginning of life"

should produce an entity similar to:

Operative Power Consumption

with a description containing exactly:

"The EL 4.0 operative power consumption is 2.4 kW,
beginning of life."


"Water input pressure range | 1 – 4 barg"

should produce:

Water Input Pressure Range

with a description containing the range "1 – 4 barg".


#### 3. Canonical equipment naming

Use one canonical name for the same real-world equipment.

For this document:

- "EL40"
- "EL 4.0"
- "AEM Electrolyser EL 4.0"
- "AEM Electrolyser EL40"

all refer to the same Enapter electrolyser.

Use the canonical entity name:

EL 4.0

Do not create separate entities solely because an abbreviation,
spacing difference, capitalization difference, or longer product name
is used.


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

For example:

EL 4.0
and
Operative Power Consumption

should have relationship keyword:

has_parameter

and the relationship description should explain that the EL 4.0 has
an operative power consumption of 2.4 kW at beginning of life.


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

    config = RAGAnythingConfig(
        parser="mineru",
        parser_output_dir="./output",

        # Fresh KG for this experiment.
        working_dir=TEST_STORAGE,

        # Isolate table processing.
        enable_image_processing=False,
        enable_table_processing=True,
        enable_equation_processing=False,
    )

    embedding_func = EmbeddingFunc(
        embedding_dim=4096,
        max_token_size=8192,
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
            "llm_model_name": "qwen3.8:latest",

            "llm_model_kwargs": {
                "host": "http://localhost:11434",
                "options": {
                    "num_ctx": 8192,
                },
                "timeout": 300,
            },

            # LightRAG v1.4.16 reads these during KG extraction.
            "addon_params": {
                "language": "English",
                "entity_types": ENGINEERING_ENTITY_TYPES,
            },
        },
    )

    table = load_table_from_cache()

    print("\n=== TABLE LOADED FROM PARSE CACHE ===")
    print("type:", table.get("type"))
    print("page_idx:", table.get("page_idx"))
    print("table_type:", table.get("table_type"))

    print("\nTABLE BODY:")
    print(table.get("table_body"))

    # Keep the contextual text very small.
    #
    # Its purpose is only to tell the extractor what equipment the
    # following table belongs to.
    content_list = [
        {
            "type": "text",
            "text": (
                "The equipment described in this document is the "
                "Enapter EL 4.0 AEM electrolyser. "
                "The following table contains technical specifications "
                "for the EL 4.0."
            ),
            "page_idx": 1,
        },
        table,
    ]

    print("\n=== STARTING HAZOP TABLE INSERTION ===")

    await rag.insert_content_list(
        content_list=content_list,
        file_path="Enapter_Datasheet_EL40_EN.pdf",

        # New ID for this experiment.
        doc_id="doc-enapter-el40-table-hazop",

        display_stats=True,

        # Avoid the old multimodal_processed status preventing processing.
        force_multimodal_reprocess=True,
    )

    await rag.finalize_storages()

    print("\n=== HAZOP TABLE TEST COMPLETE ===")
    print(f"Storage: {TEST_STORAGE}")


if __name__ == "__main__":
    asyncio.run(main())