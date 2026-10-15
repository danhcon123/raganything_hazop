"""
Engineering / HAZOP entity types and extraction rules added to LightRAG's
entity extraction prompt. Parser-independent (MinerU and Ollama OCR).
"""

from lightrag.prompt import PROMPTS


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
