from pathlib import Path
import networkx as nx


GRAPH_PATH = Path(
    "rag_storage_table_test_4/graph_chunk_entity_relation.graphml"
)

SEARCH_TERMS = [
    "500",
    "NL/h",
    "35 barg",
    "4.8",
    "2.4 kW",
    "3 kW",
    "420",
    "1 – 4",
    "45",
    "42 kg",
]

def main():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(GRAPH_PATH)

    graph = nx.read_graphml(GRAPH_PATH)

    print("\n=== KG SUMMARY ===")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    print("\n=== NODES ===")

    for node_id, data in graph.nodes(data=True):
        print(f"\nNode: {node_id}")

        for key, value in data.items():
            print(f"  {key}: {value}")

    print("\n=== EDGES ===")

    for source, target, data in graph.edges(data=True):
        print(f"\n{source} -> {target}")

        for key, value in data.items():
            print(f"  {key}: {value}")

    print("\n=== ENGINEERING VALUE SEARCH ===")

    for term in SEARCH_TERMS:
        matches = []

        for node_id, data in graph.nodes(data=True):
            text = f"{node_id} {data}"
            if term.lower() in text.lower():
                matches.append(("NODE", node_id, data))

        for source, target, data in graph.edges(data=True):
            text = f"{source} {target} {data}"
            if term.lower() in text.lower():
                matches.append(
                    ("EDGE", f"{source} -> {target}", data)
                )

        print(f"\n{term}: {len(matches)} match(es)")

        for match_type, identifier, data in matches:
            print(f"  [{match_type}] {identifier}")
            print(f"    {data}")

if __name__ == "__main__":
    main()