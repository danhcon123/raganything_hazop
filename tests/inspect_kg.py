from pathlib import Path
import networkx as nx


GRAPH_PATH = Path(
    "rag_storage_phase2/graph_chunk_entity_relation.graphml"
)


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

    print("\n=== RELATIONSHIPS ===")

    for source, target, data in graph.edges(data=True):
        print(f"\n{source} -> {target}")

        for key, value in data.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()