from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


GRAPH_PATH = Path(
    "rag_storage_phase2/graph_chunk_entity_relation.graphml"
)


def main():
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(GRAPH_PATH)

    graph = nx.read_graphml(GRAPH_PATH)

    plt.figure(figsize=(14, 10))

    pos = nx.spring_layout(
        graph,
        seed=42,
        k=1.5,
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=2200,
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowsize=20,
        width=1.5,
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=9,
        font_weight="bold",
    )

    edge_labels = {}

    for source, target, data in graph.edges(data=True):
        description = data.get("description")

        if description:
            edge_labels[(source, target)] = str(description)

    if edge_labels:
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            font_size=7,
        )

    plt.title(
        "RAG-Anything / LightRAG Knowledge Graph\n"
        "Enapter EL 4.0 Datasheet"
    )

    plt.axis("off")

    output = Path("rag_storage_phase2/kg_visualization.png")

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"Saved visualization to: {output}")

    plt.show()


if __name__ == "__main__":
    main()