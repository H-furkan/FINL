"""Generate optimized knowledge graph images for the website."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from rag.data import build_condition_index, build_drug_index, load_data

# Colors matching the website theme
COLOR_DRUG = "#2563eb"
COLOR_CONDITION = "#7c3aed"
COLOR_SIDE_EFFECT = "#f59e0b"
COLOR_EDGE_TREATS = "#2563eb"
COLOR_EDGE_CAUSES = "#d97706"
BG_COLOR = "#ffffff"


def generate_overview(drug_index: dict) -> None:
    """Full knowledge graph, web-optimized."""
    G = nx.Graph()

    for drug_name, info in drug_index.items():
        G.add_node(drug_name, node_type="drug")
        for cond in info["conditions"]:
            G.add_node(cond, node_type="condition")
            G.add_edge(drug_name, cond, edge_type="treats")
        for se in info["side_effects"]:
            G.add_node(se, node_type="side_effect")
            G.add_edge(drug_name, se, edge_type="causes")

    node_colors = []
    node_sizes = []
    for node in G.nodes():
        ntype = G.nodes[node]["node_type"]
        if ntype == "drug":
            node_colors.append(COLOR_DRUG)
            node_sizes.append(600)
        elif ntype == "condition":
            node_colors.append(COLOR_CONDITION)
            node_sizes.append(350)
        else:
            node_colors.append(COLOR_SIDE_EFFECT)
            node_sizes.append(200)

    edge_colors = [
        COLOR_EDGE_TREATS if G.edges[e]["edge_type"] == "treats" else COLOR_EDGE_CAUSES
        for e in G.edges()
    ]

    fig, ax = plt.subplots(1, 1, figsize=(20, 16), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    pos = nx.spring_layout(G, k=0.7, iterations=150, seed=42)

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.15, width=0.6, ax=ax)
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes,
        alpha=0.9, edgecolors="white", linewidths=0.5, ax=ax,
    )

    # Label only drugs
    drug_labels = {n: n.replace("Drug ", "") for n in G.nodes() if G.nodes[n]["node_type"] == "drug"}
    nx.draw_networkx_labels(G, pos, labels=drug_labels, font_size=7, font_color="white", font_weight="bold", ax=ax)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_DRUG, markersize=12, label="Drug (50)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CONDITION, markersize=10, label="Condition (41)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SIDE_EFFECT, markersize=8, label="Side Effect (68)"),
        plt.Line2D([0], [0], color=COLOR_EDGE_TREATS, linewidth=2.5, label="treats"),
        plt.Line2D([0], [0], color=COLOR_EDGE_CAUSES, linewidth=2.5, label="causes"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11, facecolor="white", edgecolor="#e2e8f0")

    ax.set_title("Drug Knowledge Graph — 50 Drugs, 41 Conditions, 68 Side Effects", fontsize=16, pad=20, color="#1e293b")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("docs/kg_overview.png", dpi=120, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close()
    print("Saved docs/kg_overview.png")


def generate_focus(drug_index: dict, drug_name: str = "Drug F") -> None:
    """Single drug focus view showing the sibling linking concept."""
    info = drug_index[drug_name]
    G = nx.Graph()
    G.add_node(drug_name, node_type="drug")

    for cond in info["conditions"]:
        G.add_node(cond, node_type="condition")
        G.add_edge(drug_name, cond, edge_type="treats")
    for se in info["side_effects"]:
        G.add_node(se, node_type="side_effect")
        G.add_edge(drug_name, se, edge_type="causes")

    node_colors = []
    node_sizes = []
    for node in G.nodes():
        ntype = G.nodes[node]["node_type"]
        if ntype == "drug":
            node_colors.append(COLOR_DRUG)
            node_sizes.append(3000)
        elif ntype == "condition":
            node_colors.append(COLOR_CONDITION)
            node_sizes.append(1800)
        else:
            node_colors.append(COLOR_SIDE_EFFECT)
            node_sizes.append(1200)

    edge_colors = [
        COLOR_EDGE_TREATS if G.edges[e]["edge_type"] == "treats" else COLOR_EDGE_CAUSES
        for e in G.edges()
    ]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    pos = nx.spring_layout(G, k=3, seed=42)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.5, width=2.5, ax=ax)
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes,
        alpha=0.9, edgecolors="white", linewidths=2, ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=11, font_color="white", font_weight="bold", ax=ax)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_DRUG, markersize=14, label="Drug"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_CONDITION, markersize=12, label="Condition"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SIDE_EFFECT, markersize=10, label="Side Effect"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11, facecolor="white", edgecolor="#e2e8f0")

    ax.set_title(f"{drug_name} — Usage & Side Effects (Sibling Linking Demo)", fontsize=14, pad=15, color="#1e293b")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("docs/kg_focus.png", dpi=120, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close()
    print("Saved docs/kg_focus.png")


def generate_condition_cluster(drug_index: dict, condition_index: dict) -> None:
    """Infection drugs cluster -- shows multi-drug retrieval."""
    condition = "infections"
    G = nx.Graph()
    G.add_node(condition, node_type="condition")

    for drug_name in condition_index[condition]:
        info = drug_index[drug_name]
        G.add_node(drug_name, node_type="drug")
        G.add_edge(drug_name, condition, edge_type="treats")
        for se in info["side_effects"]:
            G.add_node(se, node_type="side_effect")
            G.add_edge(drug_name, se, edge_type="causes")

    node_colors = []
    node_sizes = []
    for node in G.nodes():
        ntype = G.nodes[node]["node_type"]
        if ntype == "drug":
            node_colors.append(COLOR_DRUG)
            node_sizes.append(2000)
        elif ntype == "condition":
            node_colors.append(COLOR_CONDITION)
            node_sizes.append(2800)
        else:
            node_colors.append(COLOR_SIDE_EFFECT)
            node_sizes.append(800)

    edge_colors = [
        COLOR_EDGE_TREATS if G.edges[e]["edge_type"] == "treats" else COLOR_EDGE_CAUSES
        for e in G.edges()
    ]

    fig, ax = plt.subplots(1, 1, figsize=(14, 10), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    pos = nx.spring_layout(G, k=2.5, seed=42)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.4, width=1.5, ax=ax)
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes,
        alpha=0.9, edgecolors="white", linewidths=1.5, ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="white", font_weight="bold", ax=ax)

    ax.set_title("Condition Cluster: Infection Drugs & Their Side Effects", fontsize=14, pad=15, color="#1e293b")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("docs/kg_cluster.png", dpi=120, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close()
    print("Saved docs/kg_cluster.png")


if __name__ == "__main__":
    df = load_data()
    drug_index = build_drug_index(df)
    condition_index = build_condition_index(drug_index)

    generate_overview(drug_index)
    generate_focus(drug_index, "Drug F")
    generate_condition_cluster(drug_index, condition_index)
