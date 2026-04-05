"""Visualize the drug knowledge graph as a colored network diagram."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from rag.data import build_drug_index, load_data

# -- Color palette --
COLOR_DRUG = "#4ECDC4"
COLOR_CONDITION = "#FF6B6B"
COLOR_SIDE_EFFECT = "#FFE66D"
COLOR_EDGE_TREATS = "#2196F3"
COLOR_EDGE_CAUSES = "#FF7043"
BG_COLOR = "#ffffff"
TEXT_COLOR = "#222222"


def build_kg_graph(drug_index: dict, max_drugs: int | None = None) -> nx.Graph:
    """Build a networkx graph from the drug index."""
    G = nx.Graph()

    drugs = sorted(drug_index.keys(), key=lambda d: (len(d), d))
    if max_drugs:
        drugs = drugs[:max_drugs]

    for drug_name in drugs:
        info = drug_index[drug_name]
        G.add_node(drug_name, node_type="drug")

        for cond in info["conditions"]:
            G.add_node(cond, node_type="condition")
            G.add_edge(drug_name, cond, edge_type="treats")

        for se in info["side_effects"]:
            G.add_node(se, node_type="side_effect")
            G.add_edge(drug_name, se, edge_type="causes")

    return G


def visualize_full(drug_index: dict, output_path: str = "kg_full.png") -> None:
    """Visualize the full knowledge graph (all 50 drugs)."""
    G = build_kg_graph(drug_index)

    node_colors = []
    node_sizes = []
    for node in G.nodes():
        ntype = G.nodes[node]["node_type"]
        if ntype == "drug":
            node_colors.append(COLOR_DRUG)
            node_sizes.append(1800)
        elif ntype == "condition":
            node_colors.append(COLOR_CONDITION)
            node_sizes.append(1200)
        else:
            node_colors.append(COLOR_SIDE_EFFECT)
            node_sizes.append(800)

    edge_colors = [
        COLOR_EDGE_TREATS if G.edges[e]["edge_type"] == "treats" else COLOR_EDGE_CAUSES
        for e in G.edges()
    ]

    fig, ax = plt.subplots(1, 1, figsize=(42, 42), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    pos = nx.spring_layout(G, k=0.8, iterations=200, seed=42)

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.25, width=1.0, ax=ax)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.95,
        edgecolors="#333333",
        linewidths=0.8,
        ax=ax,
    )

    # Label all nodes
    drug_nodes = [n for n in G.nodes() if G.nodes[n]["node_type"] == "drug"]
    drug_labels = {n: n.replace("Drug ", "") for n in drug_nodes}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=drug_labels,
        font_size=13,
        font_color="white",
        font_weight="bold",
        ax=ax,
    )

    condition_nodes = [n for n in G.nodes() if G.nodes[n]["node_type"] == "condition"]
    condition_labels = {n: n for n in condition_nodes}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=condition_labels,
        font_size=10,
        font_color="white",
        font_weight="bold",
        ax=ax,
    )

    se_nodes = [n for n in G.nodes() if G.nodes[n]["node_type"] == "side_effect"]
    se_labels = {n: n for n in se_nodes}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=se_labels,
        font_size=8,
        font_color="#222222",
        font_weight="normal",
        ax=ax,
    )

    ax.set_title(
        "Drug Knowledge Graph — 50 Drugs, 41 Conditions, 68 Side Effects",
        fontsize=22,
        color=TEXT_COLOR,
        pad=25,
    )

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLOR_DRUG,
            markersize=16,
            label="Drug",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLOR_CONDITION,
            markersize=14,
            label="Condition",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLOR_SIDE_EFFECT,
            markersize=12,
            label="Side Effect",
        ),
        plt.Line2D([0], [0], color=COLOR_EDGE_TREATS, linewidth=3, label="treats"),
        plt.Line2D([0], [0], color=COLOR_EDGE_CAUSES, linewidth=3, label="causes"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=14,
        facecolor="#f5f5f5",
        edgecolor="#cccccc",
        labelcolor=TEXT_COLOR,
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close()
    print(f"Saved full graph → {output_path}")


def visualize_drug_focus(
    drug_index: dict, drug_name: str, output_path: str | None = None
) -> None:
    """Visualize a single drug's neighborhood (its conditions + side effects)."""
    if drug_name not in drug_index:
        print(f"Drug '{drug_name}' not found.")
        return

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
            node_sizes.append(2000)
        elif ntype == "condition":
            node_colors.append(COLOR_CONDITION)
            node_sizes.append(1200)
        else:
            node_colors.append(COLOR_SIDE_EFFECT)
            node_sizes.append(800)

    edge_colors = [
        COLOR_EDGE_TREATS if G.edges[e]["edge_type"] == "treats" else COLOR_EDGE_CAUSES
        for e in G.edges()
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    pos = nx.spring_layout(G, k=3, seed=42)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.6, width=2, ax=ax)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors="white",
        linewidths=1.5,
        ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos, font_size=9, font_color="white", font_weight="bold", ax=ax
    )

    ax.set_title(f"{drug_name}", fontsize=16, color=TEXT_COLOR, pad=15)
    ax.axis("off")
    plt.tight_layout()

    out = output_path or f"kg_{drug_name.replace(' ', '_').lower()}.png"
    plt.savefig(out, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close()
    print(f"Saved focus graph → {out}")


def visualize_condition_cluster(
    drug_index: dict,
    condition_index: dict,
    condition: str,
    output_path: str | None = None,
) -> None:
    """Visualize all drugs that treat a given condition, plus their side effects."""
    if condition not in condition_index:
        print(f"Condition '{condition}' not found.")
        return

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
            node_sizes.append(1500)
        elif ntype == "condition":
            node_colors.append(COLOR_CONDITION)
            node_sizes.append(2000)
        else:
            node_colors.append(COLOR_SIDE_EFFECT)
            node_sizes.append(600)

    edge_colors = [
        COLOR_EDGE_TREATS if G.edges[e]["edge_type"] == "treats" else COLOR_EDGE_CAUSES
        for e in G.edges()
    ]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    pos = nx.spring_layout(G, k=2.5, seed=42)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.5, width=1.5, ax=ax)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors="white",
        linewidths=1.5,
        ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos, font_size=8, font_color="white", font_weight="bold", ax=ax
    )

    ax.set_title(f"Drugs treating: {condition}", fontsize=16, color=TEXT_COLOR, pad=15)
    ax.axis("off")
    plt.tight_layout()

    out = output_path or f"kg_condition_{condition.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close()
    print(f"Saved condition graph → {out}")


if __name__ == "__main__":
    df = load_data()
    drug_index = build_drug_index(df)

    visualize_full(drug_index)
