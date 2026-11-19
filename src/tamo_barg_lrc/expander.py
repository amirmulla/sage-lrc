"""Bipartite coset graph construction, component extraction, and expander metrics."""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms import cuts

from tamo_barg_lrc.subgroups import field_element_to_int

matplotlib.use("Agg")

__all__ = [
    "compute_expansion_metrics",
    "create_bipartite_coset_graph",
    "get_error_position_subgraph",
    "save_bipartite_graph_plot",
]


def create_bipartite_coset_graph(
    partitions: Sequence[Sequence[Sequence[Any]]],
    evaluation_points_idx: Sequence[Sequence[Sequence[int]]],
    evaluation_points: Sequence[Any],
) -> tuple[nx.MultiGraph, set[int], set[int]]:
    """Construct the bipartite coset intersection multigraph in O(n) time.

    Nodes in partition 0 (left) and partition 1 (right) represent subgroup cosets,
    and each evaluation point shared by a left coset and a right coset induces an
    undirected edge between the corresponding nodes.

    Args:
        partitions: Two-element sequence of coset partitions ``[P_0, P_1]``.
        evaluation_points_idx: Two-element sequence mirroring ``partitions`` with
            canonical indices into ``evaluation_points``.
        evaluation_points: Ordered sequence of code evaluation points.

    Returns:
        A tuple ``(graph, left_nodes, right_nodes)`` where ``graph`` is an
        ``nx.MultiGraph`` annotated with ``partition``, ``coset``,
        ``coset_indices``, and ``bipartite`` on nodes, and ``evalpts`` and
        ``evalpt_idx`` on edges.

    Raises:
        ValueError: If ``partitions`` or ``evaluation_points_idx`` do not have
            length 2, have mismatched shapes, or contain invalid point indices.
    """
    if len(partitions) != 2 or len(evaluation_points_idx) != 2:
        raise ValueError("partitions and evaluation_points_idx must each have length 2.")

    num_eval_pts = len(evaluation_points)
    left_owner: list[int | None] = [None] * num_eval_pts
    right_owner: list[int | None] = [None] * num_eval_pts

    graph = nx.MultiGraph()
    left_nodes: set[int] = set()
    right_nodes: set[int] = set()

    node_id = 0
    for part_idx in (0, 1):
        part_cosets = partitions[part_idx]
        part_indices = evaluation_points_idx[part_idx]
        if len(part_cosets) != len(part_indices):
            raise ValueError(
                f"Partition {part_idx} coset count ({len(part_cosets)}) does not match "
                f"evaluation_points_idx ({len(part_indices)})."
            )

        owner_table = left_owner if part_idx == 0 else right_owner
        target_node_set = left_nodes if part_idx == 0 else right_nodes

        for coset_idx, (coset, coset_pt_indices) in enumerate(
            zip(part_cosets, part_indices, strict=True)
        ):
            coset_list = list(coset)
            coset_idx_list = [int(pt_idx) for pt_idx in coset_pt_indices]
            if len(coset_list) != len(coset_idx_list):
                raise ValueError(
                    f"Coset ({part_idx}, {coset_idx}) size ({len(coset_list)}) does not match "
                    f"index list size ({len(coset_idx_list)})."
                )

            for pt_idx in coset_idx_list:
                if pt_idx < 0 or pt_idx >= num_eval_pts:
                    raise ValueError(
                        f"Evaluation point index {pt_idx} out of range for {num_eval_pts} points."
                    )
                if owner_table[pt_idx] is not None:
                    raise ValueError(
                        f"Duplicate evaluation point index {pt_idx} within partition {part_idx}."
                    )
                owner_table[pt_idx] = node_id

            graph.add_node(
                node_id,
                partition=part_idx,
                coset=coset_list,
                coset_indices=coset_idx_list,
                bipartite=part_idx,
            )
            target_node_set.add(node_id)
            node_id += 1

    for pt_idx, eval_pt in enumerate(evaluation_points):
        u = left_owner[pt_idx]
        v = right_owner[pt_idx]
        if u is not None and v is not None:
            graph.add_edge(u, v, evalpts=eval_pt, evalpt_idx=pt_idx)

    return graph, left_nodes, right_nodes


def _compute_k_edge_components(graph: nx.Graph, k: int) -> Iterable[Iterable[Any]]:
    """Compute k-edge-connected components, supporting both Graph and MultiGraph."""
    if k == 1:
        return nx.connected_components(graph)
    if not graph.is_multigraph():
        return nx.k_edge_components(graph, k=k)

    # NetworkX 3.x marks nx.k_edge_components as @not_implemented_for("multigraph").
    # Subdividing extra parallel edges with degree-2 nodes preserves exact s-t edge
    # connectivity between all original vertices by Menger's theorem.
    orig_nodes = set(graph.nodes())
    simple = nx.Graph()
    simple.add_nodes_from(orig_nodes)
    dummy_idx = 0
    for u, v in graph.edges():
        if u == v:
            continue
        if not simple.has_edge(u, v):
            simple.add_edge(u, v)
        else:
            dummy = ("__subdiv__", dummy_idx)
            while dummy in orig_nodes:
                dummy_idx += 1
                dummy = ("__subdiv__", dummy_idx)
            dummy_idx += 1
            simple.add_edge(u, dummy)
            simple.add_edge(dummy, v)

    components: list[set[Any]] = []
    for comp in nx.k_edge_components(simple, k=k):
        filtered = comp.intersection(orig_nodes)
        if filtered:
            components.append(filtered)
    return components


def get_error_position_subgraph(
    graph: nx.Graph, h_idx: int = 0, edge_connectivity_k: int = 1
) -> list[int]:
    """Return sorted codeword evaluation-point indices for the h_idx-th component.

    Args:
        graph: Bipartite coset graph (``nx.Graph`` or ``nx.MultiGraph``).
        h_idx: Zero-based index into the deterministically sorted list of
            ``edge_connectivity_k``-edge-connected components.
        edge_connectivity_k: Desired edge connectivity ``k >= 1``. Uses the
            O(|V| + |E|) ``nx.connected_components`` fast-path when ``k == 1``.

    Returns:
        Sorted list of unique integer codeword positions belonging to the selected
        component.

    Raises:
        ValueError: If ``edge_connectivity_k < 1``.
        IndexError: If ``h_idx`` is out of range for the discovered components.
        KeyError: If a selected node lacks both ``coset_indices`` and ``coset``.
    """
    if edge_connectivity_k < 1:
        raise ValueError(f"edge_connectivity_k must be >= 1, got {edge_connectivity_k}.")

    raw_components = _compute_k_edge_components(graph, edge_connectivity_k)
    subgraphs = sorted(sorted(comp) for comp in raw_components)

    if not 0 <= h_idx < len(subgraphs):
        raise IndexError(f"Component index {h_idx} out of range for {len(subgraphs)} components")

    component = subgraphs[h_idx]
    target_nodes = [n for n in component if graph.nodes[n].get("bipartite") == 0]
    if not target_nodes:
        target_nodes = component

    indices: list[int] = []
    for node in target_nodes:
        node_data = graph.nodes[node]
        if "coset_indices" in node_data:
            indices.extend(int(idx) for idx in node_data["coset_indices"])
        elif "coset" in node_data:
            indices.extend(field_element_to_int(elem) for elem in node_data["coset"])
        else:
            raise KeyError(f"Node {node!r} is missing both 'coset_indices' and 'coset' attributes.")

    return sorted(set(indices))


def _collapse_to_weighted_simple_graph(graph: nx.Graph) -> nx.Graph:
    """Collapse a MultiGraph's parallel edges into a weighted simple Graph."""
    if not graph.is_multigraph():
        return graph

    simple_g = nx.Graph()
    simple_g.add_nodes_from(graph.nodes())
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        edge_weight = float(data.get("weight", 1.0))
        if simple_g.has_edge(u, v):
            simple_g[u][v]["weight"] += edge_weight
        else:
            simple_g.add_edge(u, v, weight=edge_weight)
    return simple_g


def _compute_adjacency_eigenvalues(graph: nx.Graph) -> list[float]:
    """Compute sorted real adjacency eigenvalues, falling back to NumPy if SciPy is absent."""
    if graph.number_of_nodes() == 0:
        return []

    try:
        raw_evals = nx.adjacency_spectrum(graph)
    except ModuleNotFoundError:
        adj_mat = nx.to_numpy_array(graph, weight="weight", multigraph_weight=sum)
        if graph.is_directed():
            raw_evals = np.linalg.eigvals(adj_mat)
        else:
            raw_evals = np.linalg.eigvalsh(adj_mat)

    return sorted(round(float(val.real), 6) for val in raw_evals)


def _compute_algebraic_connectivity(graph: nx.Graph, is_connected: bool) -> float:
    """Compute algebraic connectivity, collapsing MultiGraphs cleanly across NetworkX 3.x."""
    if graph.number_of_nodes() < 2 or not is_connected:
        return 0.0

    simple_g = _collapse_to_weighted_simple_graph(graph)
    try:
        return float(nx.algebraic_connectivity(simple_g, method="tracemin_lu"))
    except ModuleNotFoundError:
        adj_mat = np.abs(nx.to_numpy_array(simple_g, weight="weight", nonedge=0.0))
        np.fill_diagonal(adj_mat, 0.0)
        lap_mat = np.diag(adj_mat.sum(axis=1)) - adj_mat
        lap_evals = np.linalg.eigvalsh(lap_mat)
        return max(0.0, float(lap_evals[1]))


def compute_expansion_metrics(
    graph: nx.Graph, right_nodes: set[int] | None = None
) -> dict[str, Any]:
    """Compute connectivity, cut expansion, and spectral metrics of a coset graph.

    Args:
        graph: Undirected coset graph (``nx.Graph`` or ``nx.MultiGraph``).
        right_nodes: Optional set of right-partition node IDs. If ``None``,
            inferred from nodes having ``bipartite == 1``.

    Returns:
        Dictionary with keys ``is_connected``, ``num_connected_components``,
        ``edge_expansion``, ``eigenvalues``, ``spectral_gap``, and
        ``algebraic_connectivity``.
    """
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        return {
            "is_connected": False,
            "num_connected_components": 0,
            "edge_expansion": 0.0,
            "eigenvalues": [],
            "spectral_gap": 0.0,
            "algebraic_connectivity": 0.0,
        }

    is_connected = bool(nx.is_connected(graph))
    num_components = int(nx.number_connected_components(graph))

    if right_nodes is None:
        inferred_right = {
            node for node, data in graph.nodes(data=True) if data.get("bipartite") == 1
        }
    else:
        inferred_right = set(right_nodes).intersection(graph.nodes())

    complement_size = num_nodes - len(inferred_right)
    if inferred_right and complement_size > 0:
        edge_exp = float(cuts.edge_expansion(graph, inferred_right))
    else:
        edge_exp = 0.0

    eigenvalues = _compute_adjacency_eigenvalues(graph)
    spectral_gap = float(eigenvalues[-1] - eigenvalues[-2]) if len(eigenvalues) >= 2 else 0.0
    alg_conn = _compute_algebraic_connectivity(graph, is_connected)

    return {
        "is_connected": is_connected,
        "num_connected_components": num_components,
        "edge_expansion": edge_exp,
        "eigenvalues": eigenvalues,
        "spectral_gap": spectral_gap,
        "algebraic_connectivity": alg_conn,
    }


def save_bipartite_graph_plot(
    graph: nx.Graph,
    left_nodes: set[int],
    right_nodes: set[int],
    output_path: str | Path,
) -> None:
    """Render and save a two-layer bipartite graph plot without leaking figures.

    Args:
        graph: Bipartite coset graph to draw.
        left_nodes: Set of node IDs in the top/left layer.
        right_nodes: Set of node IDs in the bottom/right layer.
        output_path: Destination image file path (parent directories are created
            automatically).
    """
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    left_sorted = sorted(left_nodes)
    right_sorted = sorted(right_nodes)
    m, n = len(left_sorted), len(right_sorted)

    pos: dict[Any, tuple[float, float]] = {}
    for idx, node in enumerate(left_sorted):
        pos[node] = (idx - m / 2.0, 1.0)
    for idx, node in enumerate(right_sorted):
        pos[node] = (idx - n / 2.0, 0.0)

    extra_nodes = sorted(set(graph.nodes()) - set(pos))
    for idx, node in enumerate(extra_nodes):
        pos[node] = (idx - len(extra_nodes) / 2.0, 0.5)

    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        nx.draw(graph, pos=pos, ax=ax, with_labels=True, node_size=300, width=0.4)
        fig.savefig(out, bbox_inches="tight")
    finally:
        plt.close(fig)
