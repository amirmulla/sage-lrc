"""Bipartite coset graph construction, component extraction, and expander metrics."""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from networkx.algorithms import cuts

__all__ = [
    "compute_expansion_metrics",
    "create_bipartite_coset_graph",
    "get_error_position_subgraph",
    "save_bipartite_graph_plot",
]

_DENSE_ADJ_THRESHOLD = 650
_DENSE_LAP_THRESHOLD = 256


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
    field_element_to_int_fn: Any = None
    for node in target_nodes:
        node_data = graph.nodes[node]
        if "coset_indices" in node_data:
            indices.extend(int(idx) for idx in node_data["coset_indices"])
        elif "coset" in node_data:
            if field_element_to_int_fn is None:
                from tamo_barg_lrc.subgroups import field_element_to_int

                field_element_to_int_fn = field_element_to_int
            indices.extend(field_element_to_int_fn(elem) for elem in node_data["coset"])
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


def _try_import_scipy_sparse() -> tuple[Any, Any] | None:
    """Lazily import ``scipy.sparse`` and ``scipy.sparse.linalg`` if installed."""
    try:
        import scipy.sparse
        import scipy.sparse.linalg
    except ModuleNotFoundError:
        return None
    return scipy.sparse, scipy.sparse.linalg


def _extract_bipartite_structure(
    graph: nx.Graph,
) -> tuple[int, int, list[int], list[int], list[float]] | None:
    """Return ``(n_left, n_right, rows, cols, weights)`` if bipartite by attribute.

    Uses the ``bipartite`` node attribute (``0`` vs ``1``) and verifies that every edge
    strictly connects partition ``0`` to partition ``1`` (no self-loops or intra-side edges).
    Returns ``None`` if the graph is not a valid two-sided bipartite graph.
    """
    left_idx: dict[Any, int] = {}
    right_idx: dict[Any, int] = {}
    for node, data in graph.nodes(data=True):
        side = data.get("bipartite")
        if side == 0:
            left_idx[node] = len(left_idx)
        elif side == 1:
            right_idx[node] = len(right_idx)
        else:
            return None

    if not left_idx or not right_idx:
        return None

    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for u, v, data in graph.edges(data=True):
        if u in left_idx and v in right_idx:
            rows.append(left_idx[u])
            cols.append(right_idx[v])
            weights.append(float(data.get("weight", 1.0)))
        elif u in right_idx and v in left_idx:
            rows.append(left_idx[v])
            cols.append(right_idx[u])
            weights.append(float(data.get("weight", 1.0)))
        else:
            return None

    return len(left_idx), len(right_idx), rows, cols, weights


def _regular_degree_if_applicable(graph: nx.Graph) -> float | None:
    """Return common weighted degree ``d`` if ``L = dI - A`` holds, else ``None``."""
    if graph.is_directed() or graph.number_of_nodes() < 2:
        return None

    degrees: dict[Any, float] = {node: 0.0 for node in graph.nodes()}
    for u, v, data in graph.edges(data=True):
        if u == v:
            return None
        w = float(data.get("weight", 1.0))
        if w < 0.0:
            return None
        degrees[u] += w
        degrees[v] += w

    deg_vals = degrees.values()
    deg_min = min(deg_vals)
    deg_max = max(deg_vals)
    if deg_max - deg_min <= 1e-12 * max(1.0, deg_max):
        return deg_max
    return None


def _compute_adjacency_eigenvalues_raw(graph: nx.Graph) -> np.ndarray:
    """Compute sorted full-precision real adjacency eigenvalues.

    Avoids ``nx.adjacency_spectrum`` (which calls nonsymmetric ``scipy.linalg.eigvals``).
    For undirected bipartite graphs, uses the smaller Gram matrix ``B @ B.T`` (or
    ``B.T @ B``) via ``np.linalg.eigvalsh`` when dense, or ``scipy.sparse.linalg.svds``
    when SciPy is installed and ``V > 650``.
    """
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        return np.empty(0, dtype=np.float64)

    if graph.is_directed():
        adj_mat = nx.to_numpy_array(graph, weight="weight", multigraph_weight=sum)
        return np.sort(np.linalg.eigvals(adj_mat).real)

    bipartite_info = _extract_bipartite_structure(graph)
    if bipartite_info is not None:
        n_left, n_right, rows, cols, weights = bipartite_info
        if not rows:
            return np.zeros(num_nodes, dtype=np.float64)

        if num_nodes > _DENSE_ADJ_THRESHOLD and min(n_left, n_right) > 2:
            sp_mods = _try_import_scipy_sparse()
            if sp_mods is not None:
                sp, spla = sp_mods
                b_csr = sp.csr_matrix(
                    (weights, (rows, cols)), shape=(n_left, n_right), dtype=np.float64
                )
                try:
                    svals = spla.svds(b_csr, k=2, which="LM", return_singular_vectors=False)
                    return np.sort(np.asarray(svals, dtype=np.float64))
                except spla.ArpackError:
                    pass

        b_mat = np.zeros((n_left, n_right), dtype=np.float64)
        np.add.at(b_mat, (rows, cols), weights)
        if n_left <= n_right:
            gram = b_mat @ b_mat.T
            diff = n_right - n_left
        else:
            gram = b_mat.T @ b_mat
            diff = n_left - n_right

        gram_evals = np.linalg.eigvalsh(gram)
        max_gram = float(gram_evals[-1]) if gram_evals.size else 0.0
        zero_tol = 1e-12 * max_gram if max_gram > 0.0 else 0.0
        svals = np.sqrt(np.where(gram_evals > zero_tol, gram_evals, 0.0))
        if diff > 0:
            return np.concatenate((-svals[::-1], np.zeros(diff, dtype=np.float64), svals))
        return np.concatenate((-svals[::-1], svals))

    if num_nodes > _DENSE_ADJ_THRESHOLD:
        sp_mods = _try_import_scipy_sparse()
        if sp_mods is not None:
            sp, spla = sp_mods
            node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
            rows_a: list[int] = []
            cols_a: list[int] = []
            vals_a: list[float] = []
            for u, v, data in graph.edges(data=True):
                iu = node_to_idx[u]
                iv = node_to_idx[v]
                w = float(data.get("weight", 1.0))
                rows_a.append(iu)
                cols_a.append(iv)
                vals_a.append(w)
                if iu != iv:
                    rows_a.append(iv)
                    cols_a.append(iu)
                    vals_a.append(w)
            a_csr = sp.csr_matrix(
                (vals_a, (rows_a, cols_a)), shape=(num_nodes, num_nodes), dtype=np.float64
            )
            try:
                ncv = min(num_nodes - 1, 40)
                top_evals = spla.eigsh(a_csr, k=2, which="LA", ncv=ncv, return_eigenvectors=False)
                return np.sort(np.asarray(top_evals, dtype=np.float64))
            except spla.ArpackError:
                pass

    adj_mat = nx.to_numpy_array(graph, weight="weight", multigraph_weight=sum)
    return np.linalg.eigvalsh(adj_mat)


def _compute_adjacency_eigenvalues(graph: nx.Graph) -> list[float]:
    """Compute sorted real adjacency eigenvalues using symmetric/bipartite solvers."""
    raw_evals = _compute_adjacency_eigenvalues_raw(graph)
    return [round(float(val), 6) for val in raw_evals]


def _compute_algebraic_connectivity(graph: nx.Graph, is_connected: bool) -> float:
    """Compute algebraic connectivity (Fiedler value) without SuperLU fill-in."""
    num_nodes = graph.number_of_nodes()
    if num_nodes < 2 or not is_connected:
        return 0.0

    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}

    if num_nodes > _DENSE_LAP_THRESHOLD:
        sp_mods = _try_import_scipy_sparse()
        if sp_mods is not None:
            sp, spla = sp_mods
            rows: list[int] = []
            cols: list[int] = []
            vals: list[float] = []
            for u, v, data in graph.edges(data=True):
                if u == v:
                    continue
                iu = node_to_idx[u]
                iv = node_to_idx[v]
                w = float(data.get("weight", 1.0))
                rows.append(iu)
                cols.append(iv)
                vals.append(w)
                rows.append(iv)
                cols.append(iu)
                vals.append(w)
            a_csr = sp.csr_matrix(
                (vals, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float64
            )
            a_csr.sum_duplicates()
            a_csr.data = np.abs(a_csr.data)
            degrees = np.asarray(a_csr.sum(axis=1)).ravel()
            lap_csr = sp.diags(degrees, format="csr") - a_csr
            try:
                ncv = min(num_nodes - 1, 40)
                lap_evals = spla.eigsh(lap_csr, k=2, which="SA", ncv=ncv, return_eigenvectors=False)
                lap_evals = np.sort(np.asarray(lap_evals, dtype=np.float64))
                return max(0.0, float(lap_evals[1]))
            except spla.ArpackError:
                pass

    # In-place Laplacian assembly halves peak memory compared to np.diag(deg) - adj.
    lap_mat = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        iu = node_to_idx[u]
        iv = node_to_idx[v]
        w = float(data.get("weight", 1.0))
        lap_mat[iu, iv] += w
        lap_mat[iv, iu] += w

    np.abs(lap_mat, out=lap_mat)
    degrees = lap_mat.sum(axis=1)
    lap_mat *= -1.0
    lap_mat[np.arange(num_nodes), np.arange(num_nodes)] = degrees
    lap_evals = np.linalg.eigvalsh(lap_mat)
    return max(0.0, float(lap_evals[1]))


def compute_expansion_metrics(
    graph: nx.Graph,
    right_nodes: set[int] | None = None,
    include_spectrum: bool = True,
) -> dict[str, Any]:
    """Compute connectivity, cut expansion, and spectral metrics of a coset graph.

    Args:
        graph: Undirected coset graph (``nx.Graph`` or ``nx.MultiGraph``).
        right_nodes: Optional set of right-partition node IDs. If ``None``,
            inferred from nodes having ``bipartite == 1``.
        include_spectrum: If ``True`` (default), compute adjacency eigenvalues,
            ``spectral_gap``, and ``algebraic_connectivity``. If ``False``, skip
            the eigensolves and return ``eigenvalues=[]``, ``spectral_gap=0.0``,
            and ``algebraic_connectivity=0.0``.

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

    num_components = int(nx.number_connected_components(graph))
    is_connected = num_components == 1

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

    if not include_spectrum:
        return {
            "is_connected": is_connected,
            "num_connected_components": num_components,
            "edge_expansion": edge_exp,
            "eigenvalues": [],
            "spectral_gap": 0.0,
            "algebraic_connectivity": 0.0,
        }

    raw_evals = _compute_adjacency_eigenvalues_raw(graph)
    eigenvalues = [round(float(val), 6) for val in raw_evals]
    spectral_gap = float(raw_evals[-1] - raw_evals[-2]) if len(raw_evals) >= 2 else 0.0

    if not is_connected or num_nodes < 2:
        alg_conn = 0.0
    else:
        reg_deg = _regular_degree_if_applicable(graph)
        if reg_deg is not None and len(raw_evals) >= 2:
            alg_conn = max(0.0, float(reg_deg - raw_evals[-2]))
        else:
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
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

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

    fig = Figure(figsize=(10, 5))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    segments = [(pos[u], pos[v]) for u, v in graph.edges() if u in pos and v in pos]
    if segments:
        edge_collection = LineCollection(
            segments, colors="#222222", linewidths=0.4, alpha=0.6, zorder=1
        )
        ax.add_collection(edge_collection)

    if pos:
        xs = [coord[0] for coord in pos.values()]
        ys = [coord[1] for coord in pos.values()]
        num_nodes = len(pos)
        node_size = 300 if num_nodes <= 64 else max(12, 300 * 64 // num_nodes)
        ax.scatter(xs, ys, s=node_size, zorder=2)

        if num_nodes <= 64:
            for node, (x, y) in pos.items():
                ax.text(x, y, str(node), ha="center", va="center", fontsize=8, zorder=3)

        ax.autoscale_view()

    ax.set_axis_off()
    fig.savefig(out, bbox_inches="tight")
