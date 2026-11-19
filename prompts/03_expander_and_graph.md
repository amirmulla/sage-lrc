# Agent 03 — Bipartite Coset Graph & Expander Specialist (Wave 1)

## Role & Mission
You are the **Bipartite Coset Graph & Expander Specialist** in Wave 1 of the `tamo_barg_lrc` refactoring swarm. First read [`prompts/shared_context_and_contracts.md`](./shared_context_and_contracts.md) completely, then inspect `src/Expander.py`, `src/TamoBargTwoSets.py` (`_create_bipartite_graph`), `tests/TamoBarg_Expander.py`, and `tests/TamoBarg_Expansion_Measure.py` for reference.

## Exclusive File Ownership
You may **ONLY** create or modify:
* `src/tamo_barg_lrc/expander.py`

Do **NOT** delete legacy `src/Expander.py` (Agent 07 handles cleanup in Wave 3) and do **NOT** touch any other files.

---

## Bugs & Performance Targets Assigned to You
* **B12**: Fix `get_error_position_subgraph` in `src/Expander.py:13` (`[int(str(x)) for x in l_evalpts_idx]`), which crashes with `ValueError` when finite field elements use polynomial representation (e.g., `'a^2 + 1'`) and produces out-of-bounds indices when `0` is excluded from `evaluation_points` (e.g., multiplicative cosets of size $n = q - 1$).
* **B13**: Eliminate bitwise `&` boolean expressions (`(e in G.nodes[j]["coset"]) & (i != j)`).
* **B18**: Ensure headless-safe `matplotlib` rendering with explicit figure closing (`plt.close(fig)`) so batch simulations and tests never leak memory or hang on `plt.show()`.
* **P4**: Replace the $O(n^2)$ quadruple nested loop and `evalpts_done` list scans in `_create_bipartite_graph` with an $O(n)$ inverted-index construction; fast-path `edge_connectivity_k == 1` in `get_error_position_subgraph` using `nx.connected_components(graph)` instead of `EdgeComponentAuxGraph.construct(graph)` ($100\times$ faster!).

---

## Detailed Specifications: `src/tamo_barg_lrc/expander.py`

Use explicit imports (`import networkx as nx`, `from networkx.algorithms import cuts`, `import matplotlib`, `import matplotlib.pyplot as plt`, `from tamo_barg_lrc.subgroups import field_element_to_int`)—**never** `from sage.all import *` or `from networkx.algorithms import *`.

Implement the exact public API required by `shared_context_and_contracts.md`:

### 1. `create_bipartite_coset_graph(partitions, evaluation_points_idx, evaluation_points) -> tuple[nx.MultiGraph, set[int], set[int]]`
* Validate `len(partitions) == 2` and `len(evaluation_points_idx) == 2`.
* Create `graph = nx.MultiGraph()`.
* Add nodes sequentially (`node_id = 0, 1, ...`) across Partition `0` then Partition `1`:
  * For coset `j` in Partition `i`:
    ```python
    graph.add_node(
        node_id,
        partition=i,
        coset=list(partitions[i][j]),
        coset_indices=list(evaluation_points_idx[i][j]),
        bipartite=i,
    )
    ```
  * Record `left_nodes: set[int]` (for `i == 0`) and `right_nodes: set[int]` (for `i == 1`).
* **Build edges in $O(n)$**:
  * Maintain `left_owner: list[int | None] = [None] * len(evaluation_points)` and `right_owner: list[int | None] = [None] * len(evaluation_points)`.
  * Populate `left_owner[pt_idx] = node_id` during Partition 0 node creation and `right_owner[pt_idx] = node_id` during Partition 1 node creation.
  * Iterate `for pt_idx, eval_pt in enumerate(evaluation_points):`:
    * If `left_owner[pt_idx] is not None and right_owner[pt_idx] is not None`:
      `graph.add_edge(left_owner[pt_idx], right_owner[pt_idx], evalpts=eval_pt, evalpt_idx=pt_idx)`
* Return `(graph, left_nodes, right_nodes)`.

### 2. `get_error_position_subgraph(graph: nx.Graph, h_idx: int = 0, edge_connectivity_k: int = 1) -> list[int]`
* Validate `edge_connectivity_k >= 1` (raise `ValueError` otherwise).
* Compute components:
  * If `edge_connectivity_k == 1`: `raw_components = nx.connected_components(graph)`
  * Else: `raw_components = nx.k_edge_components(graph, k=edge_connectivity_k)`
* Deterministically sort components: `subgraphs = sorted( sorted(comp) for comp in raw_components )`.
* Validate `0 <= h_idx < len(subgraphs)` (raise `IndexError(f"Component index {h_idx} out of range for {len(subgraphs)} components")` otherwise).
* Extract evaluation-point indices for `subgraphs[h_idx]`:
  * Collect indices from nodes with `graph.nodes[n].get("bipartite") == 0` (and if none exist in a singleton right-node component, collect from all nodes in `subgraphs[h_idx]`):
    * Prefer `graph.nodes[n]["coset_indices"]` (exact integer positions in `0 .. length - 1`).
    * Fallback if `"coset_indices"` is absent on a custom graph: `[field_element_to_int(x) for x in graph.nodes[n]["coset"]]`.
* Return `sorted(set(indices))`.

### 3. `compute_expansion_metrics(graph: nx.Graph, right_nodes: set[int] | None = None) -> dict[str, Any]`
* Compute and return a clean dictionary containing:
  * `"is_connected"`: `bool(nx.is_connected(graph))` (handle empty graph safely).
  * `"num_connected_components"`: `int(nx.number_connected_components(graph))`.
  * `"edge_expansion"`: `float(cuts.edge_expansion(graph, right_nodes))` (infer `right_nodes` from `bipartite == 1` if `None` and non-empty; `0.0` if empty).
  * `"eigenvalues"`: Sorted list of real eigenvalues (`[round(float(val.real), 6) for val in nx.adjacency_spectrum(graph)]`).
  * `"spectral_gap"`: `float(eigenvalues[-1] - eigenvalues[-2])` if `len(eigenvalues) >= 2` else `0.0`.
  * `"algebraic_connectivity"`: Collapse parallel edges into weighted simple `nx.Graph` if `graph.is_multigraph()` (summing edge multiplicity into `weight`) before calling `float(nx.algebraic_connectivity(simple_g, method="tracemin_lu"))` when `graph.number_of_nodes() >= 2` (and `0.0` immediately if disconnected!).

### 4. `save_bipartite_graph_plot(graph: nx.Graph, left_nodes: set[int], right_nodes: set[int], output_path: str | Path) -> None`
* Render the bipartite layout cleanly without touching interactive GUI backends:
  ```python
  out = Path(output_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  left_sorted = sorted(left_nodes)
  right_sorted = sorted(right_nodes)
  m, n = len(left_sorted), len(right_sorted)
  pos = {}
  for idx, node in enumerate(left_sorted):
      pos[node] = (idx - m / 2.0, 1.0)
  for idx, node in enumerate(right_sorted):
      pos[node] = (idx - n / 2.0, 0.0)

  fig, ax = plt.subplots(figsize=(10, 5))
  try:
      nx.draw(graph, pos=pos, ax=ax, with_labels=True, node_size=300, width=0.4)
      fig.savefig(out, bbox_inches="tight")
  finally:
      plt.close(fig)
  ```

---

## Acceptance Checklist
- [ ] `ruff check src/tamo_barg_lrc/expander.py` and `ruff format --check src/tamo_barg_lrc/expander.py` pass cleanly.
- [ ] `create_bipartite_coset_graph` runs in $O(n)$ time with zero nested `.index()` or list membership scans.
- [ ] `get_error_position_subgraph` works on non-`int` field representations and when `0` is absent from `evaluation_points`.
- [ ] `save_bipartite_graph_plot` closes figures in a `finally:` block and never invokes `plt.show()`.
