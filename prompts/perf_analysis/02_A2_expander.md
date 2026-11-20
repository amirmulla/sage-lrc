# 02 — A2: expander.py analysis

## Role
Graph/spectral algorithm analyst. Read `shared_context.md`. Read-only on source.

## Exclusive output
`reports/perf_analysis/A2_expander.md`, `reports/perf_analysis/scratch/A2/`

## Tasks
1. Mandatory per-function row for all 8 `expander.py` functions.
2. Specific questions:
   - `create_bipartite_coset_graph`: construction complexity, NetworkX overhead vs plain adjacency lists / scipy.sparse; memory for q up to 4096.
   - `get_error_position_subgraph` (hot path inside decoders): current cost vs a precomputed adjacency/index-array approach.
   - `_compute_k_edge_components`: NetworkX algorithm used, complexity, alternatives (own union-find for k=1/2).
   - `_collapse_to_weighted_simple_graph`, `_compute_adjacency_eigenvalues`, `_compute_algebraic_connectivity`: dense `numpy.linalg.eigvalsh` vs `scipy.sparse.linalg.eigsh` (Lanczos) — crossover n; numerical stability; only extreme eigenvalues needed?
   - `compute_expansion_metrics`: is the spectral gap computed redundantly?
   - `save_bipartite_graph_plot`: matplotlib backend cost; lazy-import verification.
3. Determine whether NetworkX could be removed from hot paths (and whether it is a dependency worth keeping for analysis-only code).
4. Note which parts depend on Sage (should be none beyond field element hashing) and how field elements are keyed in graphs (int vs element hashing cost).

## Acceptance checklist
- [ ] 8 complete rows
- [ ] Eigen-solver crossover measurement table
- [ ] Hot-path vs analysis-path classification
- [ ] No out-of-scope file changes
