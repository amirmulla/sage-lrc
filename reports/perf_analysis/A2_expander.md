# A2 Performance & Architecture Analysis: `tamo_barg_lrc.expander`

## 1. Benchmark Environment & Methodology

| Parameter | Value |
|---|---|
| Repository | `/home/antigravity/amirmulla/optimal_lrc`, target [`src/tamo_barg_lrc/expander.py`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py) (349 LOC, 8 functions) |
| Hardware / OS | Intel Xeon Platinum 8481C, 2 vCPU, Linux `6.12.105+` x86_64; single-core pinned (`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`) |
| Primary env (`.venv-lrc`) | Python `3.11.2`, `passagemath` `10.8.x`, `networkx 3.6.1`, `numpy 2.4.6`, `matplotlib 3.11.2`, **`scipy` NOT installed** |
| Isolated SciPy env (`/tmp/venv-A2`) | Python `3.11.2`, `scipy 1.17.1`, `numpy 2.4.6`, `networkx 3.6.1` (for Lanczos / `eigsh` / `tracemin_*` crossover) |
| Methodology | Fixed seeds (`random`, `numpy`, Sage `set_random_seed` = `42`); ≥3 warm-up calls + 25 repeats (5 for >0.5 s calls, 1–3 for multi-second plots / $V=4096$ dense LAPACK); `time.perf_counter`, `cProfile`/`pstats`, `tracemalloc`; empirical exponent via log-log OLS `np.polyfit` |
| Datasets | (A) Connected `["add", "mult"]` coset multigraphs over $\mathbb{F}_q$, $q \in \{16, 64, 256, 1024, 4096\}$, $(c_0, c_1) \in \{(4,5),(4,7),(4,5),(4,11),(4,9)\}$ giving $(V,E) \in \{(7,15),(25,63),(115,255),(349,1023),(1479,4095)\}$; (B) Regular connected `["add","add"]` $K_{\sqrt{q},\sqrt{q}}$ ($q \in \{16,64,256\}$); (C) Connected $4$-regular bipartite multigraphs (sum of $4$ matchings) at $V \in \{16,32,64,128,256,512,1024,2048,4096\}$, $E=2V$ |
| Scratch scripts & data | [`bench_expander_core.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A2/bench_expander_core.py), [`bench_eigensolvers.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A2/bench_eigensolvers.py), [`bench_laplacian.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A2/bench_laplacian.py), [`bench_plot_and_extras.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A2/bench_plot_and_extras.py), and corresponding `*.json` outputs under `reports/perf_analysis/scratch/A2/` |

---

## 2. Executive Design Narrative & Critique

`expander.py` models the incidence between two coset partitions $P_0, P_1$ of an evaluation set $A \subseteq \mathbb{F}_q$ as a bipartite `nx.MultiGraph` $G = (L \cup R, E)$ with $|L| = |P_0|$, $|R| = |P_1|$, $V = |L| + |R|$, and $|E| \le n \le q$ edges (one per evaluation point appearing in both partitions; parallel edges occur whenever $|C_i^{(0)} \cap C_j^{(1)}| \ge 2$). The module serves three roles: (1) constructing $G$ during `TamoBargCodeTwoSets.__init__`, (2) extracting codeword coordinate indices of $k$-edge-connected components (`get_error_position_subgraph`), and (3) computing graph expansion metrics and saving PNG plots during simulation setup (`runner.py`).

**Strengths.** `create_bipartite_coset_graph` avoids naive $O(V^2)$ pairwise coset intersection by indexing `left_owner` / `right_owner` arrays in $O(n)$ time and keys nodes by contiguous `int` IDs (`0..V-1`) rather than Sage field elements, avoiding 3×–8× element hashing overhead inside NetworkX. It also cleanly guards `nx.k_edge_components` and `nx.algebraic_connectivity` against NetworkX 3.x `MultiGraph` limitations and handles missing SciPy without crashing.

**Architectural flaws & performance traps (evidenced below):**
1. **Eager top-level `matplotlib.pyplot` + global backend mutation + transitive Sage import.** [`expander.py:7-15`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L7-L15) eagerly imports `matplotlib`, `matplotlib.pyplot`, `networkx`, `numpy`, and `field_element_to_int` (from Sage-importing `subgroups.py`), and calls `matplotlib.use("Agg")` at import time. Because [`tamo_barg_two_sets.py:21`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L21) imports `create_bipartite_coset_graph` at module scope, **every import of `TamoBargCodeTwoSets` pays 1.81 s of `matplotlib.pyplot` startup** (out of 4.17 s total) and mutates the global Matplotlib backend—even if plotting is never called.
2. **Eager NetworkX construction on code init, yet zero decoder usage.** `TamoBargCodeTwoSets.__init__` eagerly builds an `nx.MultiGraph` ([`tamo_barg_two_sets.py:285`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L285)), consuming 3.0 MB at $q=4096$ (30× more than a `scipy.sparse` CSR or 7× more than plain CSR arrays), even though **no decoder ever reads `self._G` or calls `get_error_position_subgraph`**. Making `bipartite_graph()` lazy eliminates this cost completely for encoding/decoding workflows.
3. **Catastrophic `MultiGraph` plotting cliff (`11.0 s` at $q=1024$, `39.0 s` at $q=4096$).** `nx.draw(graph, ...)` checks `use_linecollection = not (G.is_directed() or G.is_multigraph())` (`networkx/drawing/nx_pylab.py:1948`). Passing a `MultiGraph` without `arrows=False` creates one Matplotlib `FancyArrowPatch` per edge; `bbox_inches="tight"` evaluates every Bezier path 3 times (`3,069` calls to `_get_path_in_displaycoord` at $q=1024$, 60.7% of total plot time). Passing `arrows=False` is **5.7× faster**; a direct `LineCollection` is **41.2× faster** (`101.7 ms` vs `4,183 ms`).
4. **Severely suboptimal & redundant spectral computation in `compute_expansion_metrics`.**
   - In `.venv-lrc` (no SciPy), `_compute_algebraic_connectivity` executes `_collapse_to_weighted_simple_graph` (Copy #1), then enters `nx.algebraic_connectivity`, which builds Copy #2 (`_preprocess_graph`) and runs BFS #3 (`nx.is_connected`) *before* `nx.laplacian_matrix` raises `ModuleNotFoundError`, wasting 7.7% of metrics time at $q=1024$.
   - Even *with* SciPy installed, both chosen NetworkX calls are traps: `nx.adjacency_spectrum` calls **nonsymmetric dense `scipy.linalg.eigvals`** (`spectrum.py:123`, **4.9× slower** than `np.linalg.eigvalsh`: 9.64 s vs 2.21 s at $V=2048$), while `nx.algebraic_connectivity(method="tracemin_lu")` suffers **SuperLU fill-in on expanders** (slower than dense `eigvalsh` up to $V=1024$, and **138× slower** than direct `eigsh(L, k=2, which="SA")` at $V=4096$: 3.49 s vs 25.2 ms).
   - For $d$-regular bipartite coset graphs (`["add","add"]`, `["mult","mult"]`), $\lambda_2(L) \equiv d - \lambda_2(A) = \text{spectral\_gap}$ (verified to $10^{-14}$), making the second eigensolve completely redundant.
   - All 4 simulation runners in [`runner.py:285-287`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/simulation/runner.py#L285-L287) **discard `eigenvalues`, `spectral_gap`, and `algebraic_connectivity`**, recording only `is_connected`, `num_connected_components`, and `edge_expansion`—yet pay 2.22 s at $q=4096$ for two full $O(V^3)$ eigendecompositions!
   - Worse, `cuts.edge_expansion(G, R)` on a bipartite graph cuts *all* edges, reducing identically to $|E| / \min(|L|, |R|)$ (independent of connectivity/expansion!).
5. **Multigraph edge-subdivision blowup in `_compute_k_edge_components`.** For $k=2$, subdividing parallel edges + `bridge_components` (`G.copy()` + `chain_decomposition`) is **17×–84× slower** than iterative Tarjan directly on a CSR multigraph. For $k \ge 3$, `EdgeComponentAuxGraph` runs $|V|+|E_{\text{dup}}|-1$ pure-Python `preflow_push` calls on the subdivided graph (**5.2×–17.8× slower** than a Gomory–Hu tree on the collapsed weighted graph).

---

## 3. Hot-Path vs. Analysis-Path Classification & Cold-Start Import Cost

### 3.1 Actual Repository Call Sites & Path Classification

| Function | Actual Call Sites in Repo | True Runtime Path | Cold/Per-Call Impact |
|---|---|---|---|
| [`create_bipartite_coset_graph`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L25-L114) | [`tamo_barg_two_sets.py:285`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L285) (`TamoBargCodeTwoSets.__init__`), `tests/test_expander.py` | **Code-init path** (eagerly executed once per code instance, 0.05–0.37% of `__init__`; triggers eager import of `expander.py`) | Moderate (memory + 1.81 s import tax) |
| [`get_error_position_subgraph`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L153-L198) | Public export (`__init__.py:108`), `tests/test_expander.py` (**0 calls in decoders or `runner.py`**) | **Currently analysis/public-API only** (intended for stopping-set / trapping-set error selection; *would* be hot if invoked per-word in graph-guided decoding) | Low today; high if moved into decoder loop |
| [`_compute_k_edge_components`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L117-L150) | `expander.py:177` (called only by `get_error_position_subgraph`) | **Analysis / helper** | Critical bottleneck for $k \ge 2$ |
| [`_collapse_to_weighted_simple_graph`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L201-L216) | `expander.py:241` (called only by `_compute_algebraic_connectivity`) | **Analysis-only** (once per simulation run) | Low (9% of `compute_expansion_metrics` at $q=1024$) |
| [`_compute_adjacency_eigenvalues`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L219-L233) | `expander.py:294` (called only by `compute_expansion_metrics`) | **Analysis-only** (once per simulation run; output unused by `runner.py`) | High at $q \ge 1024$ (1.09 s at $q=4096$) |
| [`_compute_algebraic_connectivity`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L236-L249) | `expander.py:296` (called only by `compute_expansion_metrics`) | **Analysis-only** (once per simulation run; output unused by `runner.py`) | High at $q \ge 1024$ (1.13 s at $q=4096$) |
| [`compute_expansion_metrics`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L252-L305) | `runner.py:510, 688, 860, 1009` (all 4 simulation runners), `tests/test_expander.py` | **Analysis-only** (once per simulation run) | High at $q \ge 1024$ (2.22 s at $q=4096$) |
| [`save_bipartite_graph_plot`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L308-L348) | `runner.py:527, 706, 876, 1018` (all 4 simulation runners), `tests/test_expander.py` | **Analysis/visualization-only** (once per simulation run) | **Severe** (11.0 s at $q=1024$, 39.0 s at $q=4096$ + 1.81 s eager import tax) |

> **Important correction to initial assumption:** A full repo grep confirms **neither `self._G` nor `get_error_position_subgraph` is called inside any decoder** in `tamo_barg_two_sets.py` or `tamo_barg.py`. The **only** `expander.py` code on the core code-construction path is `create_bipartite_coset_graph` (eagerly called in `TamoBargCodeTwoSets.__init__`) and `expander.py`'s top-level module imports.

### 3.2 Cold-Start Import Cost (5 fresh subprocesses in `.venv-lrc`)

| Import Statement | Median (ms) | p10 (ms) | p90 (ms) | Notes |
|---|---:|---:|---:|---|
| `import numpy` | 220.0 | 214.6 | 233.5 | Required by core math/analysis |
| `import networkx` | 520.4 | 468.7 | 602.8 | Pulled eagerly by `tamo_barg_two_sets.py:21` |
| `import matplotlib` | 757.7 | 622.1 | 1,187.5 | Pulled eagerly by `expander.py:7` |
| `import matplotlib; use('Agg'); import matplotlib.pyplot` | **1,814.2** | 1,518.1 | 2,265.2 | **1.81 s pure waste** for non-plotting callers |
| `import tamo_barg_lrc.expander` | **4,972.4** | 3,576.3 | 6,226.3 | Includes Sage init via `__init__.py` + `subgroups.py` + `pyplot` + `networkx` |
| `import tamo_barg_lrc.tamo_barg_two_sets` | **4,172.6** | 4,009.7 | 4,354.2 | ~43% is eager `matplotlib.pyplot` + `networkx` from `expander.py` |

---

## 4. Mandatory Per-Function Analysis (All 8 Functions)

### 4.1 Unified Scaling Sweep Reference (`connected ["add","mult"]`, `.venv-lrc`, single core)

| Function / Variant | $q=16$ ($V=7,E=15$) | $q=64$ ($V=25,E=63$) | $q=256$ ($V=115,E=255$) | $q=1024$ ($V=349,E=1023$) | $q=4096$ ($V=1479,E=4095$) | Slope vs $V$ (all / $q{\ge}256$) | Slope vs $E$ (all / $q{\ge}256$) | Peak Mem @ $q=4096$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `create_bipartite_coset_graph` | 0.028 ms | 0.100 ms | 0.456 ms | 7.008 ms | 29.268 ms | 1.35 / 1.60 | 1.30 / 1.50 | 3,005 KB |
| `_compute_k_edge_components(k=1)` | 0.002 ms | 0.005 ms | 0.023 ms | 0.077 ms | 0.416 ms | 0.97 / 1.14 | 0.92 / 1.04 | 193 KB |
| `_compute_k_edge_components(k=2)` | 0.224 ms | 0.623 ms | 9.391 ms | 25.011 ms | 134.565 ms | 1.24 / 1.05 | 1.18 / 0.96 | 4,216 KB |
| `_compute_k_edge_components(k=3)` | 13.355 ms | 86.175 ms | 1,226.8 ms | — | — | 1.62 ($q{\le}256$) | 1.60 ($q{\le}256$) | 1,087 KB @ $q=64$ |
| `get_error_position_subgraph(k=1)` | 0.008 ms | 0.021 ms | 0.075 ms | 0.278 ms | 5.618 ms | 1.17 / 1.71 | 1.11 / 1.56 | 217 KB |
| `get_error_position_subgraph(k=2)` | 0.225 ms | 0.646 ms | 5.982 ms | 25.621 ms | 124.612 ms | 1.22 / 1.18 | 1.16 / 1.09 | 4,216 KB |
| `_collapse_to_weighted_simple_graph` | 0.021 ms | 0.071 ms | 0.289 ms | 5.273 ms | 17.274 ms | 1.32 / 1.56 | 1.27 / 1.47 | 1,460 KB |
| `_compute_adjacency_eigenvalues` | 0.117 ms | 0.208 ms | 4.996 ms | 24.870 ms | 1,094.5 ms | 1.73 / 2.14 | 1.65 / 1.94 | 18,175 KB |
| `_compute_algebraic_connectivity` | 0.195 ms | 0.402 ms | 7.588 ms | 29.449 ms | 1,125.4 ms | 1.62 / 1.99 | 1.54 / 1.80 | 36,994 KB |
| `compute_expansion_metrics` | 0.424 ms | 0.768 ms | 8.890 ms | 60.937 ms | 2,215.6 ms | 1.61 / 2.18 | 1.53 / 1.99 | 37,077 KB |
| `save_bipartite_graph_plot` | 258.1 ms | 770.5 ms | 2,758.7 ms | 11,011 ms | 39,003 ms | 0.95 / 1.03 | 0.91 / 0.95 | 14,788 KB @ $q=1024$ |

*(Notice: for dense eigensolvers, $V \in [115, 1479]$ transitions from $O(V^2)$ memory/conversion into $O(V^3)$ LAPACK `dsyevd`, hence tail slope $\approx 2.14$ rising to $2.8$–$3.0$ between $V=1024$ and $V=4096$ in Section 5.3.)*

---

### 4.2 Per-Function Specification & Analysis Tables

#### 1. `expander.create_bipartite_coset_graph`

| Field | Analysis |
|---|---|
| **Function** | [`tamo_barg_lrc.expander.create_bipartite_coset_graph`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L25-L114)`(partitions, evaluation_points_idx, evaluation_points) -> tuple[nx.MultiGraph, set[int], set[int]]` — Builds the bipartite coset intersection multigraph in one linear pass. |
| **Algorithm** | Direct owner-table indexing: allocates `left_owner`/`right_owner` lists of size $n$, scans both coset partitions validating index bounds and uniqueness while calling `graph.add_node(node_id, ...)` for each coset, then loops over $i \in \{0,\dots,n-1\}$ calling `graph.add_edge(left_owner[i], right_owner[i], evalpts=..., evalpt_idx=i)`. |
| **Complexity** | **Time:** $O(n + V)$ where $n = \text{len(evaluation\_points)} \le q$ and $V = |P_0| + |P_1| \le 2n$ (each evaluation point visited $\le 3$ times; `add_edge` is $O(1)$ amortized dict insertion). **Space:** $O(n + V)$ pointers, but with high constant factor (~730 B/edge in nested NetworkX 3.x dicts: `_adj[u][v][key] = {'evalpts': ..., 'evalpt_idx': ...}` stored twice for undirected graphs). |
| **Sage primitives** | **None directly.** Stores opaque Sage field elements in node attribute `coset` and edge attribute `evalpts` without hashing or calling methods on them. (NetworkX backend: pure-Python dict-of-dicts-of-dicts in `networkx/classes/multigraph.py:442`.) |
| **Measured** | $q=64$: `0.100 ms` (p10 `0.097`, p90 `0.128`, peak `42.3 KB`); $q=256$: `0.456 ms` (p10 `0.435`, p90 `4.53`, peak `181.9 KB`); $q=1024$: `7.008 ms` (p10 `5.31`, p90 `8.96`, peak `739.6 KB`); $q=4096$: `29.268 ms` (p10 `24.23`, p90 `351.8`, peak `3,005.2 KB`). Empirical slope vs $E$: `1.30` (all) / `1.50` ($q{\ge}256$, GC/dict-resize effect). Share of `TamoBargCodeTwoSets.__init__`: `0.05%–0.37%`. |
| **Bottleneck** | `cProfile` ($q=1024$, `16.88 ms/call` under profiler): `multigraph.py:442(add_edge)` = **50.9%** (`8.60 ms`), `graph.py:535(add_node)` = **13.3%** (`2.25 ms`), `misc.py:645(_clear_cache)` = **12.5%** (`2.10 ms`, 27,440 calls!), `multigraph.py:413(new_edge_key)` = **7.7%** (`1.30 ms`), listcomp `int(pt_idx)` = **7.2%**. |
| **Correctness risks** | (1) Silently drops evaluation points present in only one partition (intended for `0` in `["add","mult"]`, but hides silent partition bugs if `evaluation_points` has orphans). (2) Copies `coset` lists and keeps references to Sage field elements forever on `self._G`. (3) Does not verify `evaluation_points[pt_idx] == coset[j]` consistency. |
| **Better algorithm** | (1) Make `bipartite_graph()` lazy on `TamoBargCodeTwoSets` (0 ms when unused). (2) Store a compact integer CSR / edge-array (`left_owner`, `right_owner`, `indptr`, `indices`) or `scipy.sparse.csr_array` biadjacency matrix $B$: **17× faster / 7× less RAM** with plain adjlist (`1.69 ms / 436 KB` at $q=4096$) and **235× faster / 30× less RAM** with `scipy.sparse.csr_array` (`0.153 ms / 201 KB` vs `36.1 ms / 6,128 KB` at $V=4096, E=8192$). (3) If retaining NetworkX, batch `add_nodes_from` / `add_edges_from` to avoid 27k `_clear_cache` calls. |
| **Sage-replacement difficulty** | **1 / 5 (Trivial)** — zero Sage calls; already operates on plain `int` indices. |
| **Priority** | **P1** (cheap in absolute ms vs `TamoBargCodeTwoSets.__init__`, but triggers the **P0** eager `matplotlib` import and allocates 3 MB unnecessarily at $q=4096$). |

---

#### 2. `expander._compute_k_edge_components`

| Field | Analysis |
|---|---|
| **Function** | [`tamo_barg_lrc.expander._compute_k_edge_components`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L117-L150)`(graph: nx.Graph, k: int) -> Iterable[Iterable[Any]]` — Computes $k$-edge-connected components for both `Graph` and `MultiGraph`. |
| **Algorithm** | Dispatches by $k$: $k=1 \to$ `nx.connected_components` (BFS, `connected.py:18`); $k \ge 2$ on simple graph $\to$ `nx.k_edge_components` (`edge_kcomponents.py:28`); $k \ge 2$ on `MultiGraph` $\to$ subdivides every parallel edge beyond the first with a synthetic degree-2 node `("__subdiv__", idx)` to produce a simple graph $G'$ with $V' = V + E_{\text{dup}}$, $E' = E + E_{\text{dup}}$, runs `nx.k_edge_components(G', k)` (`bridge_components` via `G'.copy()` + `nx.bridges`/`chain_decomposition` for $k=2$; `EdgeComponentAuxGraph.construct(G')` via recursive `nx.minimum_cut` / `preflow_push` for $k \ge 3$), then filters dummy nodes out via `comp.intersection(orig_nodes)`. |
| **Complexity** | **$k=1$:** $O(V + E)$ time & space. **$k=2$:** $O(V' + E') = O(V + E)$ theoretically, but with massive constant overhead (builds $G'$, copies $G' \to H$, builds chain-decomposition DFS + set of chain edges, deletes bridges, runs BFS). **$k \ge 3$:** $O(V' \cdot F(V', E'))$ where default NetworkX `preflow_push` has $F = O(V'^2 \sqrt{E'})$, yielding **$O(V'^3 \sqrt{E'}) \approx O(V'^4)$** worst-case time (`edge_kcomponents.py:282`), inflated further whenever $E_{\text{dup}} = |E| - |E_{\text{simple}}|$ is large (e.g. $c_0 c_1 > q$). Space: $O(V' + E')$. |
| **Sage primitives** | **None.** Pure-Python NetworkX (`connected.py:267` `_plain_bfs`, `bridges.py:70`, `chains.py:9`, `edge_kcomponents.py:316`). |
| **Measured** | **$k=1$:** $q=16$: `0.002 ms`; $q=256$: `0.023 ms`; $q=1024$: `0.077 ms` (p10 `0.076`, p90 `0.094`, peak `48.9 KB`); $q=4096$: `0.416 ms` (p10 `0.376`, p90 `6.44`, peak `192.9 KB`), slope vs $E$: `0.92`. **$k=2$:** $q=16$: `0.224 ms`; $q=256$: `9.391 ms`; $q=1024$: `25.011 ms` (p10 `20.19`, p90 `32.74`, peak `1,082 KB`); $q=4096$: `134.565 ms` (p10 `111.8`, p90 `373.8`, peak `4,216 KB`), slope vs $E$: `1.18`. **$k=3$:** $q=16$: `13.36 ms`; $q=64$: `86.18 ms` (peak `1,087 KB`); $q=256$: **`1,226.8 ms`** (p10 `923`, p90 `1,341`). |
| **Bottleneck** | `cProfile` ($q=1024$, $k=2$, `84.96 ms/call` under profiler): `bridge_components` = **80.9%** (`68.7 ms`), inside which `bridges()`/`chain_decomposition()` = **58.8%** (`50.0 ms`, 13,520 generator yields) and `H.copy()` + `remove_edges_from()` = **22%**, plus subdivision graph creation (`16.2 ms`, **19.1%**). For $k \ge 3$, `EdgeComponentAuxGraph` recursive `preflow_push` dominates (>99%). |
| **Correctness risks** | (1) **Self-loops skipped** (`if u == v: continue`), harmless for bipartite graphs. (2) For $k \ge 3$, note that `nx.k_edge_components` computes $k$-edge-connected *components* (vertices pairwise $k$-edge-connected in $G$), which can differ from $k$-edge-connected *subgraphs* (`nx.k_edge_subgraphs`) for $k \ge 3$—verify if stopping-set analysis actually wants $k$-edge-connected subgraphs or $k$-cores! (3) No check for `k < 1` in helper itself (guarded by caller). |
| **Better algorithm** | **$k=1$:** Plain integer BFS or DSU on CSR arrays (zero NetworkX overhead; DSU is `0.18 ms` at $q=1024$ in pure Python, or `scipy.sparse.csgraph.connected_components` in Cython is ~`0.02 ms`). **$k=2$:** Iterative **Tarjan 2-edge-connected components directly on CSR multigraph** (skipping only *one* edge copy back to parent; zero subdivision, zero graph copy!) — measured **16.8×–83.8× faster** (`0.112 ms` vs `9.39 ms` at $q=256$; `0.448 ms` vs `25.0 ms` at $q=1024$; `8.03 ms` vs `134.6 ms` at $q=4096$; **8.4×–18.9× less memory**). **$k \ge 3$:** Collapse multigraph to weighted simple graph and build a **Gomory–Hu tree** (`nx.gomory_hu_tree(sub, capacity="weight")`), cutting $V' \to V$ — measured **5.2×–17.8× faster** (`0.75 ms` vs `13.36 ms` at $q=16$; `167.7 ms` vs `1,226.8 ms` at $q=256$; **9.8× less memory** at $q=64$). |
| **Sage-replacement difficulty** | **1 / 5 (Trivial)** — zero Sage dependency; Tarjan ($k=2$) is ~45 LOC pure Python/NumPy. |
| **Priority** | **P2** (currently unused in simulations/decoders; jumps to **P1** if $k \ge 2$ trapping-set extraction is enabled in simulations). |

---

#### 3. `expander.get_error_position_subgraph`

| Field | Analysis |
|---|---|
| **Function** | [`tamo_barg_lrc.expander.get_error_position_subgraph`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L153-L198)`(graph: nx.Graph, h_idx: int = 0, edge_connectivity_k: int = 1) -> list[int]` — Returns sorted unique codeword indices belonging to the `h_idx`-th $k$-edge-connected component. |
| **Algorithm** | Calls `_compute_k_edge_components(graph, k)`, sorts every component and sorts the list of components lexicographically (`sorted(sorted(comp) for comp in raw_components)`), selects `subgraphs[h_idx]`, filters nodes with `bipartite == 0` (fallback to all component nodes if none), concatenates `node_data["coset_indices"]` (or legacy `field_element_to_int` on `"coset"`), and returns `sorted(set(indices))`. |
| **Complexity** | **Time:** $T_{\text{k-edge}}(V, E, k) + O(V \log V + n_c \log n_c)$ where $n_c \le n$ is the number of evaluation points in component `h_idx`. **Space:** $O(V + E + n_c)$. |
| **Sage primitives** | **Legacy fallback only** ([`expander.py:194`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L194)): `field_element_to_int(elem)` calls `elem.is_zero()` and `elem.to_integer()` (`sage/rings/finite_rings/element_base.pyx:1049`, Cython polynomial evaluation over $\mathbb{Z}$). Never invoked on graphs built by `create_bipartite_coset_graph` (which always populates `coset_indices`), yet its top-level import at line 13 forces Sage loading on `import tamo_barg_lrc.expander`. |
| **Measured** | **$k=1$ (fast path):** $q=16$: `0.008 ms`; $q=64$: `0.021 ms`; $q=256$: `0.075 ms` (p10 `0.074`, p90 `0.083`, peak `14.2 KB`); $q=1024$: `0.278 ms` (p10 `0.264`, p90 `3.48`, peak `53.8 KB`); $q=4096$: `5.618 ms` (p10 `1.41`, p90 `7.64`, peak `216.8 KB`), slope vs $E$: `1.11`. **$k=1$ (legacy `"coset"` path):** `0.154 ms` ($q=256$), `0.610 ms` ($q=1024$), `10.64 ms` ($q=4096$) — **1.9×–2.2× slower** due to `field_element_to_int`. **$k=2$:** `0.646 ms` ($q=64$), `5.982 ms` ($q=256$), `25.621 ms` ($q=1024$), `124.612 ms` ($q=4096$, peak `4,216 KB`). |
| **Bottleneck** | For $k=1$ (`cProfile` at $q=1024$, `4.04 ms/call`): `indices.extend(int(idx) for idx in ...)` generator + `int()` coercion (**33.3%**, `1.35 ms`), double `sorted()` (**30.3%**, `1.23 ms`), `_plain_bfs` (**26.4%**, `1.07 ms`). For $k=2$: `_compute_k_edge_components` is **96.3%** (`85.0 / 88.2 ms`). |
| **Correctness risks** | (1) **Subtle edge case on disconnected / $k \ge 2$ components**: selecting only `bipartite == 0` nodes (`line 184`) assumes every evaluation point in the component is covered by a left coset in that component; for $k \ge 2$ on a bipartite graph, a 2-edge-connected component can contain left cosets whose *other* evaluation points lie on bridges going to *other* components, so returning the entire `coset_indices` of left nodes includes codeword positions whose incident edges are **outside** the $k$-edge component! (2) Unnecessary `int(idx)` re-wrapping on elements already cast to `int` in `create_bipartite_coset_graph`. (3) Lexicographic sorting of components fails with `TypeError` if custom graphs mix non-comparable node key types. |
| **Better algorithm** | Precompute CSR arrays + cache component partitions per `(graph, k)` (immutable once code is built!). Without caching, precomputed CSR + DSU ($k=1$) / Tarjan ($k=2$) already drops $q=4096$ from `5.62 ms → 0.91 ms` (**6.2× faster**) for $k=1$ and `124.6 ms → 7.52 ms` (**16.6× faster**) for $k=2$ (`25.6 ms → 0.47 ms`, **55× faster** at $q=1024$). With caching on `TamoBargCodeTwoSets`, repeated lookups become $O(1)$ (`<0.001 ms`). Lazy-import `field_element_to_int` inside the `elif "coset"` branch to sever `expander.py`'s top-level Sage dependency. |
| **Sage-replacement difficulty** | **1 / 5 (Trivial)** — remove top-level `field_element_to_int` import; use `int(elem)` or lazy fallback only when `"coset_indices"` is absent. |
| **Priority** | **P2** (0 decoder/simulation call sites today; **P1** to lazy-import `field_element_to_int` and sever Sage dependency). |

---

#### 4. `expander._collapse_to_weighted_simple_graph`

| Field | Analysis |
|---|---|
| **Function** | [`tamo_barg_lrc.expander._collapse_to_weighted_simple_graph`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L201-L216)`(graph: nx.Graph) -> nx.Graph` — Collapses parallel edges of a `MultiGraph` into a simple weighted `nx.Graph`, dropping self-loops. |
| **Algorithm** | Returns `graph` unchanged if not multigraph; otherwise creates empty `nx.Graph()`, copies all nodes, iterates `graph.edges(data=True)`, skips `u == v`, and accumulates `float(data.get("weight", 1.0))` into `simple_g[u][v]["weight"]`. |
| **Complexity** | **Time:** $O(V + E)$ (one pass over nodes + multiedges; `has_edge` is $O(1)$ dict lookup). **Space:** $O(V + E_{\text{simple}})$ in NetworkX dict-of-dicts. |
| **Sage primitives** | **None.** Pure-Python NetworkX (`graph.py`). |
| **Measured** | $q=16$: `0.021 ms` (peak `5.8 KB`); $q=64$: `0.071 ms` (`22.4 KB`); $q=256$: `0.289 ms` (p10 `0.274`, p90 `4.34`, `93.9 KB`); $q=1024$: `5.273 ms` (p10 `1.21`, p90 `7.34`, `376.4 KB`); $q=4096$: `17.274 ms` (p10 `13.09`, p90 `23.31`, `1,459.7 KB`). Empirical slope vs $E$: `1.27` (all) / `1.47` ($q{\ge}256$). |
| **Bottleneck** | Python loop over `graph.edges(data=True)` + per-edge `has_edge` / `add_edge` dict mutations and `_clear_cache` invalidations. |
| **Correctness risks** | (1) If `graph` is already a simple `Graph` containing self-loops, returns it *without* stripping self-loops (inconsistent with the `MultiGraph` branch; downstream `_compute_algebraic_connectivity` compensates with `np.fill_diagonal(..., 0.0)`). (2) Sums signed weights rather than `abs(weight)` before passing to `nx.algebraic_connectivity` (benign for coset graphs where all weights are `+1.0`, subtle if negative weights ever existed). |
| **Better algorithm** | **Eliminate entirely** by accumulating directly from edge arrays into a `scipy.sparse.csr_array` (`.sum_duplicates()` in C++, `0.153 ms` at $V=4096, E=8192$) or dense NumPy biadjacency `np.add.at(B, (u, v - n_left), 1.0)` — measured **`0.021 ms` vs `5.27 ms` (251× faster!) at $q=1024$** and **`0.216 ms` vs `17.27 ms` (80× faster!) at $q=4096$**. |
| **Sage-replacement difficulty** | **1 / 5 (Trivial)** — zero Sage dependency. |
| **Priority** | **P2** (minor fraction of spectral runtime at $q \ge 1024$, disappears automatically if spectral routines build matrix directly). |

---

#### 5. `expander._compute_adjacency_eigenvalues`

| Field | Analysis |
|---|---|
| **Function** | [`tamo_barg_lrc.expander._compute_adjacency_eigenvalues`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L219-L233)`(graph: nx.Graph) -> list[float]` — Computes all sorted real adjacency eigenvalues rounded to 6 decimal places. |
| **Algorithm** | Tries `nx.adjacency_spectrum(graph)` (which in `networkx/linalg/spectrum.py:121-123` imports SciPy and calls **nonsymmetric dense** `scipy.linalg.eigvals(nx.adjacency_matrix(G).todense())`!). When SciPy is absent (as in `.venv-lrc`), catches `ModuleNotFoundError`, builds dense $V \times V$ `adj_mat` via `nx.to_numpy_array(graph, weight="weight", multigraph_weight=sum)`, and runs symmetric LAPACK `np.linalg.eigvalsh(adj_mat)` (`dsyevd`). Finally sorts `round(float(val.real), 6)`. |
| **Complexity** | **Time:** $O(V^3)$ dense LAPACK eigensolve + $O(V^2 + E)$ dense matrix assembly (`eigvals` constant factor is ~5× worse than `eigvalsh` because it uses Hessenberg + QR `dgeev` instead of tridiagonal divide-and-conquer `dsyevd`). **Space:** $O(V^2)$ (`8 V^2` bytes for `float64` matrix + LAPACK workspace $\approx 16 V^2$ bytes; 18.2 MB at $V=1479$, 134 MB at $V=4096$). |
| **Sage primitives** | **None.** NumPy LAPACK (`dsyevd`) in `.venv-lrc`; SciPy LAPACK (`dgeev` via `nx.adjacency_spectrum`) if SciPy is installed. |
| **Measured** | (`.venv-lrc`, NumPy fallback): $q=16$: `0.117 ms` (`8.2 KB`); $q=64$: `0.208 ms` (`22.4 KB`); $q=256$: `4.996 ms` (p10 `1.00`, p90 `6.97`, `164.1 KB`); $q=1024$: `24.870 ms` (p10 `18.73`, p90 `30.96`, `1,189 KB`); $q=4096$ ($V=1479$): **`1,094.5 ms`** (p10 `1,040`, p90 `1,209`, peak **`18,175 KB`**). Tail empirical slope vs $V$: `2.14` (approaching `3.0` for $V \ge 1024$; see Section 5.3 where $V=2048 \to 4096$ gives `2.21 s → 10.40 s`, ratio `4.7×`–`8×`). |
| **Bottleneck** | At $q=1024$ (`cProfile`): `np.linalg.eigvalsh` (~60%) + `nx.to_numpy_array` (~35%). At $q=4096$ ($V=1479$), LAPACK `dsyevd` dominates (>92%). With SciPy installed, `nx.adjacency_spectrum` (`dgeev`) is **4.9× slower** (`9,641 ms` vs `2,164 ms` at $V=2048$)! |
| **Correctness risks** | (1) **Premature 6-decimal rounding** (`line 233`) corrupts downstream `spectral_gap = eigenvalues[-1] - eigenvalues[-2]` by up to $10^{-6}$ (measured `3.20e-7` error at $q=1024$); weak expanders with gap $< 5 \times 10^{-7}$ report `0.0`. (2) **Performance regression when SciPy is installed**: installing SciPy silently switches from symmetric `np.linalg.eigvalsh` to nonsymmetric `scipy.linalg.eigvals`, making `_compute_adjacency_eigenvalues` **4.9×–10× slower**! (3) Does not check `is_multigraph` edge weight consistency if mixed weighted/unweighted edges exist. |
| **Better algorithm** | (1) **Never call `nx.adjacency_spectrum`** on undirected graphs; always use symmetric/bipartite solvers. (2) **Exploit bipartiteness** ($A = \begin{pmatrix} 0 & B \\ B^T & 0 \end{pmatrix}$): for full spectrum at $V \le 512$, compute eigenvalues of the smaller Gram matrix $B B^T$ (size $\min(|L|,|R|)^2$) via `np.linalg.eigvalsh` and mirror $\pm\sqrt{\mu_i}$ (**7.1×–8.0× faster**: `282 ms` vs `2,212 ms` at $V=2048$, error $\sim 10^{-15}$). (3) **When only `spectral_gap` is needed** (all real callers!): for $V > 256$, use sparse Lanczos `scipy.sparse.linalg.svds(B_csr, k=2)` or `eigsh(A_csr, k=2, which="LA")` — **345× faster** at $V=4096$ (`30.1 ms` vs `10,400 ms`, error `6.7e-15`). |
| **Sage-replacement difficulty** | **1 / 5 (Trivial)** — pure NumPy / SciPy. |
| **Priority** | **P1** (high speed-up 8×–345× for large $q$; analysis-only so lower frequency than decoders). |

---

#### 6. `expander._compute_algebraic_connectivity`

| Field | Analysis |
|---|---|
| **Function** | [`tamo_barg_lrc.expander._compute_algebraic_connectivity`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L236-L249)`(graph: nx.Graph, is_connected: bool) -> float` — Computes the Fiedler value $\lambda_2(L)$ of the graph Laplacian. |
| **Algorithm** | Short-circuits to `0.0` if $V < 2$ or disconnected. Calls `_collapse_to_weighted_simple_graph(graph)` (Copy #1), then tries `nx.algebraic_connectivity(simple_g, method="tracemin_lu")` (`algebraicconnectivity.py:310`: builds Copy #2 via `_preprocess_graph`, runs BFS #3 via `nx.is_connected`, builds sparse Laplacian, runs TraceMIN with SuperLU `splu` factorization). In `.venv-lrc` (no SciPy), `nx.laplacian_matrix` raises `ModuleNotFoundError` *after* Copy #2 + BFS #3; the `except` block builds dense `adj_mat` via `nx.to_numpy_array`, constructs dense $L = D - A$, and runs full `np.linalg.eigvalsh(L)[1]`. |
| **Complexity** | **NumPy fallback:** $O(V^3)$ time, $O(V^2)$ space (allocates both `adj_mat` and `lap_mat` simultaneously: `37.0 MB` peak at $V=1479$, double the memory of `_compute_adjacency_eigenvalues`!). **With SciPy (`tracemin_lu`):** Sparse LU (`splu`) on a $d$-regular expander graph has no small separators, causing superlinear fill-in $O(V^{1.5}\text{–}V^2)$ time/memory per factorization (measured `3,488 ms` at $V=4096$). |
| **Sage primitives** | **None.** NetworkX + NumPy LAPACK (`dsyevd`) in `.venv-lrc`; SuperLU (`scipy.sparse.linalg.splu`) + `scipy.linalg.eigh` in TraceMIN if SciPy is installed. |
| **Measured** | (`.venv-lrc`, NumPy fallback): $q=16$: `0.195 ms` (`18.4 KB`); $q=64$: `0.402 ms` (`58.2 KB`); $q=256$: `7.588 ms` (`489.8 KB`); $q=1024$: `29.449 ms` (p10 `25.03`, p90 `37.17`, `2,633 KB`); $q=4096$ ($V=1479$): **`1,125.4 ms`** (p10 `1,075`, p90 `1,181`, peak **`36,994 KB`**). Tail empirical slope vs $V$: `1.99` (rising toward `3.0` for $V \ge 1024$). |
| **Bottleneck** | Step breakdown at $q=1024$ (`8.12 ms` unprofiled): dense `to_numpy_array` + `eigvalsh(L)` = **67.2%** (`5.45 ms`), `_collapse_to_weighted_simple_graph` = **17.4%** (`1.42 ms`), **wasted `nx.algebraic_connectivity` try-block before `ModuleNotFoundError`** = **14.9%** (`1.21 ms`!). Peak memory is doubled because `np.diag(adj_mat.sum(axis=1)) - adj_mat` creates a full second $V \times V$ matrix instead of mutating `-adj_mat` in-place and writing the diagonal! |
| **Correctness risks** | (1) **In-place memory bloat**: `np.diag(...)` allocates an extra $V \times V$ zero matrix (`17.5 MB` at $V=1479$, `134 MB` at $V=4096$). (2) **If SciPy is installed**, `method="tracemin_lu"` is slower than dense `eigvalsh` up to $V=1024$ and **138× slower** than Lanczos at $V=4096$; and switching to NetworkX's default `method="tracemin_pcg"` **hangs indefinitely (`>12 s` TIMEOUT)** for $V \ge 512$ because `_PCGSolver` has no iteration cap (`algebraicconnectivity.py:33`)! (3) **Plain Lanczos small-$V$ inaccuracy**: `eigsh(L, k=2, which="SA")` had a `2.6e-2` error at $V=32$ (Section 5.3). |
| **Better algorithm** | (1) **Regular-graph shortcut**: if all degrees equal $d$ (`["add","add"]`, `["mult","mult"]`), return `d - lambda_2(A)` = `spectral_gap` in **$O(1)$ time** (verified exact to $3.5 \times 10^{-14}$). (2) Otherwise: for $V \le 256$, build `-A` directly in-place and write diagonal degrees (`np.linalg.eigvalsh(L)[1]`, `0.02–2.83 ms`, exact to $10^{-15}$, half the RAM). For $V > 256$ (with SciPy), use **`scipy.sparse.linalg.eigsh(L_csr, k=2, which="SA", ncv=min(V-1, 40))`** (or `nx.algebraic_connectivity(..., method="lanczos")`) — **`25.2 ms` vs `9,549 ms` dense (378× faster!) and vs `3,488 ms` `tracemin_lu` (138× faster!) at $V=4096$**, error `3.1e-15`. |
| **Sage-replacement difficulty** | **1 / 5 (Trivial)** — pure NumPy / SciPy. |
| **Priority** | **P1** (large speed-up + halves peak RAM + avoids `tracemin_lu` SuperLU trap if SciPy is added). |

---

#### 7. `expander.compute_expansion_metrics`

| Field | Analysis |
|---|---|
| **Function** | [`tamo_barg_lrc.expander.compute_expansion_metrics`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L252-L305)`(graph: nx.Graph, right_nodes: set[int] \| None = None) -> dict[str, Any]` — Computes connectivity, cut expansion, adjacency spectrum, spectral gap, and algebraic connectivity. |
| **Algorithm** | Sequentially runs: (1) `nx.is_connected(graph)` (BFS #1), (2) `nx.number_connected_components(graph)` (BFS #2), (3) `cuts.edge_expansion(graph, inferred_right)` (`cuts.py:230` — sums all edges crossing $(R, V \setminus R)$ divided by $\min(|R|, |L|)$), (4) `_compute_adjacency_eigenvalues(graph)` (full dense $V \times V$ eigensolve #1), (5) `spectral_gap = eigenvalues[-1] - eigenvalues[-2]`, (6) `_compute_algebraic_connectivity(graph, is_connected)` (full dense $V \times V$ eigensolve #2 after collapsing & failing SciPy try-block). |
| **Complexity** | **Time:** $O(V^3)$ (two full dense eigendecompositions + 3 BFS passes + $O(E)$ cut scan). **Space:** $O(V^2)$ (peak `37.1 MB` at $q=4096$). |
| **Sage primitives** | **None.** Pure NetworkX + NumPy. |
| **Measured** | $q=16$: `0.424 ms` (`19.3 KB`); $q=64$: `0.768 ms` (`59.7 KB`); $q=256$: `8.890 ms` (p10 `6.79`, p90 `14.90`, `498.4 KB`); $q=1024$: `60.937 ms` (p10 `47.82`, p90 `74.05`, `2,655 KB`); $q=4096$ ($V=1479$): **`2,215.6 ms`** (p10 `2,198`, p90 `2,606`, peak **`37,077 KB`**). Empirical tail slope vs $V$: `2.18`. |
| **Bottleneck** | Breakdown at $q=1024$ (`15.59 ms` unprofiled; `105.4 ms` under `cProfile`): `_compute_algebraic_connectivity` = **52.1%** (`8.12 ms`), `_compute_adjacency_eigenvalues` = **41.6%** (`6.48 ms`), `cuts.edge_expansion` = **2.1%** (`0.32 ms`), double BFS = **1.0%** (`0.15 ms`). Together the two eigensolvers are **93.7%** at $q=1024$ and **99.6%** at $q=4096$ — and **100% unused by `runner.py`**! |
| **Correctness risks** | (1) **`edge_expansion` is mathematically degenerate on bipartite graphs**: when `right_nodes = R`, EVERY edge of $G=(L \cup R, E)$ crosses the cut $(R, L)$, so `cuts.edge_expansion(G, R) == |E| / min(|L|, |R|)` identically (verified to $<10^{-12}$ across all $q=16..4096$!). A completely disconnected bipartite graph with identical degrees reports the **exact same `edge_expansion`** as a Ramanujan expander—giving misleading CSV headers in `runner.py`! (2) `spectral_gap` loses precision from 6-decimal rounding in `_compute_adjacency_eigenvalues` (`3.2e-7` error at $q=1024$). (3) For disconnected graphs, `eigenvalues[-1] - eigenvalues[-2]` can be `0.0` (equal max degree components) or $>0$ (unequal component radii), which is not the Cheeger spectral gap of a component. |
| **Better algorithm** | (1) Replace double BFS with single pass (`num_components = ...; is_connected = (num_components == 1)`). (2) Replace `cuts.edge_expansion(G, R)` with $O(1)$ `E / min(|L|, |R|)` (if preserving exact legacy value) or document its triviality and report `spectral_gap` / `algebraic_connectivity` in `runner.py` instead! (3) Add `include_spectrum: bool = False` (or `compute_spectral: bool = True`): skipping unused eigensolvers cuts `runner.py` setup at $q=4096$ from **`2,216 ms → <1 ms` (>2,000× speedup!)**. (4) When spectral metrics *are* requested: build matrix once, use regular identity $\lambda_2(L) = d - \lambda_2(A)$ when regular, and dispatch dense ($V \le 256$) vs sparse Lanczos ($V > 256$) to cut $q=4096$ from `2,216 ms → ~35 ms` (**~63× faster**). |
| **Sage-replacement difficulty** | **1 / 5 (Trivial)** — zero Sage dependency. |
| **Priority** | **P1** (trivial to fix redundant BFS/collapse/unused-spectral overhead; fixes misleading `edge_expansion` metric). |

---

#### 8. `expander.save_bipartite_graph_plot`

| Field | Analysis |
|---|---|
| **Function** | [`tamo_barg_lrc.expander.save_bipartite_graph_plot`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L308-L348)`(graph: nx.Graph, left_nodes: set[int], right_nodes: set[int], output_path: str \| Path) -> None` — Renders a two-layer bipartite PNG headlessly without leaking Matplotlib figures. |
| **Algorithm** | Checks `matplotlib.get_backend()`, sorts left/right/extra nodes to assign `(x, y)` coordinates on $y \in \{1, 0, 0.5\}$, allocates a `10x5` figure, calls `nx.draw(graph, pos=pos, ax=ax, with_labels=True, node_size=300, width=0.4)`, saves with `fig.savefig(out, bbox_inches="tight")`, and closes `fig` in `finally`. |
| **Complexity** | **Time:** $O((V + E) \cdot C_{\text{patch}})$ where $C_{\text{patch}}$ is huge (~10 ms/edge-bundle) because `nx.draw_networkx_edges` (`nx_pylab.py:1948`) disables vectorized `LineCollection` whenever `G.is_multigraph()` is `True`, creating $E$ individual `FancyArrowPatch` artists + $V$ `Text` label artists, each rendered/measured 3× due to `bbox_inches="tight"`. **Space:** $O(V + E)$ Matplotlib artist objects (~14.8 MB at $q=1024$). |
| **Sage primitives** | **None.** NetworkX `nx_pylab.py` + Matplotlib Agg renderer. |
| **Measured** | $q=16$ ($V=7,E=15$): `258.1 ms` (p10 `239`, p90 `266`, peak `623 KB`); $q=64$ ($V=25,E=63$): `770.5 ms` (`1,410 KB`); $q=256$ ($V=115,E=255$): `2,758.7 ms` (p10 `2,655`, p90 `3,121`, `4,376 KB`); $q=1024$ ($V=349,E=1023$): **`11,011 ms`** (p10 `10,698`, p90 `11,152`, `14,788 KB`); $q=4096$ ($V=1479,E=4095$): **`39,003 ms` (39.0 s!)**. Empirical slope vs $E$: `0.91` (all) / `0.95` ($q{\ge}256$) — linear $O(E)$, with an absurd ~10 ms/edge constant factor! |
| **Bottleneck** | `cProfile` ($q=1024$, `15.04 s` under profiler): `patches.py:4619(_get_path_in_displaycoord)` = **60.7%** (`9,124 ms`, **3,069 calls** = $3 \times 1,023$ edges!), `savefig`/`print_figure` = **55.5%** (`8,345 ms`, overlapping with `draw_wrapper` = `7,931 ms`). Plus **1,814 ms** cold-start import penalty paid by every caller of `tamo_barg_two_sets`! |
| **Correctness risks** | (1) **Global side effect at import time**: [`expander.py:15`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L15) calls `matplotlib.use("Agg")` unconditionally at module import, breaking interactive notebook/GUI backends for any user importing `TamoBargCodeTwoSets`. (2) Calling `matplotlib.use("Agg")` inside the function after `pyplot` is already imported (`line 324`) is a no-op / raises warning in Matplotlib if another backend was active. (3) For $V \ge 100$, drawing 300-pt nodes and overlapping text labels on a `10x5` canvas produces an unreadable black smear while burning 11–39 seconds! |
| **Better algorithm** | (1) **Lazy-import** `matplotlib` (using `matplotlib.figure.Figure` + `FigureCanvasAgg` directly instead of `matplotlib.pyplot`, avoiding global backend mutation *and* saving **1.81 s** on every non-plotting import!). (2) **Pass `arrows=False`** (or draw edges with a single vectorized `matplotlib.collections.LineCollection` + `ax.scatter`, disabling labels when $V > 64$): measured at $q=1024$ (`4,183 ms` baseline): `arrows=False` $\to$ **`734 ms` (5.7× faster)**; `LineCollection` + labels $\to$ **`543 ms` (7.7× faster)**; `LineCollection` without unreadable labels $\to$ **`101.7 ms` (41.2× faster!!)**. |
| **Sage-replacement difficulty** | **1 / 5 (Trivial)** — zero Sage dependency; `Figure` + `FigureCanvasAgg` + `LineCollection` removes even the NetworkX drawing dependency. |
| **Priority** | **P0** (lazy import eliminates **1.81 s** startup cost on every `TamoBargCodeTwoSets` import; `arrows=False`/`LineCollection` gives **5.7×–41.2×** speedup on every simulation run). |

---

## 5. Deep-Dive Answers to Specific Questions

### 5.1 `create_bipartite_coset_graph`: NetworkX vs. Plain Adjacency Lists / CSR / `scipy.sparse`

Measured across both real connected `["add","mult"]` coset graphs (`.venv-lrc`) and $4$-regular bipartite multigraphs (`/tmp/venv-A2`):

**A. Real connected `["add","mult"]` coset graphs (`.venv-lrc`, single core):**

| $q$ | $V$ | $E$ | `nx.MultiGraph` ms / peak KB | Plain Python `adjlist` ms / peak KB | NumPy CSR arrays ms / peak KB | Speedup (`adjlist` / `CSR`) | RAM Reduction (`adjlist` / `CSR`) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 7 | 15 | 0.028 / 10.3 | 0.004 / 1.0 | 0.027 / 7.7 | 7.0× / 1.0× | 10.3× / 1.3× |
| 64 | 25 | 63 | 0.100 / 42.3 | 0.010 / 3.2 | 0.063 / 9.1 | 10.0× / 1.6× | 13.2× / 4.6× |
| 256 | 115 | 255 | 0.456 / 181.9 | 0.037 / 15.6 | 0.215 / 14.9 | 12.3× / 2.1× | 11.7× / 12.2× |
| 1024 | 349 | 1,023 | 7.008 / 739.6 | 0.155 / 67.8 | 1.884 / 82.1 | **45.2×** / 3.7× | 10.9× / 9.0× |
| 4096 | 1,479 | 4,095 | 29.268 / 3,005.2 | **1.687 / 435.9** | 10.443 / 422.7 | **17.3×** / 2.8× | **6.9× / 7.1×** |

**B. `nx.MultiGraph` vs. `scipy.sparse.csr_array` biadjacency $B$ ($4$-regular bipartite, `/tmp/venv-A2`):**

| $V$ | $E$ | `nx.MultiGraph` ms / peak KB | `scipy.sparse.csr_array` ms / peak KB | Speedup | RAM Reduction |
|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 0.037 / 19.9 | 0.068 / 2.2 | 0.5× | 9.0× |
| 32 | 64 | 0.070 / 41.0 | 0.066 / 3.0 | **1.1×** *(crossover)* | 13.7× |
| 128 | 256 | 0.287 / 165.2 | 0.069 / 7.7 | 4.1× | 21.5× |
| 512 | 1,024 | 4.288 / 723.8 | 0.077 / 26.4 | 55.3× | 27.4× |
| 1,024 | 2,048 | 6.700 / 1,496.3 | 0.084 / 51.4 | 80.1× | 29.1× |
| 2,048 | 4,096 | 19.612 / 3,039.8 | 0.112 / 101.4 | 174.6× | 30.0× |
| 4,096 | 8,192 | 36.077 / 6,127.8 | **0.153 / 201.4** | **235.1×** | **30.4×** |

**Takeaway:** At $q=4096$, `nx.MultiGraph` uses **~3.0 MB** (or **6.1 MB** at $V=4096, E=8192$) solely in Python dict overhead, vs **~200–435 KB** for CSR/adjlist (**7×–30× smaller**). Even `scipy.sparse.csr_array` beats `nx.MultiGraph` in wall-clock time for all $V \ge 32$ and is **235× faster** at $V=4096$.

---

### 5.2 `get_error_position_subgraph` & `_compute_k_edge_components`: Current vs. Precomputed CSR/DSU/Tarjan/Gomory–Hu

| $q$ ($V, E, E_{\text{dup}}$) | $k=1$ `nx` vs CSR+DSU (`get_error_pos`) | $k=1$ legacy `"coset"` fallback | $k=2$ `nx` subdiv vs CSR+Tarjan (`_compute_k_edge`) | $k=2$ `get_error_pos` `nx` vs CSR | $k=3$ `nx` subdiv vs Gomory–Hu |
|---|---:|---:|---:|---:|---:|
| 16 (7, 15, 5) | 0.008 vs 0.005 ms (**1.6×**) | 0.013 ms | 0.224 vs 0.007 ms (**32.0×**) | 0.225 vs 0.008 ms (**28.1×**) | 13.36 vs 0.75 ms (**17.8×**) |
| 64 (25, 63, 9) | 0.021 vs 0.014 ms (**1.5×**) | 0.039 ms | 0.623 vs 0.026 ms (**24.0×**) | 0.646 vs 0.028 ms (**23.1×**) | 86.18 vs 15.33 ms (**5.6×**) |
| 256 (115, 255, 6) | 0.075 vs 0.056 ms (**1.3×**) | 0.154 ms | 9.391 vs 0.112 ms (**83.8×**) | 5.982 vs 0.117 ms (**51.1×**) | 1,226.8 vs 167.7 ms (**7.3×**) |
| 1024 (349, 1023, 15) | 0.278 vs 0.204 ms (**1.4×**) | 0.610 ms | 25.011 vs 0.448 ms (**55.8×**) | 25.621 vs 0.465 ms (**55.1×**) | — |
| 4096 (1479, 4095, 12) | 5.618 vs 0.911 ms (**6.2×**) | 10.640 ms | 134.57 vs 8.032 ms (**16.8×**) | 124.61 vs 7.524 ms (**16.6×**) | — |

*(All prototype outputs were `assert`-verified bit-for-bit identical to NetworkX on every $q$ and $k \in \{1, 2, 3\}$.)*

- **Why `nx.connected_components` ($k=1$) beat pure-Python DSU slightly on raw component finding** (`0.077 ms` vs `0.181 ms` at $q=1024$): NetworkX's `_plain_bfs` iterates C-level dict keys with tight set ops, whereas our prototype DSU converted NumPy `edge_u`/`edge_v` via `.tolist()` and ran Python `while` loops; yet end-to-end `get_error_position_subgraph` on CSR was still **1.3×–6.2× faster** because it avoids per-node dict lookups (`graph.nodes[node]`) and redundant `int()` casts! A plain CSR BFS or cached component table is faster still.
- **Why Tarjan dominates $k=2$ (17×–84× faster, 8.4×–18.9× less memory):** NetworkX `bridge_components` rejects `MultiGraph`, forcing `expander.py` to allocate a subdivided graph $G'$, then `bridge_components` deep-copies $G' \to H$, runs `chain_decomposition` (generating thousands of tuples), removes bridges, and runs BFS. Iterative Tarjan handles parallel edges natively in one DFS pass over CSR arrays (`if v == p and not skipped_parent: skipped_parent = True; continue`).
- **Why Gomory–Hu dominates $k \ge 3$ (5.6×–17.8× faster, 9.8× less memory):** Collapsing parallel edges to integer capacities preserves exact $s$-$t$ min-cuts while keeping vertex count at $V$ instead of $V + E_{\text{dup}}$, and `nx.gomory_hu_tree` contracts partitions in-place rather than running $V'-1$ uncontracted max-flows on the full graph as `EdgeComponentAuxGraph` does.

---

### 5.3 Eigen-Solver Crossover Measurement Table (`V = 16..4096`, `/tmp/venv-A2`, SciPy 1.17.1)

Connected $4$-regular bipartite multigraphs ($E = 2V$, single-core `OPENBLAS_NUM_THREADS=1`). Ground truth: `np.linalg.eigvalsh` (`float64`).

#### Table A — Adjacency Spectral Gap ($\lambda_1(A) - \lambda_2(A)$)

| $V$ | `nx.adjacency_spectrum` *(SciPy nonsym `eigvals`)* | `np.linalg.eigvalsh(A)` *(current `.venv-lrc` fallback)* | Bipartite Gram `eigvalsh(B @ B.T)` *($V/2 \times V/2$)* | Sparse Lanczos `eigsh(A, k=2, LA)` | Sparse `svds(B, k=2, LM)` | Max Err (`eigsh` / `svds`) | Winner |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 0.209 ms | 0.018 ms | **0.012 ms** | 0.326 ms | 0.507 ms | 4e-16 / 9e-16 | Bipartite Gram |
| 32 | 0.325 ms | 0.048 ms | **0.020 ms** | 2.004 ms | 0.850 ms | 4e-16 / 1e-15 | Bipartite Gram |
| 64 | 1.003 ms | 0.162 ms | **0.054 ms** | 1.992 ms | 3.784 ms | 1e-14 / 4e-16 | Bipartite Gram |
| 128 | 10.365 ms | 0.686 ms | **0.173 ms** | 4.022 ms | 7.056 ms | 9e-16 / 4e-15 | Bipartite Gram |
| 256 | 58.265 ms | 6.986 ms | **3.397 ms** | 6.128 ms | 11.862 ms | 9e-15 / 4e-15 | Bipartite Gram *(eigsh beats full `eigvalsh` here!)* |
| 512 | 378.578 ms | 38.391 ms | **8.335 ms** | 12.782 ms | 15.365 ms | 4e-16 / 0.0 | Bipartite Gram *(crossover vs Lanczos $\approx 650$)* |
| 1,024 | 2,033.2 ms | 327.61 ms | 53.36 ms | **23.97 ms** | 24.22 ms | 1e-14 / 6e-15 | **Sparse `eigsh(A)` / `svds(B)`** |
| 2,048 | 9,640.9 ms | 2,212.3 ms | 282.03 ms | 51.94 ms | **36.45 ms** | 1e-14 / 6e-15 | **Sparse `svds(B)` (61× vs `eigvalsh`, 264× vs `nx`)** |
| 4,096 | *(skipped, >60 s)* | 10,400.1 ms | 1,458.9 ms | 53.97 ms | **30.13 ms** | 2e-14 / 7e-15 | **Sparse `svds(B)` (345× vs `eigvalsh`)** |

#### Table B — Algebraic Connectivity ($\lambda_2(L)$ of Laplacian)

| $V$ | `np.linalg.eigvalsh(L)` *(current `.venv-lrc` fallback)* | `nx.algebraic_connectivity` `tracemin_lu` *(expander.py target)* | `nx.algebraic_connectivity` `tracemin_pcg` *(nx default)* | `nx.algebraic_connectivity` `lanczos` (`SM`) | Direct `eigsh(L, k=2, SA)` *(abs err)* | Shift-invert `eigsh(L, σ=-1e-5)` *(SuperLU)* | Regular shortcut `d - eigsh(A, LA)` | Winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | **0.017 ms** | 3.96 ms | 32.96 ms | 2.62 ms | 0.32 ms (3e-15) | 0.64 ms (1e-15) | 0.30 ms (3e-15) | Dense `eigvalsh` |
| 32 | **0.043 ms** | 7.99 ms | 91.89 ms | 4.35 ms | 0.64 ms (**2.6e-2!**) | 0.65 ms (2e-15) | 0.91 ms (3e-15) | Dense `eigvalsh` |
| 64 | **0.164 ms** | 4.13 ms | 12.28 ms | 2.82 ms | 1.98 ms (5e-15) | 0.94 ms (6e-15) | 1.03 ms (9e-15) | Dense `eigvalsh` |
| 128 | **0.720 ms** | 7.51 ms | 26.99 ms | 3.56 ms | 4.46 ms (4e-16) | 1.14 ms (1e-15) | 1.76 ms (9e-15) | Dense `eigvalsh` |
| 256 | 2.831 ms | 11.99 ms | 18.84 ms | 4.70 ms | 4.93 ms (2e-15) | 2.77 ms (2e-15) | **1.93 ms** (8e-15) | Dense / Regular shortcut |
| 512 | 28.93 ms | 60.58 ms | **TIMEOUT (>12 s)** | 15.76 ms | **4.40 ms** (2e-15) | 8.81 ms (8e-16) | 8.82 ms (2e-15) | **Sparse `eigsh(L, SA)`** |
| 1,024 | 181.65 ms | 300.70 ms | **TIMEOUT (>12 s)** | 26.38 ms | **12.49 ms** (2e-14) | 88.53 ms (2e-14) | 16.16 ms (2e-14) | **Sparse `eigsh(L, SA)`** |
| 2,048 | 1,401.2 ms | 1,154.2 ms | **TIMEOUT (>12 s)** | 44.69 ms | **30.71 ms** (4e-15) | 608.10 ms (3e-15) | 51.94 ms (8e-15) | **Sparse `eigsh(L, SA)` (46× vs dense)** |
| 4,096 | 9,549.4 ms | 3,487.6 ms | **TIMEOUT (>12 s)** | 45.62 ms | **25.24 ms** (3e-15) | 1,720.7 ms (1e-16) | 28.83 ms (9e-15) | **Sparse `eigsh(L, SA)` (378× vs dense, 138× vs `tracemin_lu`)** |

**Key Crossover & Numerical-Stability Conclusions:**
1. **Exact crossover $V^*$:** Dense symmetric `eigvalsh` wins for **$V \le 256$** (and Bipartite Gram `eigvalsh(B @ B.T)` wins for adjacency up to **$V \approx 650$**). Above **$V = 256$** (Laplacian) / **$V = 650$** (adjacency), sparse Lanczos (`eigsh`/`svds`) pulls ahead massively (**46×–378× faster** by $V=4096$).
2. **Only extreme eigenvalues are needed:** No code anywhere in `src/` or `simulation/` reads the full `eigenvalues` list (only `tests/test_expander.py:110` checks its length). Both `spectral_gap` ($\lambda_1 - \lambda_2$ of $A$) and `algebraic_connectivity` ($\lambda_2$ of $L$) need only **2 extreme eigenvalues**.
3. **Why sparse-LU methods (`tracemin_lu`, shift-invert `eigsh`) fail to scale on expanders:** Expander graphs have high vertex expansion (no small separators), so SuperLU's `splu` suffers massive fill-in—`tracemin_lu` is slower than dense LAPACK up to $V=1024$ and takes 3.49 s at $V=4096$, while pure SpMV Lanczos (`eigsh(..., which="SA")`) converges rapidly *precisely because* expanders have a wide spectral gap ($\lambda_2(L) \approx 0.54$), finishing in **25.2 ms**!
4. **Stability guardrails:** Never use `tracemin_pcg` (diverges/hangs on expanders for $V \ge 512$ due to uncapped `_PCGSolver` loop). Never use plain `eigsh(L, k=2, which="SA")` on small graphs ($V \le 256$): at $V=32$, default `ncv=20` lost accuracy (`2.6e-2` error) near the zero nullspace. Using dense `eigvalsh` for $V \le 256$ and Lanczos with `ncv=min(V-1, 40)` for $V > 256$ gives both peak speed and $10^{-14}$ accuracy everywhere.
5. **Spectral-gap redundancy:** On regular coset graphs (`["add","add"]`, `["mult","mult"]`), $L = dI - A \implies \lambda_2(L) \equiv d - \lambda_2(A) = \text{spectral\_gap}$ (verified to `3.5e-15` at $q=16$, `6.2e-15` at $q=64$, `3.5e-14` at $q=256$, and `3e-15..9e-15` across all $V=16..4096$). Checking `if deg_min == deg_max:` skips the entire Laplacian eigensolve in $O(1)$.

---

### 5.4 `save_bipartite_graph_plot`: Backend Cost, Lazy Imports, & `LineCollection` Fix

| $q$ ($V, E$) | Current `nx.draw` (`MultiGraph`, `FancyArrowPatch`) | `nx.draw(..., arrows=False)` | Direct `LineCollection` + labels | Direct `LineCollection` (no labels) | Speedup (`arrows=False` / `LC no-lbl`) |
|---|---:|---:|---:|---:|---:|
| 64 (25, 63) | 399.8 ms | 129.7 ms | 137.1 ms | 75.7 ms | **3.1× / 5.3×** |
| 256 (115, 255) | 1,629.1 ms | 287.9 ms | 281.4 ms | 102.0 ms | **5.7× / 16.0×** |
| 1024 (349, 1023) | 4,183.2 ms *(11.0 s w/ cold cache)* | 734.0 ms | 543.4 ms | **101.7 ms** | **5.7× / 41.2×** |

- **Lazy-import verification:** `matplotlib` and `matplotlib.pyplot` are currently **NOT lazily imported** ([`expander.py:7-8, 15`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L7-L15)). Moving `from matplotlib.backends.backend_agg import FigureCanvasAgg` and `from matplotlib.figure import Figure` inside `save_bipartite_graph_plot` saves **1.81 s** on every non-plotting import of `TamoBargCodeTwoSets` and eliminates the global `matplotlib.use("Agg")` side effect.

---

### 5.5 NetworkX Dependency Verdict

- **Can NetworkX be removed from hot paths?** **YES — trivially.** NetworkX is already absent from all decoders; the only hot/init-path call is `TamoBargCodeTwoSets.__init__` calling `create_bipartite_coset_graph`. Making `bipartite_graph()` lazy (or backing the code with compact `int32` CSR arrays) completely removes NetworkX from code construction, encoding, and decoding.
- **Is NetworkX worth keeping for analysis-only code?** **Optional / removable.** Every single analysis function in `expander.py` is actually **slower or buggier through NetworkX** than direct NumPy/SciPy equivalents:
  - Connected components ($k=1$): 15-line BFS/DSU or `scipy.sparse.csgraph.connected_components`.
  - 2-edge components ($k=2$): 45-line iterative Tarjan on CSR (**17×–84× faster** than NetworkX).
  - Adjacency spectrum / spectral gap: direct NumPy Gram / `scipy.sparse.linalg.svds` (**5×–345× faster** than `nx.adjacency_spectrum`).
  - Algebraic connectivity: direct NumPy / `scipy.sparse.linalg.eigsh` (**138× faster** than `nx.algebraic_connectivity(method="tracemin_lu")` at $V=4096$).
  - Edge expansion on bipartite cut: 1-line arithmetic `E / min(|L|, |R|)`.
  - Plotting: direct Matplotlib `LineCollection` (**41× faster** than `nx.draw`).
  - **Only one niche feature** genuinely benefits from NetworkX: arbitrary $k \ge 3$ edge-connected components (`nx.gomory_hu_tree`). If preserving the public API `bipartite_graph() -> (nx.MultiGraph, set, set)` for backwards compatibility (`tests/test_expander.py`), keep `networkx` as a **lazy / optional analysis extra** (`pip install tamo-barg-lrc[analysis]`) and never import it during `TamoBargCodeTwoSets.__init__`.

---

### 5.6 Sage Dependencies & Field-Element Keying / Hashing Cost

1. **Direct Sage dependencies in `expander.py`:** **Zero direct `sage.*` imports.** The sole dependency is transitive: [`expander.py:13`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L13) imports `field_element_to_int` from `tamo_barg_lrc.subgroups` (which imports `sage.rings.finite_rings.finite_field_constructor.GF`, `VectorSpace`, `divisors`) solely for a legacy fallback line (`expander.py:194`) never reached by `create_bipartite_coset_graph`. Lazy-importing it inside that `elif` branch makes `expander.py` **100% Sage-free**.
2. **How field elements are keyed in graphs:** `create_bipartite_coset_graph` keys all graph vertices by sequential Python `int` (`0..V-1`), never hashing Sage field elements inside NetworkX. Field elements are hashed **once** upstream in [`tamo_barg_two_sets.py:272, 280`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L272-L280) (`pt_to_idx = {pt: idx for idx, pt in enumerate(self._evaluation_points)}`), using `FiniteField_givaroElement.__hash__` ([`element_givaro.pyx:1663-1675`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/rings/finite_rings/element_givaro.pyx#L1663-L1675)) for $q < 2^{16}$ and `FiniteField_ntl_gf2eElement.__hash__` ([`element_ntl_gf2e.pyx:1169-1180`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/rings/finite_rings/element_ntl_gf2e.pyx#L1169-L1180)) for $q \ge 2^{16}$, both calling `hash(self._integer_representation())`.

| $q$ | $N$ | Sage Element Type | Dict Build: Elem vs `int` | Dict Lookup: Elem vs `int` | `field_element_to_int` ($N$ calls) |
|---:|---:|---|---:|---:|---:|
| 16 | 16 | `FiniteField_givaroElement` | 0.002 vs 0.001 ms (**2.90×**) | 0.001 vs 0.001 ms (**2.46×**) | 0.005 ms |
| 64 | 64 | `FiniteField_givaroElement` | 0.006 vs 0.002 ms (**3.00×**) | 0.005 vs 0.002 ms (**3.11×**) | 0.017 ms |
| 256 | 256 | `FiniteField_givaroElement` | 0.021 vs 0.006 ms (**3.30×**) | 0.019 vs 0.006 ms (**3.38×**) | 0.070 ms |
| 1,024 | 1,024 | `FiniteField_givaroElement` | 0.098 vs 0.024 ms (**4.02×**) | 0.081 vs 0.021 ms (**3.84×**) | 0.292 ms |
| 4,096 | 4,096 | `FiniteField_givaroElement` | 0.396 vs 0.097 ms (**4.09×**) | 0.304 vs 0.080 ms (**3.82×**) | **5.627 ms** |
| 65,536 | 4,096 | `FiniteField_ntl_gf2eElement` | 0.671 vs 0.103 ms (**6.49×**) | 0.567 vs 0.079 ms (**7.14×**) | **5.625 ms** |

**Takeaway:** Keying graph vertices by `int` IDs was the right design decision—keying by Sage field elements would add **3×–4×** hashing/lookup overhead for Givaro ($q \le 4096$) and **6.5×–7.1×** for NTL ($q \ge 65536$). Even in pure-Python `field_element_to_int`, the per-element Cython/Python dispatch costs **5.63 ms** for 4,096 elements (explaining why the legacy `"coset"` fallback in `get_error_position_subgraph` is ~2× slower).

---

## 6. Prioritized Optimization Roadmap

| Priority | Change | Target Function(s) | Measured / Expected Gain |
|---|---|---|---|
| **P0** | Lazy-import `matplotlib` (`Figure` + `FigureCanvasAgg`, no global `matplotlib.use("Agg")`) and lazy-import `expander` in `TamoBargCodeTwoSets.bipartite_graph()` | `expander.py` top-level, `save_bipartite_graph_plot`, `TamoBargCodeTwoSets.__init__` | **−1.81 s** on every non-plotting import; **−3.0 MB** & **−29 ms** at $q=4096$ when graph is unused |
| **P0** | Replace `nx.draw` `MultiGraph` `FancyArrowPatch` rendering with `arrows=False` or direct `LineCollection` + `scatter` (disable labels for $V > 64$) | `save_bipartite_graph_plot` | **5.7×** (`arrows=False`) to **41.2×** (`LineCollection`: `4.18 s → 0.10 s` at $q=1024$; `39 s → ~0.2 s` at $q=4096$) |
| **P1** | Fix spectral solvers: remove `nx.adjacency_spectrum` (`dgeev`) & `tracemin_lu` (SuperLU fill-in); use dense Bipartite Gram / in-place `eigvalsh` for $V \le 256$, sparse Lanczos (`svds`/`eigsh`) for $V > 256$, regular-graph shortcut $\lambda_2(L) = d - \lambda_2(A)$, single BFS, full-precision `spectral_gap` before rounding, and `include_spectrum=False` default for `runner.py` | `_compute_adjacency_eigenvalues`, `_compute_algebraic_connectivity`, `compute_expansion_metrics` | **>2,000×** for `runner.py` (skips unused eigensolves); **63×–378×** when spectral metrics are requested at $V=4096$; halves peak RAM |
| **P1** | Lazy-import `field_element_to_int` inside the legacy `elif "coset"` fallback | `expander.py:13, 194` | Severs sole transitive Sage dependency in `expander.py` |
| **P2** | Replace multigraph subdivision in `_compute_k_edge_components` with iterative Tarjan on CSR ($k=2$) and Gomory–Hu on collapsed weighted graph ($k \ge 3$); cache components on immutable graph | `_compute_k_edge_components`, `get_error_position_subgraph` | **17×–84×** faster ($k=2$), **5.6×–17.8×** faster ($k=3$), **6.2×** faster ($k=1$ at $q=4096$) |
