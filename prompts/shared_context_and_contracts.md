# Shared Context, API Contracts & Defect Manifest (`optimal_lrc`)

> **MANDATORY FOR ALL AGENTS**: Every agent in the refactoring swarm MUST adhere strictly to the file-ownership boundaries, module/class/function signatures, decoder return conventions, and coding standards specified in this document. Do **not** invent alternative module paths, rename registered decoder keys, or touch files outside your assigned scope.

---

## 1. Architectural Decisions (Non-Negotiable)

1. **SageMath Backend Preserved**: Keep SageMath (`sage.coding`, `GF`, `PolynomialRing`, `matrix`, `vector`, `VectorSpace`, `GeneralizedReedSolomonCode`) as the algebraic and coding-theory foundation.
   * Initialize `import sage.all  # noqa: F401` once inside `src/optimal_lrc/__init__.py` so `sage -python` initializes Sage's Cython coercion/category system cleanly.
   * **FORBIDDEN**: Wildcard imports (`from sage.all import *` or `from networkx.algorithms import *`) anywhere in `src/`, `simulation/`, or `tests/`. Always use explicit imports.
2. **Clean `src/` Layout Package**: Migrate to `pyproject.toml` + `src/optimal_lrc/` with snake_case modules. Legacy PascalCase files (`src/AdditiveSubgroup.py`, `src/Expander.py`, `src/TamoBarg.py`, `src/TamoBargTwoSets.py`), `simulation/proces_results.py`, and ad-hoc `tests/*_Test.py` scripts are deleted in Wave 3 after replacements are complete—do **not** leave backward-compatibility shims in `src/`.
3. **Legacy Decoder Tuple Return Signatures Preserved**: Two-set iterative and two-step decoders MUST retain their historical tuple return formats (`(codeword, num_itr, ...)`), while single-set and global decoders return a single `Vector` (see Section 4).

---

## 2. File Ownership Matrix

Parallel agents within the same wave MUST ONLY create, modify, or delete files listed in their row:

| Agent | Wave | Exclusive File Ownership |
| :--- | :---: | :--- |
| **Agent 01** | 1 | `pyproject.toml`, `.gitignore`, `src/optimal_lrc/__init__.py`, `README.md` |
| **Agent 02** | 1 | `src/optimal_lrc/subgroups.py`, `src/optimal_lrc/polynomials.py` |
| **Agent 03** | 1 | `src/optimal_lrc/expander.py` |
| **Agent 04** | 2 | `src/optimal_lrc/tamo_barg.py` |
| **Agent 05** | 2 | `src/optimal_lrc/tamo_barg_two_sets.py` |
| **Agent 06** | 3 | `src/optimal_lrc/simulation/__init__.py`, `src/optimal_lrc/simulation/runner.py`, `src/optimal_lrc/simulation/postprocess.py`, `simulation/error_weight.py`, `simulation/expander_sim.py`, `simulation/global_decoding_sim.py`, `simulation/iterative_decoding_sim.py`, `simulation/process_results.py`, `simulation/two_step_decoding_sim.py`, `simulation/weight_distribution.py`, **DELETE** `simulation/proces_results.py` |
| **Agent 07** | 3 | `tests/conftest.py`, `tests/test_subgroups.py`, `tests/test_expander.py`, `tests/test_tamo_barg.py`, `tests/test_tamo_barg_two_sets.py`, `tests/test_simulations.py`, `tests/Readme.md`, **DELETE** `src/AdditiveSubgroup.py`, `src/Expander.py`, `src/TamoBarg.py`, `src/TamoBargTwoSets.py`, `tests/Subgroups_Test.py`, `tests/TamoBargTwoSets_EE_Iterative_Test.py`, `tests/TamoBargTwoSets_Group_Test.py`, `tests/TamoBargTwoSets_Test.py`, `tests/TamoBarg_Erasure_Test.py`, `tests/TamoBarg_Error_Test.py`, `tests/TamoBarg_Expander.py`, `tests/TamoBarg_Expansion_Measure.py` |
| **Agent 08** | 4 | Read-only QA, lint, pytest, CLI smoke verification & benchmarking (fixes routed via Orchestrator if needed) |

---

## 3. Frozen Cross-Module API Signatures

### `optimal_lrc/__init__.py` (Agent 01)
Exports the primary public API (using direct imports once Wave 2 completes, or guarded/lazy `__getattr__` during Wave 1 so package installation works before Wave 2):
* `TamoBargCode`, `TamoBargVectorEncoder`, `TamoBargErasureDecoder`, `TamoBargErrorDecoder` (from `optimal_lrc.tamo_barg`)
* `TamoBargCodeTwoSets`, `TamoBargTwoSetsVectorEncoder`, `TamoBargIterativeDecoder`, `TamoBargIterativeEEDecoder`, `TamoBargGlobalDecoder`, `TamoBargGlobalEEDecoder`, `TamoBargTwoStepDecoder`, `TamoBargTwoStepEEDecoder` (from `optimal_lrc.tamo_barg_two_sets`)
* `find_additive_subgroups`, `find_additive_subgroups_bruteforce`, `find_subspaces`, `is_closed_under_addition`, `list_multiplicative_subgroups`, `list_canonical_additive_subgroups`, `list_subgroup_cosets`, `shift_additive_subgroup` (from `optimal_lrc.subgroups`)
* `create_bipartite_coset_graph`, `get_error_position_subgraph`, `compute_expansion_metrics`, `save_bipartite_graph_plot` (from `optimal_lrc.expander`)

### `optimal_lrc/subgroups.py` (Agent 02)
```python
from collections.abc import Iterable, Sequence
from typing import Any, Literal

SubgroupType = Literal["add", "mult"]

def field_element_to_int(elem: Any) -> int: ...
def sort_field_elements(elements: Iterable[Any]) -> list[Any]: ...
def is_closed_under_addition(subgroup: Iterable[Any]) -> bool: ...
def find_subspaces(dim: int, ambient_dim: int, base_field: Any) -> list[Any]: ...
def find_additive_subgroups(field: Any, size: int) -> list[list[Any]]: ...
def find_additive_subgroups_bruteforce(
    field: Any, size: int, max_subgroups: int = 1000
) -> list[list[Any]]: ...
def list_multiplicative_subgroups(field: Any) -> dict[int, list[Any]]: ...
def list_canonical_additive_subgroups(field: Any) -> dict[int, list[Any]]: ...
def shift_additive_subgroup(
    field: Any, subgroup: Sequence[Any], shift_exponent: int
) -> list[Any]: ...
def list_subgroup_cosets(
    field: Any, subgroup: Sequence[Any], subgroup_type: SubgroupType
) -> list[list[Any]]: ...
```

### `optimal_lrc/polynomials.py` (Agent 02)
```python
from collections.abc import Sequence
from typing import Any, Literal

def build_annihilator_polynomial(poly_ring: Any, evaluation_points: Sequence[Any]) -> Any: ...
def build_good_polynomial(
    poly_ring: Any,
    subgroup: Sequence[Any],
    subgroup_type: Literal["add", "mult"],
    annihilator: Any | None = None,
) -> Any: ...
def build_algebra_basis(
    good_poly: Any, partition_size: int, annihilator: Any | None = None
) -> list[Any]: ...
def build_encoding_basis(
    poly_ring: Any,
    algebra_basis: Sequence[Any],
    locality: int,
    annihilator: Any | None = None,
) -> list[Any]: ...
def intersect_encoding_bases(
    poly_ring: Any,
    basis_a: Sequence[Any],
    basis_b: Sequence[Any],
    length: int,
) -> list[Any]: ...
def evaluate_message_polynomial(
    poly_ring: Any, basis: Sequence[Any], message: Sequence[Any]
) -> Any: ...
def interpolate_erasures(
    poly_ring: Any,
    eval_points: Sequence[Any],
    values: Sequence[Any],
    erasure_mask: Sequence[Any],
) -> list[Any]: ...
```

### `optimal_lrc/expander.py` (Agent 03)
```python
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import networkx as nx

def create_bipartite_coset_graph(
    partitions: Sequence[Sequence[Sequence[Any]]],
    evaluation_points_idx: Sequence[Sequence[Sequence[int]]],
    evaluation_points: Sequence[Any],
) -> tuple[nx.MultiGraph, set[int], set[int]]: ...

def get_error_position_subgraph(
    graph: nx.Graph, h_idx: int = 0, edge_connectivity_k: int = 1
) -> list[int]: ...

def compute_expansion_metrics(
    graph: nx.Graph, right_nodes: set[int] | None = None
) -> dict[str, Any]: ...

def save_bipartite_graph_plot(
    graph: nx.Graph,
    left_nodes: set[int],
    right_nodes: set[int],
    output_path: str | Path,
) -> None: ...
```

### `optimal_lrc/tamo_barg.py` (Agent 04)
* `TamoBargCode(AbstractLinearCode)`:
  * `__init__(self, base_field, length: int, dimension: int, locality: int, local_minimum_distance: int = 2, sub_group_type: Literal["any", "mult", "add"] = "any")`
  * Methods: `partition() -> tuple[list[list[Any]], int]`, `sub_group() -> tuple[list[Any], str]`, `evaluation_points() -> list[Any]`, `locality() -> int`, `local_minimum_distance() -> int`, `max_dimension() -> int`, `minimum_distance() -> int`
  * Registered encoder: `"VectorEncoder": TamoBargVectorEncoder`
  * Registered decoders: `"ErasureDecoder": TamoBargErasureDecoder`, `"ErrorDecoder": TamoBargErrorDecoder`
* `TamoBargVectorEncoder(Encoder)`:
  * Methods: `enc_basis() -> list[Any]`, `enc_algebra_basis() -> list[Any]`, `enc_good_poly() -> Any`, `enc_poly(message) -> Any`, `max_dimension() -> int`, `generator_matrix()` (`@cached_method`)

### `optimal_lrc/tamo_barg_two_sets.py` (Agent 05)
* `TamoBargCodeTwoSets(AbstractLinearCode)`:
  * `__init__(self, base_field, length: int, dimension: int, locality: Sequence[int], local_minimum_distance: Sequence[int] | int = (2, 2), sub_group_type: Sequence[Literal["add", "mult"]] = ("add", "add"), shift_add: bool = False, subgroup: Sequence[Sequence[Any] | None] | None = None)`
  * Methods: `bipartite_graph() -> tuple[nx.MultiGraph, set[int], set[int]]`, `partition() -> tuple[list[list[list[Any]]], list[int]]`, `sub_group() -> tuple[list[list[Any]], list[str]]`, `evaluation_points_idx() -> list[list[list[int]]]`, `evaluation_points() -> list[Any]`, `locality() -> list[int]`, `num_of_sets() -> int`, `local_minimum_distance() -> list[int]`, `design_distance() -> int`, `max_dimension() -> int`, `max_rate() -> Any`, `minimum_distance_py() -> int`
  * Registered encoder: `"VectorEncoder": TamoBargTwoSetsVectorEncoder`
  * Registered decoders:
    * `"IterativeDecoder": TamoBargIterativeDecoder`
    * `"TwoStepsDecoder": TamoBargTwoStepDecoder`
    * `"GlobalDecoder": TamoBargGlobalDecoder`
    * `"IterativeErasureErrorDecoder": TamoBargIterativeEEDecoder`
    * `"GlobalErasureErrorDecoder": TamoBargGlobalEEDecoder`
    * `"TwoStepsErasureErrorDecoder": TamoBargTwoStepEEDecoder`
* `TamoBargTwoSetsVectorEncoder(Encoder)`:
  * Methods: `enc_basis() -> list[Any]`, `enc_basis_ext() -> list[list[Any]]`, `enc_algebra_basis() -> list[list[Any]]`, `enc_good_poly() -> list[Any]`, `enc_poly(message) -> Any`, `max_dimension() -> int`, `generator_matrix()` (`@cached_method`)

---

## 4. Canonical Decoder Signature & Return Contract Table

Every agent implementing or calling a decoder MUST follow this exact contract:

| Decoder Class | Registered Key | `__init__` Signature | `input_space` | `decode_to_code` Input | `decode_to_code` Return |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TamoBargErasureDecoder` | `"ErasureDecoder"` | `(code)` | `cartesian_product([GF(q)^n, GF(2)^n])` | `(word, erasure_vec)` | `Vector` (`GF(q)^n`) |
| `TamoBargErrorDecoder` | `"ErrorDecoder"` | `(code)` | `GF(q)^n` | `r` | `Vector` (`GF(q)^n`) |
| `TamoBargGlobalDecoder` | `"GlobalDecoder"` | `(code)` | `GF(q)^n` | `r` | `Vector` (`GF(q)^n`) |
| `TamoBargGlobalEEDecoder` | `"GlobalErasureErrorDecoder"` | `(code)` | `cartesian_product([GF(q)^n, GF(2)^n])` | `(word, erasure_vec)` | `Vector` (`GF(q)^n`) |
| `TamoBargIterativeDecoder` | `"IterativeDecoder"` | `(code, max_num_of_itr=10)` | `GF(q)^n` | `r` | `(Vector, int)` -> `(corr_c, num_itr)` |
| `TamoBargIterativeEEDecoder` | `"IterativeErasureErrorDecoder"` | `(code, max_num_of_itr=10, erasure_start_itr=None)` | `GF(q)^n` | `r` | `((Vector, Vector), int)` -> `((corr_c, erasure_vec), num_itr)` |
| `TamoBargTwoStepDecoder` | `"TwoStepsDecoder"` | `(code, max_num_of_itr=10)` | `GF(q)^n` | `r` | `(Vector, int)` -> `(corr_c, num_itr)` |
| `TamoBargTwoStepEEDecoder` | `"TwoStepsErasureErrorDecoder"` | `(code, max_num_of_itr=10, erasure_start_itr=None)` | `GF(q)^n` | `r` | `(Vector, int, int)` -> `(corr_c, num_itr, num_rem_era)` |

**CRITICAL INVARIANT FOR ALL DECODERS**: `decode_to_code` MUST NEVER mutate its input vector `r` (or `word` / `erasure_vec`) in-place! Always create a fresh mutable copy (`work = copy(vector(self._F, r))`) before modifying entries.

---

## 5. Master Bug (`B1`–`B18`) & Performance (`P1`–`P6`) Manifest

### Bugs to Fix
* **B1 (Agent 05)**: `uncorrectable_err[0] == False & uncorrectable_err[1] == False` in `TamoBargTwoSets.py:452, 546` evaluates `False & uncorrectable_err[1]` first (`False`), turning into `uncorrectable_err[0] == False == False` and completely ignoring Partition 1 errors! Fix with `not any(uncorrectable_err)` + convergence checks.
* **B2 (Agent 05)**: `TamoBargIterativeEEDecoder` (`TamoBargTwoSets.py:515-539`) calls `ErrorErasure` *before* checking `erasure_weight == d_local - 1`, throwing an exception that appends `due = True` and sets `uncorrectable_err[i] = True`, then recovers via Lagrange without clearing `due` or `uncorrectable_err`. Check `erasure_weight == d_local - 1` *first* (`if ... elif ...`).
* **B3 (Agent 05)**: Typo `self.code().evaluation_ponts()` in `TamoBargGlobalEEDecoder` (`TamoBargTwoSets.py:603`) raises `AttributeError`. Fix to `evaluation_points()`.
* **B4 (Agents 04 & 05)**: In-place mutation of caller's received vector `r` in iterative, global-EE, and erasure decoders (`TamoBargTwoSets.py:450, 542, 616`). Always copy input vectors at decoder entry.
* **B5 (Agents 04 & 05)**: `TamoBargErrorDecoder` (`TamoBarg.py:267-271`) wrongly sets `input_space` to Cartesian `(GF(q)^n, GF(2)^n)` and `_repr_` to `"Erasure Decoder..."`; `TamoBargGlobalEEDecoder` (`TamoBargTwoSets.py:590`) wrongly sets `input_space` to `GF(q)^n` instead of the Cartesian product. Fix both.
* **B6 (Agents 04 & 05)**: `enc_poly()` references undefined `self._enc_poly` (`AttributeError`), and `TamoBargVectorEncoder` in `TamoBarg.py` forgets to slice `_enc_basis` to `code.dimension()`. Fix both (`enc_poly(message)` computes $\sum_{i=0}^{k-1} m_i b_i(x)$).
* **B7 (Agent 02)**: `AdditiveSubgroup.py:94-101` uses `''.join(str(i) for i in vec)` and `int(binary_string, p)`, corrupting multi-digit elements for $p \ge 10$, crashing for $p > 36$, and failing symbolically when $n$ is not a prime power. Validate $n = p^t$ and map $\mathbb{F}_p^m \to GF(p^m)$ directly via powers of `field.gen()`.
* **B8 (Agent 02)**: `_list_add_subgroups` omits $i=0$ ($\{0\}$) and $i=\deg(F)$ ($GF(q)$); `_list_sub_group_cosets` iterates over $F^\times$ (missing $\{0\}$ coset when $H=\{0\}$) and uses `set(...)` giving non-deterministic coset order. Fix range to `0..deg(F)` and generate cosets deterministically over $F$ (for `"add"`) and $F^\times$ (for `"mult"`).
* **B9 (Agent 05)**: In `TamoBargTwoSets.py:102-110`, `shift_add=True` with `["add", "add"]` shifts *both* `i=0` and `i=1` by `a**t_0`, making both subgroups identical. Shift only `i=1` for `["add", "add"]`, and shift the `"add"` subgroup only once in mixed `["add", "mult"]` / `["mult", "add"]` modes when `shift_add=True`.
* **B10 (Agent 05)**: `sub_group_type = ["mult", "add"]` crashes with `ValueError: 0 is not in list` because `_evaluation_points` was built only from `_partition[0]`; `num_of_sets == 1` branch in `TamoBargCodeTwoSets` is broken dead code. Remove dead 1-set branch (require 2 sets) and build `_evaluation_points` as the canonically sorted union of all partitions' cosets, validating `len(_evaluation_points) == length`.
* **B11 (Agents 04 & 05)**: Missing subgroups only `print(...)` and crash with `AttributeError`; non-divisible `length` truncates evaluation points and crashes in `generator_matrix()` with `IndexError`; `dimension > max_dimension` is checked too late in `TamoBarg.py`. Validate all parameters upfront in `__init__` and raise `ValueError`.
* **B12 (Agent 03)**: `Expander.py:13` converts field elements via `int(str(x))`, crashing on polynomial representations and producing out-of-bounds indices when $0 \notin \text{evaluation\_points}$. Store `coset_indices` directly on graph nodes and return those.
* **B13 (Agents 02, 04, 05)**: Bitwise `&` precedence bug `j**i == 1 & first_identity_found == 1`, plus class/attribute typos `TamoBargIterariveDecoder`, `TamoBargIterariveEEDecoder`, `_parition_size`. Replace all boolean `&` with `and` and fix all names.
* **B14 (Agent 06)**: `proces_results.py:34` computes `design_distnace - 1 // 2` (`15 - 0 = 15` instead of `(15 - 1) // 2 = 7`) due to `//` binding tighter than `-`. Fix to `(design_distance - 1) // 2` in `process_results.py`.
* **B15 (Agent 06)**: Simulation scripts pass `max_num_of_itr` to `GlobalDecoder` / `GlobalErasureErrorDecoder` (`TypeError`) and fail to unpack `IterativeErasureErrorDecoder` / `GlobalDecoder` return values. Centralize instantiation and return unpacking in `optimal_lrc.simulation.runner`.
* **B16 (Agent 06)**: `print_freq = int(sim_itr / print_freq_factor)` raises `ZeroDivisionError` when `sim_itr < print_freq_factor`. Use `max(1, sim_itr // print_freq_factor)`.
* **B17 (Agents 04 & 05)**: Bare `except:` clauses swallow `KeyboardInterrupt` and hide bugs. Catch `(DecodingError, ValueError)` explicitly where Sage decoders signal decoding failure.
* **B18 (Agents 03, 06, 07)**: Leaked file handles (`open` without `with`), leaked `matplotlib` figures, duplicate CSV rows in `weight_distribution.py`, and blocking `plt.show()` in tests. Fix all resource management.

### Performance Optimizations to Implement
* **P1 (Agents 04 & 05)**: Precompute local GRS codes, decoders, and parity-check matrices $H_{i,j}$ in decoder `__init__`; check `H_{i,j} * r_coset == 0` in Cython/FLINT to skip error-free cosets ($10\times$–$50\times$ iterative decoding speedup).
* **P2 (Agent 05)**: Add convergence & fixed-point stall detection in iterative decoders (halt immediately when both partitions are syndrome-clean OR when `(r, erasure_vector)` stopped changing after erasure marking has started).
* **P3 (Agent 02)**: $O(q)$ multiplicative subgroup construction via `divisors(q - 1)` + primitive element powers, $O(q)$ coset enumeration via `visited` set, and lazy generator + `set` closure in `find_additive_subgroups_bruteforce`.
* **P4 (Agents 03 & 05)**: $O(n)$ dict-based evaluation-point indexing and $O(n)$ bipartite graph edge construction; $O(|V|+|E|)$ `nx.connected_components` fast-path for `k=1` in `get_error_position_subgraph`.
* **P5 (Agent 02)**: Direct univariate `PolynomialRing(F, 'x')` arithmetic (eliminating `PolynomialQuotientRing` wrapper overhead), FLINT product-tree annihilator, `poly_ring(list(row))` conversion, and reverse-column RREF for degree-minimizing two-set intersection bases.
* **P6 (Agent 05)**: $(q - 1)\times$ speedup in `minimum_distance_py()` by iterating only over monic (projectively normalized, first non-zero coordinate $= 1$) message vectors in $GF(q)^k$.

---

## 6. Code Formatting & Style Standards
* Python **3.10+** type annotations (`list[int]`, `tuple[...]`, `X | None`).
* Line length: **100 characters** (`ruff format` / `ruff check` with `E, F, I, B, UP, SIM`).
* Concise Google-style docstrings on all public classes, methods, and functions.
* Zero commented-out dead code blocks, zero stray characters (e.g. `#åDec`), zero typos in log/print strings (`Received`, `Successful`, `Remaining`, `Distance`, `Algebraic`).
