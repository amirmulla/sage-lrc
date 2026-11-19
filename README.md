# tamo_barg_lrc

**Optimal Locally Recoverable Codes (LRCs)** in SageMath, implementing single-locality and two-set (multi-locality) Tamo–Barg algebraic constructions over finite fields $\mathbb{F}_q$, bipartite coset expander analysis, and iterative, global, and two-step error/erasure decoders.

---

## Overview

`tamo_barg_lrc` provides coding-theoretic constructions and decoding algorithms built on SageMath's finite-field, polynomial, and linear-code frameworks:

- **Single-Set Tamo–Barg Codes (`TamoBargCode`)**: Optimal $(n, k, r, \delta)$ LRCs constructed from additive or multiplicative subgroup coset partitions of $\mathbb{F}_q$ using annihilator ("good") polynomials, achieving the Singleton-like minimum distance bound
  $$d \ge n - k + 1 - \left(\left\lceil \frac{k}{r} \right\rceil - 1\right)(\delta - 1).$$
- **Two-Set Tamo–Barg Codes (`TamoBargCodeTwoSets`)**: Multi-locality LRCs defined over two transverse coset partitions (`["add", "add"]`, `["mult", "mult"]`, `["add", "mult"]`, or `["mult", "add"]`) via degree-minimizing encoding-basis intersection.
- **Bipartite Coset Expander Analysis (`tamo_barg_lrc.expander`)**: Construction of the bipartite coset intersection graph, spectral gap and algebraic connectivity metrics, trapping/stopping subgraph extraction, and headless visualization.
- **Local, Iterative, Global & Two-Step Decoders**:
  - Single-set local Lagrange erasure recovery (`ErasureDecoder`) and syndrome-gated local Reed–Solomon Berlekamp–Welch decoding (`ErrorDecoder`).
  - Two-set iterative hard-decision (`IterativeDecoder`), iterative erasure-error with DUE marking (`IterativeErasureErrorDecoder`), global GRS (`GlobalDecoder`, `GlobalErasureErrorDecoder`), and hybrid iterative + global two-step decoders (`TwoStepsDecoder`, `TwoStepsErasureErrorDecoder`).

---

## Requirements & Installation

- **Python**: `>= 3.10`
- **SageMath**: Either a system SageMath installation (`sage -python`, SageMath 9.8+) or modular [`passagemath`](https://github.com/passagemath/passagemath) wheels (`passagemath-modules`, `passagemath-flint`, `passagemath-pari`, `passagemath-groups`) in a virtual environment.
- **Python packages**: `networkx >= 2.8`, `matplotlib >= 3.6` (plus `pytest >= 7.2` and `ruff >= 0.4` for development).

Install in editable mode inside your SageMath Python environment:

```bash
sage -python -m pip install -e ".[dev]" --no-build-isolation
```

*(If using a virtual environment with modular `passagemath` already installed, run `python -m pip install -e ".[dev]" --no-build-isolation`.)*

---

## Quickstart

### 1. Single-Set `TamoBargCode`

Construct an optimal $[16, 6, r=2]$ LRC with local minimum distance $\delta = 3$ over $\mathbb{F}_{16}$ (each local group of size $r + \delta - 1 = 4$ forms a $[4, 2, 3]$ Reed–Solomon code):

```python
from sage.modules.free_module import VectorSpace
from sage.modules.free_module_element import vector
from sage.rings.finite_rings.finite_field_constructor import GF

from tamo_barg_lrc import TamoBargCode

F = GF(16, "a")
code = TamoBargCode(
    base_field=F,
    length=16,
    dimension=6,
    locality=2,
    local_minimum_distance=3,
    sub_group_type="add",
)

msg = VectorSpace(F, code.dimension()).random_element()
codeword = code.encode(msg)

# Local erasure recovery (up to d_local - 1 = 2 erasures per coset)
erased = vector(F, list(codeword))
erasure_mask = vector(GF(2), [1, 1] + [0] * 14)
erased[0] = F.zero()
erased[1] = F.zero()
recovered_erasure = code.decode_to_code((erased, erasure_mask), "ErasureDecoder")
assert recovered_erasure == codeword

# Local error decoding (up to floor((d_local - 1) / 2) = 1 error per coset)
corrupted = vector(F, list(codeword))
corrupted[0] += F.one()
recovered_error = code.decode_to_code(corrupted, "ErrorDecoder")
assert recovered_error == codeword
```

### 2. Two-Set `TamoBargCodeTwoSets`

Construct two-set Tamo–Barg codes using either disjoint additive subgroups (`["add", "add"]`) or mixed additive/multiplicative coset partitions (`["add", "mult"]`), and decode using the two-step decoders:

```python
from sage.modules.free_module import VectorSpace
from sage.modules.free_module_element import vector
from sage.rings.finite_rings.finite_field_constructor import GF

from tamo_barg_lrc import TamoBargCodeTwoSets, compute_expansion_metrics

F = GF(64, "a")

# Disjoint additive subgroups of size r + d_local - 1 = 8 over GF(64)
code_add = TamoBargCodeTwoSets(
    base_field=F,
    length=64,
    dimension=6,
    locality=[4, 4],
    local_minimum_distance=[5, 5],
    sub_group_type=["add", "add"],
)

# Mixed additive (size 8) and multiplicative (size 7 dividing 63) partitions
code_mixed = TamoBargCodeTwoSets(
    base_field=F,
    length=64,
    dimension=6,
    locality=[4, 3],
    local_minimum_distance=[5, 5],
    sub_group_type=["add", "mult"],
)

# Inspect bipartite coset graph expansion metrics
graph, left_nodes, right_nodes = code_add.bipartite_graph()
metrics = compute_expansion_metrics(graph, right_nodes)
print("Spectral gap:", metrics["spectral_gap"])

# Encode and inject errors
msg = VectorSpace(F, code_add.dimension()).random_element()
codeword = code_add.encode(msg)
received = vector(F, list(codeword))
for pos in (0, 9, 18, 27, 36):
    received[pos] += F.one()

# TwoStepsDecoder returns (decoded_codeword, num_iterations)
two_step_dec = code_add.decoder("TwoStepsDecoder", 10)
decoded, num_itr = two_step_dec.decode_to_code(received)
assert decoded == codeword

# TwoStepsErasureErrorDecoder returns (decoded_codeword, num_iterations, remaining_erasures)
two_step_ee_dec = code_add.decoder("TwoStepsErasureErrorDecoder", 10)
decoded_ee, num_itr_ee, rem_erasures = two_step_ee_dec.decode_to_code(received)
assert decoded_ee == codeword
```

### Decoder Return Conventions

| Code Class | Registered Decoder Key | `decode_to_code` Input | Return Signature |
| :--- | :--- | :--- | :--- |
| `TamoBargCode` | `"ErasureDecoder"` | `(word, erasure_vec)` | `Vector` |
| `TamoBargCode` | `"ErrorDecoder"` | `r` | `Vector` |
| `TamoBargCodeTwoSets` | `"GlobalDecoder"` | `r` | `Vector` |
| `TamoBargCodeTwoSets` | `"GlobalErasureErrorDecoder"` | `(word, erasure_vec)` | `Vector` |
| `TamoBargCodeTwoSets` | `"IterativeDecoder"` | `r` | `(codeword, num_itr)` |
| `TamoBargCodeTwoSets` | `"IterativeErasureErrorDecoder"` | `r` | `((codeword, erasure_vec), num_itr)` |
| `TamoBargCodeTwoSets` | `"TwoStepsDecoder"` | `r` | `(codeword, num_itr)` |
| `TamoBargCodeTwoSets` | `"TwoStepsErasureErrorDecoder"` | `r` | `(codeword, num_itr, num_rem_era)` |

---

## Running Tests & Simulations

### Test Suite & Linting

Run the deterministic `pytest` suite:

```bash
sage -python -m pytest -v
```

Check code formatting and lint rules with `ruff`:

```bash
ruff check src/ simulation/ tests/
ruff format --check src/ simulation/ tests/
```

### CLI Simulation Scripts

All scripts in `simulation/` accept `--help` and configurable CLI flags (`--q`, `--n`, `--k`, `--r`, `--local-min-dist`, `--subgroup-types`, `--decoder`, `--sim-itr`, `--max-err`, `--output-dir`, `--seed`, etc.):

```bash
# Residual error-weight distribution across injected error weights
sage -python simulation/error_weight.py --sim-itr 100 --max-err 15 --output-dir ./results

# Two-step (iterative + global) decoding success probability simulation
sage -python simulation/two_step_decoding_sim.py --sim-itr 100 --max-err 15 --output-dir ./results

# Bipartite expander / stopping-set error simulation
sage -python simulation/expander_sim.py --q 64 --n 64 --sim-itr 100 --max-err 15 --output-dir ./results

# Global GRS decoding threshold simulation
sage -python simulation/global_decoding_sim.py --q 64 --n 64 --sim-itr 100 --max-err 15 --output-dir ./results

# Pure iterative decoding success probability simulation
sage -python simulation/iterative_decoding_sim.py --q 64 --n 64 --r 2 5 --local-min-dist 3 3 --sim-itr 100 --max-err 10 --output-dir ./results

# Exact codeword Hamming weight distribution analysis
sage -python simulation/weight_distribution.py --q 64 --n 64 --k 3 --output-dir ./results

# Post-process residual error CSV outputs into global decoding success rates
sage -python simulation/process_results.py --help
```

---

## Package Architecture

```text
src/tamo_barg_lrc/
├── __init__.py             # SageMath initialization & PEP 562 lazy public API exports
├── subgroups.py            # Finite-field additive/multiplicative subgroups & coset partitions
├── polynomials.py          # Annihilator ("good") polynomials, encoding bases & interpolation
├── expander.py             # Bipartite coset graphs, expansion metrics & headless plotting
├── tamo_barg.py            # Single-set TamoBargCode, encoder & local error/erasure decoders
├── tamo_barg_two_sets.py   # Two-set TamoBargCodeTwoSets, encoder & iterative/global/two-step decoders
└── simulation/
    ├── __init__.py         # Simulation subpackage exports
    ├── runner.py           # Unified decoder dispatch, CSV metadata writers & Monte Carlo runners
    └── postprocess.py      # Residual error CSV post-processing utilities
```

---

## License

Licensed under the [MIT License](LICENSE).