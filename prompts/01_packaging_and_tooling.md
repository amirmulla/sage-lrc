# Agent 01 — Packaging, Tooling & Documentation Architect (Wave 1)

## Role & Mission
You are the **Packaging, Tooling & Documentation Architect** in Wave 1 of the `tamo_barg_lrc` refactoring swarm. First read [`prompts/shared_context_and_contracts.md`](./shared_context_and_contracts.md) completely.

## Exclusive File Ownership
You may **ONLY** create or modify the following files:
* `pyproject.toml` *(create)*
* `.gitignore` *(create or update)*
* `src/tamo_barg_lrc/__init__.py` *(create)*
* `README.md` *(replace)*

Do **NOT** touch any other files in `src/`, `simulation/`, or `tests/`.

---

## Detailed Implementation Specifications

### 1. `pyproject.toml`
Create a modern PEP 621 `pyproject.toml` configured for `src/` layout, SageMath compatibility, `ruff` formatting/linting, and `pytest`:

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "tamo-barg-lrc"
version = "0.2.0"
description = "Optimal Locally Recoverable Codes (single-set and two-set Tamo-Barg codes) in SageMath"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.10"
dependencies = [
    "networkx>=2.8",
    "matplotlib>=3.6",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.2",
    "ruff>=0.4",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"

[tool.ruff]
line-length = 100
target-version = "py310"
src = ["src", "simulation", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]
ignore = []

[tool.ruff.lint.per-file-ignores]
"src/tamo_barg_lrc/__init__.py" = ["F401"]
```

*(Note: SageMath itself is provided by the host `sage -python` runtime and must NOT be listed as a PyPI install dependency in `[project.dependencies]`, because monolithic `sage` is not a standard wheel.)*

---

### 2. `.gitignore`
Ensure `.gitignore` cleanly excludes build/test/simulation artifacts:
* `__pycache__/`, `*.py[cod]`, `*.egg-info/`, `build/`, `dist/`, `.eggs/`
* `.pytest_cache/`, `.ruff_cache/`, `.coverage`
* `results/`, `*.png` (generated bipartite graph plots), `*.sage.py`
* `.venv/`, `venv/`, `.DS_Store`

---

### 3. `src/tamo_barg_lrc/__init__.py`
Implement `src/tamo_barg_lrc/__init__.py` with:
1. One-time SageMath runtime initialization (`import sage.all  # noqa: F401`) so importing any submodule under `sage -python` initializes Sage's Cython coercion and category framework reliably.
2. **PEP 562 lazy `__getattr__` exports** so Wave 1 submodules (`tamo_barg_lrc.subgroups`, `tamo_barg_lrc.polynomials`, `tamo_barg_lrc.expander`) can be imported immediately during Wave 1 **before** Wave 2 creates `tamo_barg.py` and `tamo_barg_two_sets.py`, while `from tamo_barg_lrc import TamoBargCode, TamoBargCodeTwoSets, ...` works seamlessly once Wave 2 completes!

Exact export mapping required in `__all__` and `__getattr__`:
* From `tamo_barg_lrc.tamo_barg`:
  * `TamoBargCode`, `TamoBargVectorEncoder`, `TamoBargErasureDecoder`, `TamoBargErrorDecoder`
* From `tamo_barg_lrc.tamo_barg_two_sets`:
  * `TamoBargCodeTwoSets`, `TamoBargTwoSetsVectorEncoder`, `TamoBargIterativeDecoder`, `TamoBargIterativeEEDecoder`, `TamoBargGlobalDecoder`, `TamoBargGlobalEEDecoder`, `TamoBargTwoStepDecoder`, `TamoBargTwoStepEEDecoder`
* From `tamo_barg_lrc.subgroups`:
  * `field_element_to_int`, `sort_field_elements`, `is_closed_under_addition`, `find_subspaces`, `find_additive_subgroups`, `find_additive_subgroups_bruteforce`, `list_multiplicative_subgroups`, `list_canonical_additive_subgroups`, `shift_additive_subgroup`, `list_subgroup_cosets`
* From `tamo_barg_lrc.expander`:
  * `create_bipartite_coset_graph`, `get_error_position_subgraph`, `compute_expansion_metrics`, `save_bipartite_graph_plot`

Also implement `__dir__()` returning `sorted(list(globals().keys()) + __all__)` and `TYPE_CHECKING` imports for static type checkers.

---

### 4. `README.md`
Replace the minimal 4-line `README.md` with clear, concise documentation covering:
1. **Overview**: Single-locality (`TamoBargCode`) and two-set / multi-locality (`TamoBargCodeTwoSets`) optimal Locally Recoverable Codes over finite fields $GF(q)$, bipartite coset expander analysis, and iterative / global / two-step error & erasure decoders.
2. **Requirements & Installation**:
   ```bash
   sage -python -m pip install -e ".[dev]" --no-build-isolation
   ```
3. **Quickstart Python Examples**:
   * Single-set `TamoBargCode` construction, encoding, erasure decoding, and local error decoding.
   * Two-set `TamoBargCodeTwoSets` construction (`["add", "add"]` and `["add", "mult"]`), encoding, and decoding with `TwoStepsDecoder` / `TwoStepsErasureErrorDecoder` (showing the tuple return convention).
4. **Running Tests & Simulations**:
   * `sage -python -m pytest -v`
   * CLI examples for `simulation/error_weight.py`, `simulation/two_step_decoding_sim.py`, `simulation/expander_sim.py`, `simulation/global_decoding_sim.py`, `simulation/iterative_decoding_sim.py`, `simulation/weight_distribution.py`, and `simulation/process_results.py`.
5. **Package Architecture**: Brief module map of `src/tamo_barg_lrc/`.

---

## Acceptance Checklist
- [ ] `ruff check pyproject.toml src/tamo_barg_lrc/__init__.py` passes with zero warnings.
- [ ] `ruff format --check src/tamo_barg_lrc/__init__.py` passes.
- [ ] `sage -python -m pip install -e . --no-build-isolation` succeeds cleanly even before Wave 2 files exist.
- [ ] No files outside `pyproject.toml`, `.gitignore`, `src/tamo_barg_lrc/__init__.py`, and `README.md` were created or modified.
