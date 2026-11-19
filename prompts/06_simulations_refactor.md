# Agent 06 — Simulation Engine, Post-Processing & CLI Scripts Specialist (Wave 3)

## Role & Mission
You are the **Simulation Engine, Post-Processing & CLI Scripts Specialist** in Wave 3 of the `tamo_barg_lrc` refactoring swarm. First read [`prompts/shared_context_and_contracts.md`](./shared_context_and_contracts.md) completely, then inspect all 7 legacy scripts in `simulation/` plus the Wave 1–2 package modules in `src/tamo_barg_lrc/`.

## Exclusive File Ownership
You may **ONLY** create, modify, or delete:
* `src/tamo_barg_lrc/simulation/__init__.py` *(create)*
* `src/tamo_barg_lrc/simulation/runner.py` *(create)*
* `src/tamo_barg_lrc/simulation/postprocess.py` *(create)*
* `simulation/error_weight.py` *(refactor)*
* `simulation/expander_sim.py` *(refactor)*
* `simulation/global_decoding_sim.py` *(refactor)*
* `simulation/iterative_decoding_sim.py` *(refactor)*
* `simulation/process_results.py` *(create — replaces `proces_results.py`)*
* `simulation/two_step_decoding_sim.py` *(refactor)*
* `simulation/weight_distribution.py` *(refactor)*
* `simulation/proces_results.py` *(**DELETE** using `git rm -f` or `rm`)*

Do **NOT** touch any files in `tests/` or core `src/tamo_barg_lrc/*.py` modules.

---

## Bugs & Code Quality Targets Assigned to You
* **B14 (CRITICAL)**: In `simulation/proces_results.py:34`, `int(row[1]) <= (design_distnace-1 // 2)` evaluates `1 // 2 == 0` before subtraction (`15 - 0 = 15` instead of `(15 - 1) // 2 = 7`), completely corrupting post-processed success probabilities! Fix to `(design_distance - 1) // 2`, fix boolean `&` on line 42, and rename to `simulation/process_results.py`.
* **B15 (CRITICAL)**:
  * `error_weight.py`, `expander_sim.py`, and `two_step_decoding_sim.py` call `C.decoder(decoder_str, max_num_of_itr)`, which crashes with `TypeError` when `decoder_str` is `"GlobalDecoder"` or `"GlobalErasureErrorDecoder"`.
  * `global_decoding_sim.py` and `iterative_decoding_sim.py` unpack `correct_c, num_of_itr = Dec.decode_to_code(r)` without handling `"IterativeErasureErrorDecoder"` returning `((corr_c, era_vec), num_itr)`.
* **B16**: All simulation scripts compute `print_freq = int(sim_itr / print_freq_factor)`, which raises `ZeroDivisionError` whenever `sim_itr < print_freq_factor`. Fix to `max(1, sim_itr // print_freq_factor)`.
* **B18**: Replace ~800 lines of duplicated setup/CSV/logging boilerplate, unclosed `open(...)` handles, duplicate CSV metadata rows (`weight_distribution.py:87-88`), and global `plt.savefig` calls with context managers (`with open(...)`) and `save_bipartite_graph_plot`.
* **Shortened Mixed-Subgroup Error Sampling Fix**: In legacy `error_weight.py` and `expander_sim.py`, when `shorten=True` (`sub_group_type[0] != sub_group_type[1]`), sampling `n_err` positions from `0..n-1` and overwriting `r[0] = c[0]` silently reduced the actual injected error weight whenever index `0` was sampled! When `shorten=True`, exclude the index of `0` from `error_positions` (and when using `StaticErrorRateChannel`, ensure exact weight or resample/exclude index `0`) so every trial injects exact weight `n_err`.

---

## Detailed Implementation Specifications

### 1. `src/tamo_barg_lrc/simulation/runner.py`
Design clean, reusable data structures and simulation runners:

* **`DecodeOutcome` (`@dataclass(frozen=True)`)**:
  ```python
  @dataclass(frozen=True)
  class DecodeOutcome:
      codeword: Any
      num_iterations: int
      remaining_erasures: int = 0
  ```
* **`build_decoder(code: TamoBargCodeTwoSets, decoder_name: str, max_num_of_itr: int = 10, erasure_start_itr: int | None = None) -> Any`** (**Fixes B15**):
  * `"GlobalDecoder"` / `"GlobalErasureErrorDecoder"`: call `code.decoder(decoder_name)` (no extra positional arguments).
  * `"IterativeDecoder"` / `"TwoStepsDecoder"`: call `code.decoder(decoder_name, max_num_of_itr)`.
  * `"IterativeErasureErrorDecoder"` / `"TwoStepsErasureErrorDecoder"`: call `code.decoder(decoder_name, max_num_of_itr, erasure_start_itr)`.
  * Raise `ValueError` for unknown `decoder_name`.
* **`invoke_decoder(decoder: Any, decoder_name: str, received_word: Any) -> DecodeOutcome`** (**Fixes B15** while respecting legacy tuple returns on `decode_to_code`!):
  * `"GlobalDecoder"`: try `corr = decoder.decode_to_code(received_word)` catching `(DecodingError, ValueError)` (fallback `copy(received_word)`), return `DecodeOutcome(corr, 1, 0)`.
  * `"GlobalErasureErrorDecoder"`: wrap `(received_word, zero_vector(GF(2), len(received_word)))`, catch `(DecodingError, ValueError)`, return `DecodeOutcome(corr, 1, 0)`.
  * `"IterativeDecoder"` / `"TwoStepsDecoder"`: `corr, num_itr = decoder.decode_to_code(received_word)` $\to$ `DecodeOutcome(corr, num_itr, 0)`.
  * `"IterativeErasureErrorDecoder"`: `(corr, era_vec), num_itr = decoder.decode_to_code(received_word)` $\to$ `DecodeOutcome(corr, num_itr, int(era_vec.hamming_weight()))`.
  * `"TwoStepsErasureErrorDecoder"`: `corr, num_itr, rem_era = decoder.decode_to_code(received_word)` $\to$ `DecodeOutcome(corr, num_itr, rem_era)`.
* **Shared Helpers**:
  * `write_code_metadata_csv(writer, code, graph_metrics, codeword, extra_rows=None)`: writes the standardized code summary header rows cleanly without duplication.
  * `sample_fixed_weight_error(field, length: int, weight: int, candidate_positions: Sequence[int])`: samples `weight` distinct positions from `candidate_positions` and assigns uniform random non-zero elements of `field` ($F \setminus \{0\}$) at those positions.
  * Runners for:
    * `run_error_weight_simulation(...)` (for `error_weight.py` and `expander_sim.py`)
    * `run_global_threshold_simulation(...)` (for `global_decoding_sim.py`)
    * `run_decoding_success_simulation(...)` (for `iterative_decoding_sim.py` and `two_step_decoding_sim.py`)
    * `run_weight_distribution_analysis(...)` (for `weight_distribution.py`)

### 2. `src/tamo_barg_lrc/simulation/postprocess.py`
* Implement `process_error_distribution_csv(err_csv_path: str | Path, design_distance: int, real_distance: int, sim_itr: int = 5000, threshold_pct: float = 99.0) -> dict[str, Any]` (**Fixes B14**):
  * Uses exact `(design_distance - 1) // 2` threshold.
  * Uses `with open(err_csv_path, newline="") as csvfile:`.
  * Computes `succ_count`, `prob`, and first index below `threshold_pct` using standard boolean `and` (no `&`).

### 3. Refactor All 7 CLI Scripts in `simulation/`
Every script in `simulation/` MUST:
1. Import directly from `tamo_barg_lrc` and `tamo_barg_lrc.simulation` (remove all `sys.path.append('./src')` hacks and wildcard `*` imports).
2. Guard execution behind `def main(argv: Sequence[str] | None = None) -> None:` and `if __name__ == "__main__": main()`, so importing any script in tests has **zero side effects**!
3. Provide `argparse.ArgumentParser` flags with defaults matching the original script parameters (`--q`, `--n`, `--k`, `--r`, `--local-min-dist`, `--subgroup-types`, `--decoder`, `--sim-itr`, `--max-err`, `--max-itr`, `--use-enc`, `--output-dir`, `--seed`, etc.) so both quick smoke runs (`--sim-itr 5 --max-err 3`) and full Monte Carlo runs work from the CLI without editing source code!
4. Delete `simulation/proces_results.py` after creating `simulation/process_results.py`.

---

## Acceptance Checklist
- [ ] `simulation/proces_results.py` is deleted and `simulation/process_results.py` exists.
- [ ] `ruff check src/tamo_barg_lrc/simulation/ simulation/` and `ruff format --check src/tamo_barg_lrc/simulation/ simulation/` pass with zero issues.
- [ ] Every script in `simulation/` supports `--help` and runs cleanly with `--sim-itr 3 --max-err 2 --output-dir <tmp>` across all 6 decoder names without `ZeroDivisionError` (**B16**) or `TypeError` (**B15**).
- [ ] Zero unclosed files or `matplotlib` figure leaks (**B18**).
