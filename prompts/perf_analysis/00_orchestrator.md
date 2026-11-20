# 00 — Orchestrator: Performance & SageMath-Alternatives Analysis

## Role
You coordinate the analysis swarm. You do not analyse code yourself. Read `shared_context.md` first.

## Waves
| Wave | Agents (parallel) | Prompt files | Gate |
|---|---|---|---|
| 1 | A1–A6 per-function analysis | `01_…` – `06_…` | each report exists, every inventory function has a complete row |
| 2 | B1 profiling, B2 Sage audit | `07_…`, `08_…` | B1 has scaling data; B2 lists every Sage symbol with file:line |
| 3 | B3–B6 alternatives (need A+B1+B2 reports as input) | `09_…` – `12_…` | each has kernel micro-benchmarks + score matrix |
| 4 | C1 synthesizer | `13_synthesizer.md` | `FINAL_REPORT.md` complete |

## Procedure
1. `mkdir -p reports/perf_analysis/scratch`; confirm `git status` is clean and `pytest -q` → 34 passed.
2. Launch each wave with `invoke_subagent` (type `self`, default workspace). Prompt = "Read `prompts/perf_analysis/shared_context.md` and `prompts/perf_analysis/<file>` and execute it."
3. At each gate verify:
   - Wave 1: row-completeness check — for each function in the inventory, `grep -c` its name in the owner's report ≥ 1; no empty fields.
   - Output paths are disjoint; `git status` shows changes only under `reports/perf_analysis/`; `git diff --stat src simulation tests` is empty.
4. If a gate fails, message the owning agent with the exact missing items; max 2 retries, then record the gap in the final report.
5. After Wave 4, re-run `pytest -q` and `ruff check src simulation tests` (must be unchanged), then `kill_all` subagents.
6. Report to the user: link `FINAL_REPORT.md`, the recommendation, top 5 optimizations. Do NOT commit/push unless asked.

## Acceptance
- [ ] 6 per-function reports, 2 cross-cutting reports, 4 alternative reports, 1 final report
- [ ] No source file modified
- [ ] All agents terminated
