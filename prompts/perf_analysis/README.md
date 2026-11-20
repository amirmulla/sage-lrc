# Performance & SageMath-Alternatives Analysis Swarm

Analysis-only swarm: per-function algorithm review, deep performance analysis, and evaluation of replacements for the SageMath dependency. **No source changes.**

## Topology
```mermaid
flowchart TD
    O[00 Orchestrator] --> W1
    subgraph W1[Wave 1 - per-function]
        A1[01 subgroups+polynomials]
        A2[02 expander]
        A3[03 tamo_barg]
        A4[04 two-set code/encoder]
        A5[05 two-set decoders]
        A6[06 simulation]
    end
    W1 --> W2
    subgraph W2[Wave 2 - cross-cutting]
        B1[07 profiling+scaling]
        B2[08 Sage audit + ff_backend spec]
    end
    W2 --> W3
    subgraph W3[Wave 3 - alternatives]
        B3[09 galois]
        B4[10 python-flint]
        B5[11 custom tables/numba/cython]
        B6[12 slim passagemath + others]
    end
    W3 --> C1[13 Synthesizer -> FINAL_REPORT.md]
```

## Files
| File | Purpose |
|---|---|
| [shared_context.md](shared_context.md) | inventory, baselines, methodology, per-function row schema |
| [00_orchestrator.md](00_orchestrator.md) | waves, gates |
| 01–06 | per-function analysts (every function owned by exactly one) |
| 07–08 | profiling; Sage surface audit |
| [alternatives_template.md](alternatives_template.md) + 09–12 | alternative evaluations |
| [13_synthesizer.md](13_synthesizer.md) | final report |

## Launch
Start one agent with `00_orchestrator.md` ("Read prompts/perf_analysis/shared_context.md and prompts/perf_analysis/00_orchestrator.md and execute it"). Outputs land in `reports/perf_analysis/`.
