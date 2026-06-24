# Release Checklist

Status reflects this repository checkout and the validation recorded in `docs/codex_milestone_result.md`.

- [x] Aggregate tests and phase checks completed in the current release run.
- [x] Documentation and Markdown links checked in the current release run.
- [x] Repository cleanliness checked in the current release run.
- [x] Raw WISDM data and processed windows are excluded from version control.
- [x] Generated packages, checkpoints, logs, caches, and temporary workspaces are excluded.
- [x] Sparrow-V external checkout cleanliness checked when `SPARROWV_ROOT` is available.
- [x] `make help` and `python3 -m sparrowml.cli --help` verified.
- [x] README, architecture, reproduction commands, and documentation links verified.
- [x] Claims distinguish WISDM from synthetic results, software reference from RTL, and measured counters from derived counts.
- [x] No physical hardware, optimized monolithic latency, general-compiler, or sparse-speedup claim is made.
- [x] [LICENSE](../LICENSE) is present and linked.
- [x] Canonical metrics are consolidated in [final results](results/final_results.md).
- [x] Reproduction guide and portfolio summary are present.
- [x] No commit, push, tag, or release was created by this milestone.

Items are marked complete only after the final commands run; the finalized result file is the release evidence.
