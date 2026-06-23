# SparrowML Agent Instructions

## Project rules

SparrowML is an ML systems, compiler, runtime, and hardware-aware optimization project. Sparrow-V is an external target, not copied source. Do not modify Sparrow-V unless a milestone explicitly requests a coordinated change. Do not invent measured results or claim general model accuracy from fixtures or small subsets. Distinguish measured, simulated, estimated, and derived values. Preserve determinism. Prefer simple, inspectable code over large frameworks. Do not introduce a database, web frontend, distributed system, or cloud dependency without a milestone requiring it. Do not commit or push.

## Token-efficient workflow

Read `docs/codex_context.md` first and `docs/current_milestone.md` second. Inspect only files directly relevant to the milestone; avoid repository-wide exploration unless a concrete failure requires it. Use focused searches, do not repeatedly reread large files, and run focused tests while developing. Run aggregate regressions once at final acceptance. Avoid verbose progress narration and keep final output compact. Stop only for genuine architectural blockers. Preserve partial work and record failures in the result file.

## Milestone discipline

Implement only the current milestone. Do not add speculative features or expand scope because something seems useful. Update documentation only where materially affected. Maintain backward compatibility unless the milestone says otherwise. Update `docs/codex_milestone_result.md` during the run and never exit normally with its result status still `IN_PROGRESS`.

## Required statuses

Use exactly `STATUS: IN_PROGRESS`, `STATUS: COMPLETE`, `STATUS: FAILED`, or `STATUS: BLOCKED`.

- `COMPLETE`: every acceptance criterion and required validation passes.
- `FAILED`: required work or tests remain incomplete.
- `BLOCKED`: a genuine human decision or architectural stop condition exists.
