#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"
required=(AGENTS.md docs/codex_context.md docs/current_milestone.md docs/codex_milestone_prompt.md)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "error: missing workflow file: $file" >&2; exit 2; }
done
[[ -f pyproject.toml && -d src/sparrowml ]] || { echo "error: not a SparrowML repository" >&2; exit 2; }
milestone="$(sed -n 's/^# \{1,\}//p' docs/current_milestone.md | head -n 1)"
[[ -n "$milestone" ]] || milestone="Unnamed Milestone"
result_file="docs/codex_milestone_result.md"
printf 'STATUS: IN_PROGRESS\nMILESTONE: %s\nSTARTED_AT: %s\n\nLauncher initialized this result file.\n' "$milestone" "$(date -Iseconds)" > "$result_file"
if [[ -n "$(git status --porcelain 2>/dev/null || true)" ]]; then
  echo "warning: working tree is dirty; preserving existing changes" >&2
fi
prompt="$(<docs/codex_milestone_prompt.md)"
prompt+=$'\n\n--- CURRENT MILESTONE ---\n'
prompt+="$(<docs/current_milestone.md)"
codex_command="${CODEX_CMD:-codex exec --sandbox workspace-write}"
if ! command -v "${codex_command%% *}" >/dev/null 2>&1; then
  echo "error: Codex command not found: ${codex_command%% *}. Set CODEX_CMD to override." >&2
  exit 127
fi
trap 'echo "warning: interrupted; result file preserved at docs/codex_milestone_result.md" >&2' INT TERM
set +e
if [[ "$codex_command" == *" "* ]]; then
  # CODEX_CMD is a documented local override, for example: "codex exec".
  eval "$codex_command" '"$prompt"'
else
  "$codex_command" "$prompt"
fi
codex_status=$?
set -e
trap - INT TERM
[[ -f "$result_file" ]] || { echo "error: Codex removed the result file" >&2; exit 1; }
status="$(sed -n 's/^STATUS: \(IN_PROGRESS\|COMPLETE\|FAILED\|BLOCKED\)$/\1/p' "$result_file" | head -n 1)"
if [[ "$status" == "IN_PROGRESS" || -z "$status" ]]; then
  echo "warning: result status remains IN_PROGRESS or is invalid" >&2
else
  cat "$result_file"
fi
exit "$codex_status"
