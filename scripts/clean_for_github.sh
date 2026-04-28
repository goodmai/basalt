#!/usr/bin/env bash
# Clean temporary/local artifacts before publishing repository to GitHub.
# Default mode is dry-run (no deletions).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="dry-run"
INCLUDE_LOCAL=false

usage() {
  cat <<'EOF'
Usage:
  ./scripts/clean_for_github.sh [--dry-run] [--apply] [--include-local]

Options:
  --dry-run        Show what would be removed (default)
  --apply          Actually remove matched artifacts
  --include-local  Also remove local working artifacts (.archive, .claude, .gemini.md, test_*.swift in repo root)
  -h, --help       Show this help
EOF
}

for arg in "$@"; do
  case "$arg" in
    --dry-run) MODE="dry-run" ;;
    --apply) MODE="apply" ;;
    --include-local) INCLUDE_LOCAL=true ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

declare -a always_dirs=(
  ".build"
  ".swiftpm"
  "logs"
  "docs-output"
  "docs-static"
  ".docc"
)

declare -a always_files=(
  ".coverage"
  "default.metallib"
  "context_latency.json"
  "auth.sqlite3"
)

declare -a recursive_file_patterns=(
  ".DS_Store"
  "*.log"
  "*.tmp"
  "*.swp"
  "*~"
  "*.profdata"
  "server_*.log"
  "benchmark_*.json"
  "res_*.json"
)

declare -a root_file_patterns=(
  "*.db"
  "*.sqlite"
  "*.sqlite3"
)

declare -a optional_local_paths=(
  ".archive"
  ".claude"
  ".gemini.md"
)

declare -a collected=()

add_if_exists() {
  local p="$1"
  [[ -e "$p" ]] || return 0

  if [[ "$INCLUDE_LOCAL" == false ]]; then
    case "$p" in
      "$REPO_ROOT/.archive"|"$REPO_ROOT/.archive/"*|"$REPO_ROOT/.claude"|"$REPO_ROOT/.claude/"*|"$REPO_ROOT/.gemini.md")
        return 0
        ;;
    esac
  fi
  collected+=("$p")
}

for d in "${always_dirs[@]}"; do
  add_if_exists "${REPO_ROOT}/${d}"
done

for f in "${always_files[@]}"; do
  add_if_exists "${REPO_ROOT}/${f}"
done

for pattern in "${recursive_file_patterns[@]}"; do
  while IFS= read -r match; do
    add_if_exists "$match"
  done < <(find "${REPO_ROOT}" -path "${REPO_ROOT}/.git" -prune -o -type f -name "$pattern" -print)
done

shopt -s nullglob dotglob
for pattern in "${root_file_patterns[@]}"; do
  for match in "${REPO_ROOT}"/$pattern; do
    add_if_exists "$match"
  done
done

if [[ "$INCLUDE_LOCAL" == true ]]; then
  for p in "${optional_local_paths[@]}"; do
    add_if_exists "${REPO_ROOT}/${p}"
  done
  for match in "${REPO_ROOT}"/test_*.swift; do
    add_if_exists "$match"
  done
fi
shopt -u nullglob dotglob

declare -a unique_collected=()
for p in "${collected[@]}"; do
  already_added=false
  for up in "${unique_collected[@]-}"; do
    if [[ "$up" == "$p" ]]; then
      already_added=true
      break
    fi
  done
  if [[ "$already_added" == false ]]; then
    unique_collected+=("$p")
  fi
done

is_tracked_file() {
  local abs="$1"
  local rel="${abs#$REPO_ROOT/}"
  git -C "$REPO_ROOT" ls-files --error-unmatch -- "$rel" >/dev/null 2>&1
}

has_tracked_in_dir() {
  local abs="$1"
  local rel="${abs#$REPO_ROOT/}"
  local tracked
  tracked="$(git -C "$REPO_ROOT" ls-files -- "$rel" 2>/dev/null || true)"
  [[ -n "$tracked" ]]
}

declare -a to_remove=()
declare -a skipped_tracked=()

for p in "${unique_collected[@]}"; do
  if [[ -d "$p" ]]; then
    if has_tracked_in_dir "$p"; then
      skipped_tracked+=("$p")
    else
      to_remove+=("$p")
    fi
  else
    if is_tracked_file "$p"; then
      skipped_tracked+=("$p")
    else
      to_remove+=("$p")
    fi
  fi
done

echo "Repository: $REPO_ROOT"
echo "Mode: $MODE"
echo "Include local artifacts: $INCLUDE_LOCAL"
echo

if [[ ${#to_remove[@]} -eq 0 ]]; then
  echo "No removable temporary artifacts found."
else
  echo "Artifacts selected for cleanup (${#to_remove[@]}):"
  for p in "${to_remove[@]}"; do
    echo "  - ${p#$REPO_ROOT/}"
  done
fi

if [[ ${#skipped_tracked[@]} -gt 0 ]]; then
  echo
  echo "Skipped because tracked by git (${#skipped_tracked[@]}):"
  for p in "${skipped_tracked[@]}"; do
    echo "  - ${p#$REPO_ROOT/}"
  done
fi

if [[ "$MODE" == "dry-run" ]]; then
  echo
  echo "Dry-run complete. Re-run with --apply to delete listed artifacts."
  exit 0
fi

if [[ ${#to_remove[@]} -eq 0 ]]; then
  echo
  echo "Nothing to delete."
  exit 0
fi

for p in "${to_remove[@]}"; do
  if [[ -d "$p" ]]; then
    rm -rf "$p"
  else
    rm -f "$p"
  fi
done

echo
echo "Cleanup applied."
