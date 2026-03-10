#!/usr/bin/env bash
set -euo pipefail

workspace_dir="${containerWorkspaceFolder:-$(pwd)}"
safe_dirs="$(git config --global --get-all safe.directory || true)"
if ! printf '%s\n' "$safe_dirs" | grep -Fxq "$workspace_dir"; then
  git config --global --add safe.directory "$workspace_dir"
fi

retry() {
  local max_attempts="$1"
  local sleep_seconds="$2"
  shift 2
  local attempt=1

  until "$@"; do
    if [ "$attempt" -ge "$max_attempts" ]; then
      return 1
    fi
    echo "Command failed (attempt ${attempt}/${max_attempts}): $*" >&2
    attempt=$((attempt + 1))
    sleep "$sleep_seconds"
  done
}

retry 3 3 python3 -m pip install --upgrade pip
retry 3 3 python3 -m pip install -r requirements.txt
