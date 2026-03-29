#!/usr/bin/env bash
set -euo pipefail

workspace_dir="${containerWorkspaceFolder:-$(pwd)}"
cd "$workspace_dir"

# Ensure repo-scoped folders exist in both fresh clones and long-lived branches.
mkdir -p env io/input io/output

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

if [ ! -f requirements.txt ]; then
  echo "requirements.txt not found in $workspace_dir" >&2
  exit 1
fi

retry 3 3 python3 -m pip install -r requirements.txt
python3 -m pip check

ensure_symlink() {
  local target="$1"
  local link_path="$2"

  if [ -L "$link_path" ] && [ "$(readlink "$link_path")" = "$target" ]; then
    return 0
  fi

  if [ -e "$link_path" ] && [ ! -L "$link_path" ]; then
    echo "Skipping shortcut creation because $link_path already exists and is not a symlink." >&2
    return 0
  fi

  rm -f "$link_path"
  ln -s "$target" "$link_path"
}

ensure_io_shortcuts() {
  local io_root="/io"

  if [ ! -f /.dockerenv ]; then
    return 0
  fi

  if [ -w / ]; then
    mkdir -p "$io_root"
    ensure_symlink "$workspace_dir/io/input" "$io_root/input"
    ensure_symlink "$workspace_dir/io/output" "$io_root/output"
    return 0
  fi

  if command -v sudo >/dev/null 2>&1; then
    sudo mkdir -p "$io_root"
    if [ -L "$io_root/input" ] || [ ! -e "$io_root/input" ]; then
      sudo rm -f "$io_root/input"
      sudo ln -s "$workspace_dir/io/input" "$io_root/input"
    else
      echo "Skipping shortcut creation because $io_root/input already exists and is not a symlink." >&2
    fi

    if [ -L "$io_root/output" ] || [ ! -e "$io_root/output" ]; then
      sudo rm -f "$io_root/output"
      sudo ln -s "$workspace_dir/io/output" "$io_root/output"
    else
      echo "Skipping shortcut creation because $io_root/output already exists and is not a symlink." >&2
    fi
    return 0
  fi

  echo "Skipping /io shortcut creation because neither write access nor sudo is available." >&2
}

ensure_io_shortcuts
