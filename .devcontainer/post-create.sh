#!/usr/bin/env bash
set -euo pipefail

workspace_dir="${containerWorkspaceFolder:-$(pwd)}"
safe_dirs="$(git config --global --get-all safe.directory || true)"
if ! printf '%s\n' "$safe_dirs" | grep -Fxq "$workspace_dir"; then
  git config --global --add safe.directory "$workspace_dir"
fi

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

touch "$HOME/.bashrc"
if ! grep -Fq 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc"; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

if [ ! -d "$HOME/.oh-my-bash" ]; then
  git clone --depth=1 https://github.com/ohmybash/oh-my-bash.git "$HOME/.oh-my-bash"
fi

if ! grep -Fq 'source "$OSH/oh-my-bash.sh"' "$HOME/.bashrc"; then
  cat <<'EOF' >> "$HOME/.bashrc"

export OSH="$HOME/.oh-my-bash"
OSH_THEME="font"
plugins=(git)
source "$OSH/oh-my-bash.sh"
EOF
fi
