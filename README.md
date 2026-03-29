# CSC 4260 Project

Data engineering and analysis workspace for loading HVAC and whole-building energy CSVs into MySQL, then exploring results in notebooks.

## Project Status (As of 2026-03-29)

- Project phase: Mid-project (data ingestion and analysis ramp-up).
- Environment evaluation: Dev container workflow validated and hardened for repeatable rebuilds.
- Current focus: Loading newly collected CSV data and progressing analysis in notebooks.

## What This Repo Includes

- `loader.py`: ingestion script that creates schema and upserts CSV data into MySQL.
- `scripts/test_db_connections.py`: validates DB connectivity for one or more user env files.
- `data/`: source CSV files.
- `.devcontainer/`: reproducible VS Code Dev Container setup.
- `io/input` and `io/output`: repo-scoped file exchange folders.

## Prerequisites

### Recommended (Dev Container)

- Docker Engine running (Docker Desktop or Colima)
- VS Code
- VS Code extension: Dev Containers (`ms-vscode-remote.remote-containers`)

### Local (No Container)

- Python 3.12+
- `pip`

## 1) Setup With Dev Container (Recommended)

1. Open this folder in VS Code.
2. Run `Dev Containers: Reopen in Container`.
3. Wait for first-time build and post-create setup to finish.
   - `.devcontainer/post-create.sh` installs Python dependencies from `requirements.txt`.
   - JupyterLab is installed as part of the project dependencies, so `python -m jupyter lab` works without extra setup.
   - The container is pinned to Python `3.12-bookworm` in `.devcontainer/devcontainer.json` build args for reproducibility.
4. Open a terminal in the container and verify:

```bash
python3 --version
pip --version
```

Notes:
- Port `8888` is forwarded for Jupyter.
- `io/input` and `io/output` live in the repo and are always available after clone.
- `/io/input` and `/io/output` are recreated inside the container as convenience symlinks to those repo folders.
- `updateContentCommand` re-runs `.devcontainer/post-create.sh` when repository contents change after rebuild/reopen.

## 2) Setup Local Python Environment (Alternative)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3) Configure Database Credentials

Database credentials are not stored in this repo. Request credentials from the project maintainer.

Create a runtime env file for `loader.py`:

```bash
cp .env.example .env
```

Fill in the required values:

- `MYSQL_HOST`
- `MYSQL_PORT` (default `3306`)
- `MYSQL_USER`
- `MYSQL_PASSWORD`
- `MYSQL_DATABASE`
- Optional SSL fields: `MYSQL_SSL_ENABLED`, `MYSQL_SSL_VERIFY_CERT`, `MYSQL_SSL_VERIFY_IDENTITY`, `MYSQL_SSL_CA`

For team testing with per-user env files:

```bash
cp .env.example env/.env.<yourname>
```

The `env/` folder is tracked with a `.gitkeep`, so the copy command works on a fresh clone.

## 4) Test DB Connectivity

Test all `env/.env.*` files:

```bash
python scripts/test_db_connections.py
```

Test only your env file:

```bash
python scripts/test_db_connections.py --users <yourname>
```

## 5) Run Data Load

Run ingestion (uses `.env` in repo root):

```bash
python loader.py
```

Defaults:
- `CSV_GLOB` defaults to `data/*.csv` (repo-relative).
- `CHUNK_SIZE` defaults to `5000`.

You can override either via environment variables in `.env`.

## 6) Jupyter (Optional)

```bash
python -m jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

In Dev Container, VS Code auto-forwards port `8888`.

## Troubleshooting

- `Missing required env vars`: verify `.env` exists and has all `MYSQL_*` values.
- `No CSV files matched`: verify `CSV_GLOB` in `.env` and file locations.
- Connection timeout/auth errors: run `scripts/test_db_connections.py` and review hints.
- If you pulled `.devcontainer/` changes into an existing local clone, run `Dev Containers: Rebuild Container` once so the updated container config takes effect.
- If devcontainer dependencies drift, rebuild container and rerun:
  - `bash .devcontainer/post-create.sh`
  - `python3 -m pip check`

## Devcontainer Reliability Checklist

- Base image version is pinned via `.devcontainer/Dockerfile` + build arg (`3.12-bookworm`).
- Post-create is idempotent and retry-safe for pip installs.
- Post-create validates dependency consistency with `pip check`.
- Repo-scoped `env/`, `io/input`, and `io/output` folders are created or preserved by post-create.
- `/io/input` and `/io/output` are restored as container-local symlinks instead of extra host bind mounts.

## Security Notes

- Never commit real credentials.
- `.env`, `env/`, and virtual environments are already ignored by `.gitignore`.
