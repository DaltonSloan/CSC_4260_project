# CSC 4260 Project

Data engineering and analysis workspace for loading HVAC and whole-building energy CSVs into MySQL, then exploring results in notebooks.

## What This Repo Includes

- `loader.py`: ingestion script that creates schema and upserts CSV data into MySQL.
- `scripts/test_db_connections.py`: validates DB connectivity for one or more user env files.
- `data/`: source CSV files.
- `.devcontainer/`: reproducible VS Code Dev Container setup.
- `io/input` and `io/output`: host-container shared folders for file exchange.

## Prerequisites

### Recommended (Dev Container)

- Docker Desktop running
- VS Code
- VS Code extension: Dev Containers (`ms-vscode-remote.remote-containers`)

### Local (No Container)

- Python 3.12+
- `pip`

## 1) Setup With Dev Container (Recommended)

1. Open this folder in VS Code.
2. Run `Dev Containers: Reopen in Container`.
3. Wait for first-time build and post-create setup to finish.
   - `.devcontainer/post-create.sh` installs `requirements.txt`, configures `uv`, and shell tooling.
4. Open a terminal in the container and verify:

```bash
python3 --version
pip --version
```

Notes:
- Port `8888` is forwarded for Jupyter.
- `io/input` and `io/output` are mounted into container paths `/io/input` and `/io/output`.

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

## Security Notes

- Never commit real credentials.
- `.env`, `env/`, and virtual environments are already ignored by `.gitignore`.
