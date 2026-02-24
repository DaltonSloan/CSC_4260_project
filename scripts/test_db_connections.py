#!/usr/bin/env python3
"""Test MySQL connectivity for user-specific .env files."""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from dotenv import dotenv_values
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: python-dotenv. "
        "Install with '/usr/local/bin/python3 -m pip install --user python-dotenv' "
        "or run with your project venv: './.venv/bin/python scripts/test_db_connections.py'."
    ) from exc

try:
    import pymysql
except ModuleNotFoundError:
    pymysql = None


def parse_env_file(path: Path) -> Dict[str, str]:
    raw = dotenv_values(path)
    env: Dict[str, str] = {}
    for key, value in raw.items():
        if key and value is not None:
            env[key] = value
    return env


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def find_env_files(env_dir: Path) -> List[Path]:
    pattern = str(env_dir / ".env.*")
    return sorted(Path(p) for p in glob.glob(pattern) if Path(p).is_file())


def test_connection_with_mysql_cli(config: Dict[str, str], timeout_seconds: int) -> Tuple[str, str]:
    host = config["MYSQL_HOST"]
    port = int(config.get("MYSQL_PORT", "3306"))
    user = config["MYSQL_USER"]
    password = config["MYSQL_PASSWORD"]
    database = config["MYSQL_DATABASE"]

    ssl_enabled = parse_bool(config.get("MYSQL_SSL_ENABLED"), True)
    ssl_verify_cert = parse_bool(config.get("MYSQL_SSL_VERIFY_CERT"), False)
    ssl_verify_identity = parse_bool(config.get("MYSQL_SSL_VERIFY_IDENTITY"), False)
    ssl_ca = config.get("MYSQL_SSL_CA")

    cmd = [
        "mysql",
        "--connect-timeout",
        str(timeout_seconds),
        "--silent",
        "--skip-column-names",
        "--get-server-public-key",
        "-h",
        host,
        "-P",
        str(port),
        "-u",
        user,
        "-D",
        database,
        "-e",
        "SELECT CURRENT_USER(), DATABASE();",
    ]

    if ssl_enabled:
        if ssl_verify_identity:
            cmd.extend(["--ssl-mode", "VERIFY_IDENTITY"])
        elif ssl_verify_cert:
            cmd.extend(["--ssl-mode", "VERIFY_CA"])
        else:
            cmd.extend(["--ssl-mode", "REQUIRED"])
        if ssl_ca:
            cmd.extend(["--ssl-ca", ssl_ca])
    else:
        cmd.extend(["--ssl-mode", "DISABLED"])

    env = os.environ.copy()
    env["MYSQL_PWD"] = password

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"mysql exited with code {result.returncode}"
        raise RuntimeError(message)

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("mysql returned no output")

    parts = lines[-1].split("\t")
    if len(parts) < 2:
        raise RuntimeError(f"unexpected mysql output: {lines[-1]}")

    return parts[0], parts[1]


def test_connection(config: Dict[str, str], timeout_seconds: int) -> Tuple[str, str]:
    if pymysql is None:
        return test_connection_with_mysql_cli(config, timeout_seconds)

    host = config["MYSQL_HOST"]
    port = int(config.get("MYSQL_PORT", "3306"))
    user = config["MYSQL_USER"]
    password = config["MYSQL_PASSWORD"]
    database = config["MYSQL_DATABASE"]

    ssl_enabled = parse_bool(config.get("MYSQL_SSL_ENABLED"), True)
    ssl_verify_cert = parse_bool(config.get("MYSQL_SSL_VERIFY_CERT"), False)
    ssl_verify_identity = parse_bool(config.get("MYSQL_SSL_VERIFY_IDENTITY"), False)
    ssl_ca = config.get("MYSQL_SSL_CA")

    connect_kwargs = {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
        "connect_timeout": timeout_seconds,
        "read_timeout": timeout_seconds,
        "write_timeout": timeout_seconds,
        "autocommit": True,
    }

    if ssl_enabled:
        ssl_kwargs = {
            "verify_mode": ssl_verify_cert,
            "check_hostname": ssl_verify_identity,
        }
        if ssl_ca:
            ssl_kwargs["ca"] = ssl_ca
        connect_kwargs["ssl"] = ssl_kwargs

    connection = pymysql.connect(**connect_kwargs)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT CURRENT_USER(), DATABASE()")
            current_user, current_db = cursor.fetchone()
            return str(current_user), str(current_db)
    finally:
        connection.close()


def build_failure_hint(config: Dict[str, str], exc: Exception) -> str:
    host = config.get("MYSQL_HOST", "<missing-host>")
    port = config.get("MYSQL_PORT", "3306")
    message = str(exc).lower()

    if "timed out" in message:
        return (
            f"hint=timeout_reachability_check host={host} port={port} "
            "verify DB is running, listening publicly/private-route is reachable, and firewall/security rules allow this client"
        )
    if "refused" in message:
        return (
            f"hint=connection_refused host={host} port={port} "
            "verify mysqld is running and bound to the expected interface/port"
        )
    if "name or service not known" in message or "temporary failure in name resolution" in message:
        return (
            f"hint=dns_resolution host={host} "
            "verify MYSQL_HOST value and DNS/network configuration"
        )
    if "access denied" in message:
        return "hint=auth_failed verify MYSQL_USER/MYSQL_PASSWORD and grants for the target host"
    return f"hint=unknown_connection_issue host={host} port={port} check network path and database logs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Test DB connections for all env/.env.<user> files.")
    parser.add_argument("--env-dir", default="env", help="Directory containing .env.<user> files.")
    parser.add_argument(
        "--users",
        nargs="*",
        default=[],
        help="Optional list of usernames to test (example: --users sam jun).",
    )
    parser.add_argument("--timeout", type=int, default=8, help="Connection/query timeout in seconds.")
    args = parser.parse_args()

    env_dir = Path(args.env_dir)
    if not env_dir.is_dir():
        print(f"[ERROR] env directory not found: {env_dir}", file=sys.stderr)
        return 2

    env_files = find_env_files(env_dir)
    if not env_files:
        print(f"[ERROR] no files matched: {env_dir}/.env.*", file=sys.stderr)
        return 2

    selected_users = {u.lower() for u in args.users}

    failures = 0
    tested = 0

    for env_file in env_files:
        user_from_filename = env_file.name.replace(".env.", "", 1)
        if selected_users and user_from_filename.lower() not in selected_users:
            continue

        tested += 1
        try:
            config = parse_env_file(env_file)
            required = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"]
            missing = [k for k in required if not config.get(k)]
            if missing:
                raise RuntimeError(f"missing keys: {', '.join(missing)}")

            current_user, current_db = test_connection(config, args.timeout)
            print(f"[PASS] {user_from_filename:<8} file={env_file} db={current_db} current_user={current_user}")
        except Exception as exc:
            failures += 1
            hint = build_failure_hint(config if "config" in locals() else {}, exc)
            print(f"[FAIL] {user_from_filename:<8} file={env_file} error={exc} {hint}")

    if tested == 0:
        print("[ERROR] no matching users/files to test", file=sys.stderr)
        return 2

    print(f"\nSummary: {tested - failures}/{tested} succeeded, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
