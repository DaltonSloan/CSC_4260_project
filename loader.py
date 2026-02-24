import os
import glob
import math
from typing import Dict, List, Tuple, Optional

import pandas as pd
import pymysql

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    # dotenv is optional; env vars can be set another way
    pass


# ----------------------------
# CONFIG (via env, with fallbacks)
# ----------------------------
def _get_env(*keys: str, default: Optional[str] = None) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value is not None and str(value).strip() != "":
            return value
    return default


def _get_env_int(*keys: str, default: Optional[int] = None) -> Optional[int]:
    value = _get_env(*keys)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as e:
        raise RuntimeError(f"Invalid integer in env var(s) {', '.join(keys)}: {value}") from e


def _get_env_bool(*keys: str, default: bool = False) -> bool:
    value = _get_env(*keys)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


MYSQL_HOST = _get_env("MYSQL_HOST", "DB_HOST", "HOST")
MYSQL_PORT = _get_env_int("MYSQL_PORT", "DB_PORT", "PORT", default=3306)
MYSQL_USER = _get_env("MYSQL_USER", "DB_USER")
MYSQL_PASSWORD = _get_env("MYSQL_PASSWORD", "DB_PASSWORD", "PASS")
MYSQL_DATABASE = _get_env("MYSQL_DATABASE", "DB_DATABASE", "DB")
MYSQL_SSL_ENABLED = _get_env_bool("MYSQL_SSL_ENABLED", "DB_SSL_ENABLED", default=True)
MYSQL_SSL_VERIFY_CERT = _get_env_bool("MYSQL_SSL_VERIFY_CERT", default=False)
MYSQL_SSL_VERIFY_IDENTITY = _get_env_bool("MYSQL_SSL_VERIFY_IDENTITY", default=False)
MYSQL_SSL_CA = _get_env("MYSQL_SSL_CA")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_GLOB = os.path.join(PROJECT_ROOT, "data", "*.csv")
CSV_GLOB = os.getenv("CSV_GLOB", DEFAULT_CSV_GLOB)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "5000"))


# ----------------------------
# DDL
# ----------------------------
DDL = [
    """
    CREATE TABLE IF NOT EXISTS devices (
        device_id VARCHAR(50) PRIMARY KEY,
        device_name VARCHAR(100) NULL
    ) ENGINE=InnoDB;
    """,
    """
    CREATE TABLE IF NOT EXISTS sensors (
        sensor_id INT AUTO_INCREMENT PRIMARY KEY,
        sensor_name VARCHAR(255) NOT NULL UNIQUE
    ) ENGINE=InnoDB;
    """,
    """
    CREATE TABLE IF NOT EXISTS measurements (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        device_id VARCHAR(50) NOT NULL,
        sensor_id INT NOT NULL,
        timestamp DATETIME NOT NULL,

        value_double DOUBLE NULL,
        value_text VARCHAR(255) NULL,
        status VARCHAR(50) NULL,

        FOREIGN KEY (device_id) REFERENCES devices(device_id),
        FOREIGN KEY (sensor_id) REFERENCES sensors(sensor_id),

        UNIQUE KEY uq_measurement (device_id, sensor_id, timestamp),
        INDEX idx_device_time (device_id, timestamp),
        INDEX idx_sensor_time (sensor_id, timestamp)
    ) ENGINE=InnoDB;
    """
]


# ----------------------------
# DB helpers
# ----------------------------
def get_mysql_connection(database: Optional[str] = None):
    missing = [
        k for k, v in [
            ("MYSQL_HOST/DB_HOST/HOST", MYSQL_HOST),
            ("MYSQL_USER/DB_USER", MYSQL_USER),
            ("MYSQL_PASSWORD/DB_PASSWORD/PASS", MYSQL_PASSWORD),
            ("MYSQL_DATABASE/DB_DATABASE/DB", MYSQL_DATABASE),
        ] if not v
    ]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    try:
        connect_kwargs = {
            "host": MYSQL_HOST,
            "port": MYSQL_PORT,
            "user": MYSQL_USER,
            "password": MYSQL_PASSWORD,
            "database": database or MYSQL_DATABASE,
            "autocommit": False,
        }
        if MYSQL_SSL_ENABLED:
            ssl_kwargs = {
                "verify_mode": MYSQL_SSL_VERIFY_CERT,
                "check_hostname": MYSQL_SSL_VERIFY_IDENTITY,
            }
            if MYSQL_SSL_CA:
                ssl_kwargs["ca"] = MYSQL_SSL_CA
            connect_kwargs["ssl"] = ssl_kwargs

        return pymysql.connect(
            **connect_kwargs
        )
    except Exception as e:
        raise RuntimeError(f"MySQL connection failed: {e}") from e


def run_query(query: str, params=None, fetch: bool = False):
    connection = None
    cursor = None
    try:
        connection = get_mysql_connection()
        cursor = connection.cursor()
        cursor.execute(query, params or ())
        if fetch:
            return cursor.fetchall()
        connection.commit()
        return None
    except Exception as e:
        if connection:
            connection.rollback()
        raise RuntimeError(f"MySQL query failed: {e}") from e
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def ensure_schema():
    connection = None
    cursor = None
    try:
        connection = get_mysql_connection()
        cursor = connection.cursor()
        for stmt in DDL:
            cursor.execute(stmt)
        connection.commit()
    except Exception as e:
        if connection:
            connection.rollback()
        raise RuntimeError(f"Schema creation failed: {e}") from e
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def insert_devices(device_pairs: List[Tuple[str, str]]):
    if not device_pairs:
        return
    connection = None
    cursor = None
    try:
        connection = get_mysql_connection()
        cursor = connection.cursor()
        cursor.executemany(
            """
            INSERT INTO devices (device_id, device_name)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE device_name = VALUES(device_name)
            """,
            device_pairs
        )
        connection.commit()
    except Exception as e:
        if connection:
            connection.rollback()
        raise RuntimeError(f"Insert devices failed: {e}") from e
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def insert_sensors(sensor_names: List[str]):
    if not sensor_names:
        return
    connection = None
    cursor = None
    try:
        connection = get_mysql_connection()
        cursor = connection.cursor()
        cursor.executemany(
            "INSERT IGNORE INTO sensors (sensor_name) VALUES (%s)",
            [(s,) for s in sensor_names]
        )
        connection.commit()
    except Exception as e:
        if connection:
            connection.rollback()
        raise RuntimeError(f"Insert sensors failed: {e}") from e
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def load_sensor_id_map() -> Dict[str, int]:
    rows = run_query("SELECT sensor_id, sensor_name FROM sensors", fetch=True) or []
    return {name: int(sid) for (sid, name) in rows}


def bulk_upsert_measurements(rows: List[Tuple[str, int, str, Optional[float], Optional[str], Optional[str]]]):
    if not rows:
        return
    connection = None
    cursor = None
    try:
        connection = get_mysql_connection()
        cursor = connection.cursor()
        cursor.executemany(
            """
            INSERT INTO measurements
                (device_id, sensor_id, timestamp, value_double, value_text, status)
            VALUES
                (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                value_double = VALUES(value_double),
                value_text   = VALUES(value_text),
                status       = VALUES(status)
            """,
            rows
        )
        connection.commit()
    except Exception as e:
        if connection:
            connection.rollback()
        raise RuntimeError(f"Upsert measurements failed: {e}") from e
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


# ----------------------------
# CSV helpers
# ----------------------------
def find_sensor_bases(columns: List[str]) -> List[str]:
    bases = set()
    for c in columns:
        if c.endswith("_status"):
            bases.add(c[:-7])
        elif c.endswith("_value"):
            bases.add(c[:-6])
    return sorted(bases)


def try_float(x) -> Tuple[Optional[float], Optional[str]]:
    if x is None:
        return (None, None)
    if isinstance(x, float) and math.isnan(x):
        return (None, None)
    if isinstance(x, (int, float)):
        return (float(x), None)

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return (None, None)

    try:
        return (float(s), None)
    except ValueError:
        return (None, s[:255])


def process_csv(csv_path: str):
    print(f"\n=== Processing: {os.path.basename(csv_path)} ===")

    # Read header
    header_df = pd.read_csv(csv_path, nrows=0)
    cols = list(header_df.columns)

    required = {"device_id", "device_name", "time"}
    missing = required - set(cols)
    if missing:
        raise RuntimeError(f"{csv_path} missing required columns: {missing}")

    sensor_bases = find_sensor_bases(cols)

    # Insert devices (unique devices seen in file preview)
    preview = pd.read_csv(csv_path, usecols=["device_id", "device_name"], nrows=500)
    device_pairs = list({(str(r["device_id"]), str(r["device_name"])) for _, r in preview.iterrows()})
    insert_devices(device_pairs)

    # Insert sensors and refresh map
    insert_sensors(sensor_bases)
    sensor_id_map = load_sensor_id_map()

    # Determine columns to read in chunks
    usecols = ["device_id", "time"]
    for b in sensor_bases:
        vcol = f"{b}_value"
        scol = f"{b}_status"
        if vcol in cols:
            usecols.append(vcol)
        if scol in cols:
            usecols.append(scol)

    # Chunked read
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=CHUNK_SIZE):
        # Parse timestamps (matches your exports: 2026/01/21 00:00:14)
        chunk["timestamp"] = pd.to_datetime(chunk["time"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
        chunk = chunk.dropna(subset=["timestamp"])
        if chunk.empty:
            continue

        device_ids = chunk["device_id"].astype(str).tolist()
        timestamps = chunk["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

        out_rows: List[Tuple[str, int, str, Optional[float], Optional[str], Optional[str]]] = []

        for base in sensor_bases:
            sid = sensor_id_map.get(base)
            if sid is None:
                continue

            vcol = f"{base}_value"
            scol = f"{base}_status"

            values = chunk[vcol].tolist() if vcol in chunk.columns else [None] * len(chunk)
            statuses = chunk[scol].tolist() if scol in chunk.columns else [None] * len(chunk)

            for dev, ts, v, st in zip(device_ids, timestamps, values, statuses):
                vd, vt = try_float(v)

                # normalize status
                status_str = None
                if st is not None and not (isinstance(st, float) and math.isnan(st)):
                    status_str = str(st)[:50]

                # Skip totally empty readings
                if vd is None and vt is None and status_str is None:
                    continue

                out_rows.append((dev, sid, ts, vd, vt, status_str))

        bulk_upsert_measurements(out_rows)
        print(f"  inserted/updated: {len(out_rows):,} measurements")


def main():
    csv_files = sorted(glob.glob(CSV_GLOB))
    if not csv_files:
        raise RuntimeError(f"No CSV files matched: {CSV_GLOB}")

    print(f"Using DB: {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
    print(f"Found {len(csv_files)} CSV files via: {CSV_GLOB}")

    ensure_schema()

    for f in csv_files:
        process_csv(f)

    print("\n✅ Done! Tables populated: devices, sensors, measurements")


if __name__ == "__main__":
    main()
