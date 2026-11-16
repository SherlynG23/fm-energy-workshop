"""
Load three_meters.csv into Neon via the HTTP SQL API.
Uses the Neon-Connection-String header; strips channel_binding=require
from the URL before using it.
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

raw_url = os.environ["DATABASE_URL"]

# Strip channel_binding=require
conn_str = re.sub(r"[&?]channel_binding=require", "", raw_url)

# postgresql://user:pass@<host>/db?...  →  https://<host>/sql
match = re.search(r"@([^/]+)/", conn_str)
if not match:
    sys.exit("Cannot parse host from DATABASE_URL")
host = match.group(1)
endpoint = f"https://{host}/sql"

HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": conn_str,
}

CSV_PATH = BASE_DIR / "Data" / "three_meters.csv"

# ── Helper ────────────────────────────────────────────────────────────────────

def sql(query: str, params: list | None = None):
    body: dict = {"query": query}
    if params is not None:
        body["params"] = params
    resp = requests.post(endpoint, headers=HEADERS, json=body, timeout=60)
    if not resp.ok:
        print(f"HTTP {resp.status_code}: {resp.text[:800]}")
        resp.raise_for_status()
    return resp.json()


# ── 1. Create table ───────────────────────────────────────────────────────────
print("Creating table meter_readings_source …")
sql("""
CREATE TABLE IF NOT EXISTS meter_readings_source (
    timestamp       TIMESTAMPTZ    NOT NULL,
    meter_id        TEXT           NOT NULL,
    building_id     TEXT           NOT NULL,
    electricity_kwh NUMERIC(10,4)  NOT NULL,
    CONSTRAINT meter_readings_source_pk PRIMARY KEY (meter_id, timestamp)
)
""")
print("  Table ready.")

# ── 2. Load CSV ───────────────────────────────────────────────────────────────
print(f"Reading {CSV_PATH} …")
rows = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
print(f"  {len(rows):,} rows read.")

# ── 3. Upload using multi-row VALUES inserts ──────────────────────────────────
# Build one INSERT with many rows per HTTP call.
# 100 rows × 4 params = 400 params per request — well within limits.
CHUNK = 100

def upload_chunk(chunk):
    """Insert a list of row-dicts in a single parameterised INSERT."""
    n = len(chunk)
    placeholders = ", ".join(
        f"(${i*4+1}, ${i*4+2}, ${i*4+3}, ${i*4+4})"
        for i in range(n)
    )
    query = f"""
INSERT INTO meter_readings_source (timestamp, meter_id, building_id, electricity_kwh)
VALUES {placeholders}
ON CONFLICT (meter_id, timestamp) DO NOTHING
"""
    params = []
    for r in chunk:
        params.extend([r["timestamp"], r["meter_id"], r["building_id"], r["electricity_kwh"]])
    sql(query, params)


total_chunks = (len(rows) + CHUNK - 1) // CHUNK
print(f"Uploading {len(rows):,} rows in {total_chunks} chunk(s) of {CHUNK} …")

for idx, start in enumerate(range(0, len(rows), CHUNK), 1):
    upload_chunk(rows[start : start + CHUNK])
    if idx % 10 == 0 or idx == total_chunks:
        print(f"  {idx}/{total_chunks} chunks done …")

print("Upload complete.")

# ── 4. Summary per meter ──────────────────────────────────────────────────────
print("\nSummary:")
result = sql("""
SELECT
    meter_id,
    COUNT(*)        AS row_count,
    MIN(timestamp)  AS earliest_timestamp,
    MAX(timestamp)  AS latest_timestamp
FROM meter_readings_source
GROUP BY meter_id
ORDER BY meter_id
""")

rows_out = result.get("rows", [])
if rows_out:
    fields = [f["name"] for f in result.get("fields", [])]
    col_w = [
        max(len(f), max(len(str(r[f])) for r in rows_out))
        for f in fields
    ]
    header = "  ".join(f.ljust(col_w[i]) for i, f in enumerate(fields))
    sep    = "  ".join("-" * w for w in col_w)
    print(header)
    print(sep)
    for row in rows_out:
        print("  ".join(str(row[f]).ljust(col_w[i]) for i, f in enumerate(fields)))
else:
    print(json.dumps(result, indent=2))
