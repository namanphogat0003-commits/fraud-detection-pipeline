"""
Week 1 - Step 1: Load the raw PaySim CSV into a local SQLite database.

Before running this:
1. Download 'Synthetic Financial Datasets For Fraud Detection' (paysim1) from Kaggle:
   https://www.kaggle.com/datasets/ealaxi/paysim1
2. Place the CSV file in data/raw/ and rename it to paysim.csv

The CSV is ~470MB / 6.3M rows, which does not fit comfortably in memory on a
modest machine, so it is streamed into SQLite in chunks rather than loaded
into a single DataFrame.

Run with: python src/data_prep.py [--csv PATH] [--db PATH]
"""

import argparse
import os
import sqlite3
import time

import pandas as pd

DEFAULT_CSV = os.path.join("data", "raw", "paysim.csv")
DEFAULT_DB = os.path.join("data", "processed", "fraud.db")
CHUNK_SIZE = 500_000

# Explicit dtypes keep memory flat and stop pandas re-inferring per chunk.
DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def load_to_sql(csv_path: str, db_path: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Couldn't find {csv_path}. Download the PaySim dataset from Kaggle "
            "and place it there as 'paysim.csv' first."
        )

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    # Bulk-load pragmas: safe here because the DB is rebuildable from the CSV.
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    # SQLite's 2MB default cache makes index creation on 6M+ rows thrash badly
    # (and get OOM-killed on a small VM). 256MB of page cache, spilling temp
    # b-tree sorts to disk rather than memory, keeps the index build stable.
    conn.execute("PRAGMA cache_size = -262144")
    conn.execute("PRAGMA temp_store = FILE")

    total = 0
    start = time.time()
    reader = pd.read_csv(csv_path, dtype=DTYPES, chunksize=CHUNK_SIZE)

    for i, chunk in enumerate(reader):
        chunk.to_sql("transactions", conn, if_exists="append", index=False)
        total += len(chunk)
        print(f"  chunk {i + 1}: {total:,} rows loaded", flush=True)

    # Commit the table before indexing, so an interrupted index build never
    # costs the (expensive) load itself.
    conn.commit()

    for name, cols in [("idx_type", "type"),
                       ("idx_isfraud", "isFraud"),
                       ("idx_step", "step")]:
        print(f"Creating index {name}...", flush=True)
        conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON transactions({cols})")
        conn.commit()

    conn.close()

    elapsed = time.time() - start
    size_mb = os.path.getsize(db_path) / 1e6
    print(f"Done. {total:,} rows -> {db_path} ({size_mb:,.0f} MB) in {elapsed:,.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()
    load_to_sql(args.csv, args.db)
