"""Prepare Amazon Sports & Outdoors with deterministic 5-core filtering.

The implementation is disk-backed so the 2018 full review dump can be processed
without loading all reviews into memory. Duplicate user-item reviews keep the
latest timestamp (and latest source row on ties).
"""

import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
import time
from pathlib import Path

try:
    import ujson as json_parser
except ImportError:
    json_parser = json


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-core", type=int, default=5)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.output_dir / "preprocess.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("CREATE TABLE IF NOT EXISTS interactions (user TEXT, item TEXT, ts INTEGER, row_id INTEGER, PRIMARY KEY(user,item)) WITHOUT ROWID")
    upsert = """INSERT INTO interactions VALUES (?,?,?,?)
    ON CONFLICT(user,item) DO UPDATE SET ts=excluded.ts,row_id=excluded.row_id
    WHERE excluded.ts > interactions.ts OR (excluded.ts = interactions.ts AND excluded.row_id > interactions.row_id)"""
    total = invalid = 0
    batch = []
    with gzip.open(args.reviews, "rt", encoding="utf-8") as stream:
        for row_id, line in enumerate(stream, 1):
            total += 1
            try:
                row = json_parser.loads(line)
                batch.append((str(row["reviewerID"]), str(row["asin"]),
                              int(row["unixReviewTime"]), row_id))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                invalid += 1
            if len(batch) >= 10000:
                conn.executemany(upsert, batch); conn.commit(); batch.clear()
            if row_id % 1_000_000 == 0:
                print(f"read {row_id:,} review rows", flush=True)
    if batch:
        conn.executemany(upsert, batch); conn.commit()
    deduplicated = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]

    iteration = 0
    while True:
        iteration += 1
        before = conn.total_changes
        conn.execute("DELETE FROM interactions WHERE user IN (SELECT user FROM interactions GROUP BY user HAVING COUNT(*) < ?)", (args.k_core,))
        removed_users = conn.total_changes - before
        before = conn.total_changes
        conn.execute("DELETE FROM interactions WHERE item IN (SELECT item FROM interactions GROUP BY item HAVING COUNT(*) < ?)", (args.k_core,))
        removed_items = conn.total_changes - before
        conn.commit()
        print(f"k-core iteration {iteration}: removed user-edges={removed_users:,}, item-edges={removed_items:,}", flush=True)
        if removed_users + removed_items == 0:
            break
    conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user_ts ON interactions(user,ts,row_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_item ON interactions(item)")
    conn.commit()

    interactions_csv = args.output_dir / "interactions.csv"
    with interactions_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["user_id", "item_id", "timestamp"])
        writer.writerows(conn.execute("SELECT user,item,ts FROM interactions ORDER BY user,ts,row_id"))
    item_ids = {row[0] for row in conn.execute("SELECT DISTINCT item FROM interactions")}
    metadata_by_item = {}
    meta_total = meta_invalid = 0
    with gzip.open(args.metadata, "rt", encoding="utf-8") as stream:
        for line in stream:
            meta_total += 1
            try:
                row = json_parser.loads(line); item = str(row["asin"])
            except (KeyError, json.JSONDecodeError):
                meta_invalid += 1; continue
            if item not in item_ids or item in metadata_by_item:
                continue
            parts = [row.get("title", ""), row.get("brand", "")]
            for key in ("category", "description", "feature"):
                value = row.get(key, [])
                parts.extend(value if isinstance(value, list) else [value])
            text = " ".join(str(value).strip() for value in parts if value).replace("\x00", " ")
            metadata_by_item[item] = " ".join(text.split())[:12000]
    missing_meta = len(item_ids - set(metadata_by_item))

    item_titles = {item: (metadata_by_item.get(item) or f"Product {item}") for item in item_ids}
    nodes_path = args.output_dir / "nodes.jsonl"
    with nodes_path.open("w", encoding="utf-8") as stream:
        for user, in conn.execute("SELECT DISTINCT user FROM interactions ORDER BY user"):
            titles = []
            user_count = conn.execute("SELECT COUNT(*) FROM interactions WHERE user=?", (user,)).fetchone()[0]
            for item, in conn.execute("SELECT item FROM interactions WHERE user=? ORDER BY ts,row_id LIMIT ?", (user, user_count - 2)):
                titles.append(item_titles[item][:500])
            text = "User interests: " + " ".join(titles)
            stream.write(json.dumps({"id": f"user:{user}", "type": 0, "text": text[:12000]}, ensure_ascii=False) + "\n")
        for item in sorted(item_ids):
            stream.write(json.dumps({"id": f"item:{item}", "type": 1, "text": item_titles[item]}, ensure_ascii=False) + "\n")

    counts = conn.execute("SELECT COUNT(*),COUNT(DISTINCT user),COUNT(DISTINCT item) FROM interactions").fetchone()
    manifest = {
        "dataset": "Amazon Reviews 2018 / Sports and Outdoors", "data_version": "amazon-sports-2018-5core-v1",
        "k_core": args.k_core, "deduplication": "latest user-item review by (timestamp, source_row)",
        "raw_review_rows": total, "invalid_review_rows": invalid, "deduplicated_pairs": deduplicated,
        "interactions": counts[0], "users": counts[1], "items": counts[2], "nodes": counts[1] + counts[2],
        "raw_metadata_rows": meta_total, "invalid_metadata_rows": meta_invalid, "items_missing_metadata": missing_meta,
        "reviews_sha256": sha256(args.reviews), "metadata_sha256": sha256(args.metadata),
        "interactions_sha256": sha256(interactions_csv), "nodes_sha256": sha256(nodes_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    conn.close()


if __name__ == "__main__":
    main()
