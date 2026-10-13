"""Build five-domain Type-Aware prompts and three-relation graph metadata."""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import ujson as json_parser
except ImportError:
    json_parser = json


def clean(value):
    if isinstance(value, list):
        value = " ".join(str(part) for part in value if part)
    return " ".join(str(value or "").replace("\x00", " ").split())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representative-products", type=int, default=5)
    parser.add_argument("--co-purchase-per-item", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = json.loads((args.protocol_dir / "manifest.json").read_text(encoding="utf-8"))
    items = json.loads((args.protocol_dir / "candidate_items.json").read_text(encoding="utf-8"))
    item_to_idx = {asin: idx for idx, asin in enumerate(items)}
    wanted = set(items)
    records = {}
    scanned = invalid = 0
    for domain in config["domains"]:
        with gzip.open(domain["metadata"], "rt", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                scanned += 1
                try:
                    row = json_parser.loads(line)
                    asin = str(row["asin"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    invalid += 1
                    continue
                if asin not in wanted:
                    continue
                record = records.setdefault(asin, {"title": "", "description": "", "categories": set(), "also_buy": []})
                if not record["title"]:
                    record["title"] = clean(row.get("title")) or f"Product {asin}"
                    record["description"] = clean(row.get("description"))[:200]
                values = row.get("category") or []
                if isinstance(values, list):
                    parent = "None"
                    for value in values:
                        name = clean(value)
                        if name:
                            record["categories"].add((name, parent))
                            parent = name
                also = row.get("also_buy") or []
                if isinstance(also, list):
                    for other in also:
                        other = str(other)
                        if other in wanted and other != asin and other not in record["also_buy"]:
                            record["also_buy"].append(other)

    # Category identity is canonicalized by name, intentionally allowing exact
    # shared subcategory names to bridge source domains without extra edge types.
    parents_by_category = defaultdict(lambda: defaultdict(int))
    products_by_category = defaultdict(set)
    for asin, record in records.items():
        for name, parent in record["categories"]:
            parents_by_category[name][parent] += 1
            products_by_category[name].add(asin)
    categories = sorted(products_by_category)
    category_to_idx = {name: idx for idx, name in enumerate(categories)}
    item_category = []
    for asin in items:
        for name, _parent in sorted(records.get(asin, {}).get("categories", set())):
            item_category.append((item_to_idx[asin], category_to_idx[name]))

    co_edges = set()
    for asin in items:
        for other in records.get(asin, {}).get("also_buy", [])[:args.co_purchase_per_item]:
            left, right = sorted((item_to_idx[asin], item_to_idx[other]))
            if left != right:
                co_edges.add((left, right))
    co_purchase = sorted(co_edges)

    nodes_path = args.output_dir / "llm_nodes.jsonl"
    with nodes_path.open("w", encoding="utf-8") as stream:
        for asin in items:
            record = records.get(asin, {})
            title = record.get("title") or f"Product {asin}"
            description = record.get("description") or ""
            stream.write(json.dumps({"id": f"item:{asin}", "type": "item",
                                     "text": f"Product: {title}. Description: {description}."}, ensure_ascii=False) + "\n")
        for name in categories:
            parent_counts = parents_by_category[name]
            parent = sorted(parent_counts, key=lambda value: (-parent_counts[value], value))[0]
            candidates = sorted(products_by_category[name],
                                key=lambda asin: hashlib.sha256((name + "|" + asin).encode()).digest())
            titles = [(records.get(asin, {}).get("title") or f"Product {asin}")
                      for asin in candidates[:args.representative_products]]
            examples = "; ".join(titles) if titles else "None"
            text = f"Category: {name}. Parent category: {parent}. Contains products like: {examples}."
            stream.write(json.dumps({"id": f"category:{name}", "type": "category", "text": text}, ensure_ascii=False) + "\n")

    metadata_path = args.output_dir / "graph_metadata.npz"
    np.savez_compressed(metadata_path,
                        item_category=np.asarray(item_category, dtype=np.int64).reshape(-1, 2),
                        co_purchase=np.asarray(co_purchase, dtype=np.int64).reshape(-1, 2))
    categories_path = args.output_dir / "categories.json"
    categories_path.write_text(json.dumps([
        {"index": idx, "name": name,
         "parent": sorted(parents_by_category[name], key=lambda p: (-parents_by_category[name][p], p))[0]}
        for idx, name in enumerate(categories)], ensure_ascii=False), encoding="utf-8")
    semantic_edges = protocol["num_edges"]["train"] + len(item_category) + len(co_purchase)
    manifest = {
        "variant": "type-aware-prompt-hgt-multidomain-v1",
        "protocol_version": protocol["protocol_version"],
        "num_items": len(items), "num_categories": len(categories),
        "num_llm_nodes": len(items) + len(categories),
        "num_item_category_edges": len(item_category),
        "num_co_purchase_edges": len(co_purchase),
        "semantic_train_graph_edges": semantic_edges,
        "relation_types": ["purchase", "belongs_to", "co_purchase"],
        "forbidden_relation_types": ["view"],
        "category_bridge": "exact normalized category-name sharing across source domains",
        "co_purchase_source": f"metadata also_buy, first {args.co_purchase_per_item} selected neighbors per item, undirected deduplication",
        "item_template": "Product: {title}. Description: {description[:200]}.",
        "category_template": "Category: {name}. Parent category: {most frequent parent}. Contains products like: {up to 5 deterministic representative titles}.",
        "user_encoder": "trainable ID embedding + mean pooling of frozen train-item embeddings",
        "metadata_rows_scanned": scanned, "invalid_metadata_rows": invalid,
        "items_missing_metadata": len(items) - len(records),
        "files": {"llm_nodes.jsonl": sha256(nodes_path),
                  "graph_metadata.npz": sha256(metadata_path),
                  "categories.json": sha256(categories_path)},
    }
    if not 2_000_000 <= semantic_edges <= 5_000_000:
        raise RuntimeError(f"semantic edge count outside requested range: {semantic_edges}")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
