"""Strict template ablation: same Item/Category payload, same generic wrapper."""
import argparse
import hashlib
import json
from pathlib import Path


def uniform_text(row):
    text = row["text"]
    if row["type"] == "item":
        prefix, separator = "Product: ", ". Description: "
        if not text.startswith(prefix) or separator not in text or not text.endswith("."):
            raise ValueError("invalid item source template")
        title, description = text[len(prefix):-1].split(separator, 1)
        fields = [title, description]
    elif row["type"] == "category":
        # Category ID gives exact name, including any punctuation in the name.
        name = row["id"].removeprefix("category:")
        prefix = f"Category: {name}. Parent category: "
        if not text.startswith(prefix) or not text.endswith("."):
            raise ValueError("invalid category source template")
        parent, separator, titles = text[len(prefix):-1].partition(". Contains products like: ")
        if not separator:
            raise ValueError("missing parent/products source fields")
        fields = [name, parent, titles]
    else:
        raise ValueError("User nodes must not enter the text encoder")
    return "Text: " + "; ".join(fields) + "."


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-nodes", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("empty output directory required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "llm_nodes.jsonl"
    counts = {"item": 0, "category": 0}
    with args.source_nodes.open(encoding="utf-8") as source, out.open("w", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            row["text"] = uniform_text(row)
            counts[row["type"]] += 1
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"variant": "uniform-prompt-bge-m3-hgt-v2", "counts": counts,
                "template": "Text: {same_semantic_fields}.",
                "category_parent_and_examples_preserved": True,
                "source_nodes_sha256": digest(args.source_nodes), "nodes_sha256": digest(out)}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
