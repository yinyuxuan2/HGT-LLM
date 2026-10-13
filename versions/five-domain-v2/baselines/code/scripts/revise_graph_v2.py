"""Deduplicate exact semantic edges only; retain every node ID and prompt."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from prepare_multidomain_type_aware_graph import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("new empty output directory required")
    manifest = json.loads((args.source / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(args.source / name) != expected:
            raise ValueError(f"source checksum mismatch: {name}")
    with np.load(args.source / "graph_metadata.npz") as graph:
        unique = {name: np.unique(graph[name], axis=0) for name in graph.files}
        removed = {name: int(len(graph[name]) - len(unique[name])) for name in graph.files}
    args.output.mkdir(parents=True, exist_ok=True)
    for name in manifest["files"]:
        if name != "graph_metadata.npz":
            shutil.copyfile(args.source / name, args.output / name)
    np.savez_compressed(args.output / "graph_metadata.npz", **unique)
    manifest.update(graph_revision="exact-edge-dedup-v2", removed_duplicate_edges=removed,
                    protocol_version="multidomain-temporal-leave-two-out-v2",
                    source_manifest_sha256=sha256(args.source / "manifest.json"))
    manifest["num_item_category_edges"] = len(unique["item_category"])
    manifest["num_co_purchase_edges"] = len(unique["co_purchase"])
    manifest["semantic_train_graph_edges"] -= sum(removed.values())
    manifest["files"] = {name: sha256(args.output / name) for name in manifest["files"]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"removed_duplicates": removed, "files": manifest["files"]}), flush=True)


if __name__ == "__main__":
    main()
