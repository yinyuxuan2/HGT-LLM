"""Combine frozen Item/Category LLM vectors with non-LLM User behavior means."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--encoded-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default=None,
                        help="Override the output variant for a prompt ablation.")
    args = parser.parse_args()
    protocol = json.loads((args.protocol_dir / "manifest.json").read_text())
    prepared = json.loads((args.prepared_dir / "manifest.json").read_text())
    num_users, num_items = protocol["num_users"], protocol["num_items"]
    num_categories = prepared.get("num_categories", prepared.get("counts", {}).get("category"))
    if num_categories is None:
        raise ValueError("missing category count")
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing feature cache")
    encoding_dir = args.encoded_features.parent
    encoding = json.loads((encoding_dir / "features_manifest.json").read_text())
    if encoding["features_sha256"] != sha256(args.encoded_features):
        raise ValueError("encoded feature checksum mismatch")
    if encoding["nodes_sha256"] != sha256(args.prepared_dir / "llm_nodes.jsonl"):
        raise ValueError("encoding belongs to different prompt text")
    with (encoding_dir / "node_ids.txt").open(encoding="utf-8") as ids:
        expected_ids = [json.loads(line)["id"] for line in
                        (args.prepared_dir / "llm_nodes.jsonl").open(encoding="utf-8")]
        if ids.read().splitlines() != expected_ids:
            raise ValueError("encoded node order differs from prompt order")
    encoded = np.load(args.encoded_features, mmap_mode="r")
    if encoded.shape != (num_items + num_categories, 1024):
        raise ValueError(f"unexpected encoded shape: {encoded.shape}")
    if not np.isfinite(encoded).all():
        raise ValueError("non-finite encoded features")
    train = np.loadtxt(args.protocol_dir / "train.csv", delimiter=",", skiprows=1,
                       usecols=(0, 1), dtype=np.int64)
    counts = np.bincount(train[:, 0], minlength=num_users).astype(np.float32)
    sums = np.zeros((num_users, 1024), dtype=np.float32)
    # Chunk the accumulation to keep temporary conversions bounded.
    for start in range(0, len(train), 200_000):
        pairs = train[start:start + 200_000]
        np.add.at(sums, pairs[:, 0], encoded[pairs[:, 1]].astype(np.float32))
    sums /= np.maximum(counts[:, None], 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=np.float16,
        shape=(num_users + num_items + num_categories, 1024))
    output[:num_users] = sums.astype(np.float16)
    output[num_users:] = encoded
    output.flush()
    manifest = {
        "variant": args.variant or prepared["variant"], "shape": list(output.shape), "dtype": "float16",
        "user_features": "mean of frozen embeddings for train-split interacted Items only",
        "num_users": num_users, "num_items": num_items, "num_categories": num_categories,
        "features_sha256": sha256(args.output),
        "encoding_manifest": encoding,
        "train_sha256": sha256(args.protocol_dir / "train.csv"),
        "candidate_items_sha256": sha256(args.protocol_dir / "candidate_items.json"),
    }
    (args.output.parent / "combined_features_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
