"""Keep the exact v1 split/candidates; correct only auxiliary negative sampling."""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ours_min.protocol import file_sha256, sample_negatives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("new empty output directory required; never overwrite frozen v1")
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    for name, expected in source_manifest["files"].items():
        if file_sha256(args.source / name) != expected:
            raise ValueError(f"source checksum mismatch: {name}")
    splits = {s: np.atleast_2d(np.loadtxt(args.source / f"{s}.csv", delimiter=",",
               skiprows=1, usecols=(0, 1), dtype=np.int64)) for s in ("train", "validation", "test")}
    observed = defaultdict(set)
    for pairs in splits.values():
        for user, item in pairs:
            observed[int(user)].add(int(item))
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "validation.csv", "test.csv", "candidate_items.json", "users.json"):
        shutil.copyfile(args.source / name, args.output / name)
    eval_pairs = np.concatenate([splits["validation"], splits["test"]])
    for seed in source_manifest["seeds"]:
        with np.load(args.source / f"negatives_seed_{seed}.npz") as original:
            train_neg = original["train"]
            expected_train = sample_negatives(splits["train"], source_manifest["num_users"],
                    source_manifest["num_items"], observed, 1, seed)
            if not np.array_equal(train_neg, expected_train):
                raise ValueError("v1 train negatives unexpectedly differ; refusing silent protocol change")
            ratio, remainder = divmod(len(original["auxiliary_eval"]), len(eval_pairs))
            if remainder or ratio != 100:
                raise ValueError("expected exactly 100 auxiliary negatives per positive")
            auxiliary = sample_negatives(eval_pairs, source_manifest["num_users"],
                        source_manifest["num_items"], observed, ratio, seed + 100000)
        np.savez_compressed(args.output / f"negatives_seed_{seed}.npz",
                            train=train_neg, auxiliary_eval=auxiliary)
        print(json.dumps({"seed": seed, "train_negatives_unchanged": True,
                          "auxiliary_negative_item_mean": float(auxiliary[:, 1].mean())}), flush=True)
    manifest = dict(source_manifest)
    manifest.update(protocol_version="multidomain-temporal-leave-two-out-v2",
                    source_protocol_manifest_sha256=file_sha256(args.source / "manifest.json"),
                    negative_sampling="uniform rejection, preserve first distinct accepted draws",
                    changes=["auxiliary negatives fixed; splits, candidates, train negative arrays unchanged"])
    manifest["files"] = {p.name: file_sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
