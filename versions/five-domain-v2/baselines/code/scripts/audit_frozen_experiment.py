"""Read-only full-array checks of frozen graph/protocol; save audit separately."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.protocol / "manifest.json").read_text())
    n = manifest["num_items"]
    splits = {s: np.atleast_2d(np.loadtxt(args.protocol / (s + ".csv"), delimiter=",",
              skiprows=1, usecols=(0, 1, 2), dtype=np.int64)) for s in ("train", "validation", "test")}
    keys = {s: values[:, 0] * n + values[:, 1] for s, values in splits.items()}
    audit = {"protocol": manifest["protocol_version"], "full_arrays_not_sampled": True,
             "split_sizes": {s: len(v) for s, v in splits.items()},
             "duplicates_within_split": {s: int(len(v) - len(np.unique(v))) for s, v in keys.items()},
             "cross_split_pair_overlap": {a + "/" + b: int(np.intersect1d(keys[a], keys[b]).size)
                     for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}}
    max_train = np.full(manifest["num_users"], np.iinfo(np.int64).min, dtype=np.int64)
    np.maximum.at(max_train, splits["train"][:, 0], splits["train"][:, 2])
    val, test = splits["validation"], splits["test"]
    audit["temporal_checks"] = {"train_after_validation": int((max_train[val[:, 0]] > val[:, 2]).sum()),
                                 "validation_after_test": int((val[:, 2] > test[:, 2]).sum()),
                                 "train_validation_timestamp_ties": int((max_train[val[:, 0]] == val[:, 2]).sum()),
                                 "validation_test_timestamp_ties": int((val[:, 2] == test[:, 2]).sum())}
    audit["negative_item_distributions"] = {}
    for seed in (42, 43, 44):
        with np.load(args.protocol / f"negatives_seed_{seed}.npz") as frozen:
            audit["negative_item_distributions"][str(seed)] = {
                kind: {"count": len(frozen[kind]), "mean_item_index": float(frozen[kind][:, 1].mean()),
                       "fraction_lower_half": float((frozen[kind][:, 1] < n / 2).mean())}
                for kind in ("train", "auxiliary_eval")}
    prepared = json.loads((args.prepared / "manifest.json").read_text())
    with np.load(args.prepared / "graph_metadata.npz") as graph:
        belongs = graph["item_category"]
        counts = np.bincount(belongs[:, 1], minlength=prepared["num_categories"])
        audit["categories_with_fewer_than_3_linked_items"] = int((counts < 3).sum())
        audit["graph_duplicate_rows"] = {name: int(len(graph[name]) - len(np.unique(graph[name], axis=0)))
                                          for name in ("item_category", "co_purchase")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
