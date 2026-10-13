"""Synthetic integration test. Its outputs must never enter a paper summary."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("empty synthetic test directory required")
    proto, prepared = args.output / "protocol", args.output / "prepared"
    proto.mkdir(parents=True); prepared.mkdir()
    n_users, n_items, n_categories = 4, 128, 3
    splits = {"train": [], "validation": [], "test": []}
    for user in range(n_users):
        for index in range(7):
            split = "train" if index < 5 else "validation" if index == 5 else "test"
            splits[split].append([user, user * 7 + index, index + 1])
    for name, rows in splits.items():
        np.savetxt(proto / f"{name}.csv", np.array(rows), fmt="%d", delimiter=",",
                   header="user_idx,item_idx,timestamp", comments="")
    (proto / "candidate_items.json").write_text(json.dumps(list(range(n_items))))
    (proto / "users.json").write_text(json.dumps(list(range(n_users))))
    train_neg = np.array([[u, 100 + i] for i, (u, _, _) in enumerate(splits["train"])])
    eval_neg = np.array([[u, i] for s in ("validation", "test") for u, _, _ in splits[s]
                         for i in range(28, 128)])
    np.savez_compressed(proto / "negatives_seed_42.npz", train=train_neg, auxiliary_eval=eval_neg)
    manifest = {"num_users": n_users, "num_items": n_items,
                "data_version": "SYNTHETIC_TEST_NOT_EXPERIMENT", "protocol_version": "multidomain-temporal-leave-two-out-v2",
                "files": {p.name: digest(p) for p in proto.iterdir()}}
    (proto / "manifest.json").write_text(json.dumps(manifest))
    np.savez_compressed(prepared / "graph_metadata.npz",
                        item_category=np.array([[i, i % 3] for i in range(n_items)]),
                        co_purchase=np.array([[i, i+1] for i in range(0, 126, 2)]))
    (prepared / "manifest.json").write_text(json.dumps({"num_categories": n_categories,
        "graph_revision": "exact-edge-dedup-v2", "files": {"graph_metadata.npz": digest(prepared / "graph_metadata.npz")}}))
    features = args.output / "features.npy"
    np.save(features, np.random.default_rng(0).normal(0, .01, (n_users+n_items+n_categories,1024)).astype(np.float16))
    fmanifest = args.output / "features_manifest.json"
    fmanifest.write_text(json.dumps({"features_sha256": digest(features), "variant": "uniform-prompt-bge-m3-hgt-v2"}))
    trainer = Path(__file__).with_name("train_hgt_corrected_v2.py")
    for variant in ("pure-hgt", "uniform-prompt-bge-m3-hgt"):
        cmd = [sys.executable, "-B", str(trainer), "--protocol-dir", str(proto), "--prepared-dir", str(prepared),
               "--output-dir", str(args.output / variant), "--seed", "42", "--experiment-name", variant,
               "--hidden-dim", "8", "--heads", "2", "--layers", "1", "--neighbors", "2",
               "--batch-size", "8", "--inference-batch-size", "32", "--epochs", "1", "--device", "cpu"]
        if variant != "pure-hgt":
            cmd += ["--features", str(features), "--features-manifest", str(fmanifest)]
        subprocess.run(cmd, check=True)
        record = json.loads((args.output / variant / "experiment_record.json").read_text())
        assert record["validation"]["ndcg@10"] == record["history"][0]["validation"]["ndcg@10"]
        assert np.load(args.output / variant / "best_hidden.npy").dtype == np.float32
        assert digest(args.output / variant / "best_model.pt") == record["model_sha256"]
        print("PASS end-to-end FP32 and checkpoint integrity:", variant, flush=True)


if __name__ == "__main__":
    main()
