import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_interactions(path: Path, columns):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {columns["user"], columns["item"], columns["timestamp"]}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"interaction file is missing columns: {sorted(missing)}")
        for source_row, row in enumerate(reader, 2):
            rows.append((str(row[columns["user"]]), str(row[columns["item"]]),
                         int(row[columns["timestamp"]]), source_row))
    if not rows:
        raise ValueError("interaction file is empty")
    return rows


def temporal_leave_two_out(rows, minimum_interactions=3, short_user_policy="drop"):
    by_user = defaultdict(list)
    for user, item, timestamp, source_row in rows:
        by_user[user].append((item, timestamp, source_row))
    short = sorted(user for user, values in by_user.items() if len(values) < minimum_interactions)
    if short and short_user_policy == "error":
        raise ValueError(f"{len(short)} users have fewer than {minimum_interactions} interactions")
    eligible = sorted(set(by_user) - set(short))
    user_to_idx = {value: idx for idx, value in enumerate(eligible)}
    items = sorted({item for user in eligible for item, _, _ in by_user[user]})
    item_to_idx = {value: idx for idx, value in enumerate(items)}
    splits = {"train": [], "validation": [], "test": []}
    for user in eligible:
        ordered = sorted(by_user[user], key=lambda value: (value[1], value[2]))
        assignments = [("train", value) for value in ordered[:-2]]
        assignments += [("validation", ordered[-2]), ("test", ordered[-1])]
        for split, (item, timestamp, source_row) in assignments:
            splits[split].append((user_to_idx[user], item_to_idx[item], timestamp, source_row))
    return eligible, items, splits, short


def sample_negatives(positive_pairs, num_users, num_items, observed_by_user,
                     negatives_per_positive, seed):
    rng = np.random.default_rng(seed)
    positive_pairs = np.asarray(positive_pairs, dtype=np.int64)
    if negatives_per_positive < 1 or num_items < 1:
        raise ValueError("positive item count and negative sample count required")
    for user in np.unique(positive_pairs[:, 0]):
        if not 0 <= user < num_users:
            raise ValueError(f"user out of range: {user}")
        observed = observed_by_user[user]
        if any(item < 0 or item >= num_items for item in observed):
            raise ValueError("observed item out of range")
        if num_items - len(observed) < negatives_per_positive:
            raise ValueError(f"user {user} has too few unobserved candidate items")
    samples = np.empty((len(positive_pairs) * negatives_per_positive, 2), dtype=np.int64)
    if negatives_per_positive == 1:
        users = positive_pairs[:, 0]
        items = rng.integers(num_items, size=len(users), dtype=np.int64)
        invalid = np.fromiter(
            (int(item) in observed_by_user[int(user)] for user, item in zip(users, items)),
            dtype=bool, count=len(users))
        while invalid.any():
            items[invalid] = rng.integers(num_items, size=int(invalid.sum()), dtype=np.int64)
            positions = np.flatnonzero(invalid)
            invalid[positions] = np.fromiter(
                (int(items[pos]) in observed_by_user[int(users[pos])] for pos in positions),
                dtype=bool, count=len(positions))
        samples[:, 0] = users
        samples[:, 1] = items
        return samples
    cursor = 0
    for user, _item in positive_pairs:
        observed = observed_by_user[user]
        available_count = num_items - len(observed)
        if available_count < negatives_per_positive:
            raise ValueError(f"user {user} has only {available_count} unobserved candidate items")
        # Keep draw order. Sorting an oversized set and retaining its smallest
        # IDs biases the distribution, particularly for auxiliary evaluation.
        chosen, chosen_set = [], set()
        while len(chosen) < negatives_per_positive:
            needed = negatives_per_positive - len(chosen)
            pool = rng.integers(num_items, size=max(needed + 16, needed * 2), dtype=np.int64)
            for item in pool:
                item = int(item)
                if item not in observed and item not in chosen_set:
                    chosen.append(item)
                    chosen_set.add(item)
                    if len(chosen) == negatives_per_positive:
                        break
        values = np.asarray(chosen, dtype=np.int64)
        end = cursor + negatives_per_positive
        samples[cursor:end, 0] = user
        samples[cursor:end, 1] = values
        cursor = end
    return samples


def write_split_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_idx", "item_idx", "timestamp", "source_row"])
        writer.writerows(rows)


def freeze_protocol(interactions_path: Path, output_dir: Path, config):
    if config.get("protocol_version", "").endswith("-v1"):
        raise ValueError("negative sampler changed: assign a new protocol version before freezing")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("refusing to overwrite an existing frozen protocol; use a new version directory")
    rows = load_interactions(interactions_path, config["columns"])
    split_cfg = config["split"]
    users, items, splits, dropped = temporal_leave_two_out(
        rows, split_cfg["minimum_interactions"], split_cfg["short_user_policy"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        write_split_csv(output_dir / f"{name}.csv", values)
    (output_dir / "users.json").write_text(json.dumps(users, ensure_ascii=False), encoding="utf-8")
    (output_dir / "candidate_items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    observed = defaultdict(set)
    for values in splits.values():
        for user, item, *_ in values:
            observed[user].add(item)
    train_pairs = [(u, i) for u, i, *_ in splits["train"]]
    eval_pairs = [(u, i) for name in ("validation", "test") for u, i, *_ in splits[name]]
    for seed in config["seeds"]:
        train_neg = sample_negatives(train_pairs, len(users), len(items), observed,
                                     config["training_negatives_per_positive"], seed)
        eval_neg = sample_negatives(eval_pairs, len(users), len(items), observed,
                                    config["auxiliary_eval_negatives_per_positive"], seed + 100000)
        np.savez_compressed(output_dir / f"negatives_seed_{seed}.npz",
                            train=train_neg, auxiliary_eval=eval_neg)

    manifest = {
        "protocol_version": config["protocol_version"],
        "data_version": config["data_version"],
        "source_file": interactions_path.name,
        "source_sha256": file_sha256(interactions_path),
        "num_users": len(users), "num_items": len(items),
        "num_nodes": len(users) + len(items),
        "num_edges": {name: len(values) for name, values in splits.items()},
        "dropped_short_users": len(dropped),
        "seeds": config["seeds"],
        "candidate_items_sha256": file_sha256(output_dir / "candidate_items.json"),
        "files": {},
    }
    for path in sorted(output_dir.iterdir()):
        if path.name != "manifest.json":
            manifest["files"][path.name] = file_sha256(path)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
