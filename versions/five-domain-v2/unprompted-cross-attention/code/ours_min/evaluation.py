from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def load_pairs(path: Path):
    values = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.int64)
    if values.size == 0:
        return np.empty((0, 4), dtype=np.int64)
    return np.atleast_2d(values)


def ranking_metrics(scores, targets, train_items_by_user, ks=(10, 20)):
    """Full-candidate ranking with only frozen training interactions masked."""
    if scores.ndim != 2:
        raise ValueError("scores must have shape [num_users, num_candidate_items]")
    metric_sums = {f"recall@{k}": 0.0 for k in ks}
    metric_sums.update({f"ndcg@{k}": 0.0 for k in ks})
    for user, target in targets:
        row = scores[user].copy()
        seen = list(train_items_by_user.get(int(user), ()))
        row[seen] = -np.inf
        row[target] = scores[user, target]
        order = np.argsort(-row, kind="stable")
        rank = int(np.flatnonzero(order == target)[0]) + 1
        for k in ks:
            hit = float(rank <= k)
            metric_sums[f"recall@{k}"] += hit
            metric_sums[f"ndcg@{k}"] += hit / np.log2(rank + 1.0)
    if not len(targets):
        raise ValueError("evaluation targets are empty")
    return {key: value / len(targets) for key, value in metric_sums.items()}


def auxiliary_metrics(scores, positives, negative_pairs):
    by_user = {}
    for user, item in negative_pairs:
        by_user.setdefault(int(user), []).append(int(item))
    labels, values = [], []
    for user, item in positives:
        negatives = by_user.get(int(user), [])
        labels.extend([1] + [0] * len(negatives))
        values.extend([scores[user, item]] + [scores[user, neg] for neg in negatives])
    if len(set(labels)) != 2:
        raise ValueError("AUC/AP require both positive and negative examples")
    return {"auc": float(roc_auc_score(labels, values)),
            "ap": float(average_precision_score(labels, values))}


def evaluate_scores(scores, protocol_dir: Path, seed: int, split: str):
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    train = load_pairs(protocol_dir / "train.csv")[:, :2]
    targets = load_pairs(protocol_dir / f"{split}.csv")[:, :2]
    train_items = {}
    for user, item in train:
        train_items.setdefault(int(user), set()).add(int(item))
    negative_file = np.load(protocol_dir / f"negatives_seed_{seed}.npz")
    all_eval_targets = np.concatenate([
        load_pairs(protocol_dir / "validation.csv")[:, :2],
        load_pairs(protocol_dir / "test.csv")[:, :2],
    ])
    all_negatives = negative_file["auxiliary_eval"]
    offset = 0 if split == "validation" else len(all_eval_targets) // 2
    count = len(targets)
    per_positive = len(all_negatives) // len(all_eval_targets)
    negatives = all_negatives[offset * per_positive:(offset + count) * per_positive]
    result = ranking_metrics(scores, targets, train_items)
    result.update(auxiliary_metrics(scores, targets, negatives))
    return result

