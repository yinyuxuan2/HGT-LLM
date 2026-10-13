"""Scalable Ours-min training on the frozen Amazon Sports protocol."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ours_min.link_model import OursMinLink


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pairs(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1), dtype=np.int64)
    return np.atleast_2d(data)


def make_graph(features_path, train_pairs, num_users, num_items):
    features = np.load(features_path, mmap_mode="r+")
    if features.shape != (num_users + num_items, 1024):
        raise ValueError(f"unexpected feature shape: {features.shape}")
    source = torch.from_numpy(train_pairs[:, 0].copy()).long()
    item = torch.from_numpy((train_pairs[:, 1] + num_users).copy()).long()
    edge_index = torch.stack([torch.cat([source, item]), torch.cat([item, source])])
    edge_type = torch.cat([torch.zeros(len(source), dtype=torch.long),
                           torch.ones(len(source), dtype=torch.long)])
    return Data(x=torch.from_numpy(features), edge_index=edge_index,
                node_type=torch.cat([torch.zeros(num_users, dtype=torch.long),
                                     torch.ones(num_items, dtype=torch.long)]),
                edge_type=edge_type, edge_time=torch.zeros(edge_index.size(1), dtype=torch.long))


@torch.inference_mode()
def infer_all(model, data, device, batch_size, workers):
    model.eval()
    node_count = data.num_nodes
    hidden_dim = model.hgt.layers[0].dim
    hidden = torch.empty((node_count, hidden_dim), dtype=torch.float32)
    for start in range(0, node_count, batch_size):
        end = min(start + batch_size, node_count)
        x = data.x[start:end].to(device).float()
        types = data.node_type[start:end].to(device)
        out = torch.zeros((end - start, hidden_dim), device=device)
        for type_id, adapter in enumerate(model.hgt.type_adapters):
            mask = types == type_id
            if torch.any(mask):
                out[mask] = torch.tanh(adapter(x[mask]))
        hidden[start:end] = out.cpu()
    for layer in model.hgt.layers:
        layer_data = Data(x=hidden, edge_index=data.edge_index, node_type=data.node_type,
                          edge_type=data.edge_type, edge_time=data.edge_time)
        loader = NeighborLoader(layer_data, input_nodes=None, num_neighbors=[-1],
                                batch_size=batch_size, shuffle=False, num_workers=workers,
                                directed=True)
        next_hidden = torch.empty_like(hidden)
        layer.eval()
        for batch in loader:
            batch = batch.to(device)
            out = layer(batch.x, batch.node_type, batch.edge_index,
                        batch.edge_type, batch.edge_time)
            next_hidden[batch.n_id[:batch.batch_size]] = out[:batch.batch_size].cpu()
        hidden = next_hidden
    return hidden


@torch.inference_mode()
def ranking_metrics(hidden, train_pairs, targets, num_users, device, batch_size=128):
    user_emb = hidden[:num_users].to(device)
    item_emb = hidden[num_users:].to(device)
    counts = np.bincount(train_pairs[:, 0], minlength=num_users)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    train_items = train_pairs[:, 1]
    sums = {split: {"recall@10": 0.0, "recall@20": 0.0,
                    "ndcg@10": 0.0, "ndcg@20": 0.0} for split in targets}
    item_ids = torch.arange(item_emb.size(0), device=device).view(1, -1)
    for start in range(0, num_users, batch_size):
        end = min(start + batch_size, num_users)
        scores = user_emb[start:end] @ item_emb.T
        for local, user in enumerate(range(start, end)):
            seen = torch.from_numpy(train_items[indptr[user]:indptr[user + 1]]).to(device)
            scores[local, seen] = -torch.inf
        for split, target_array in targets.items():
            target = torch.from_numpy(target_array[start:end]).to(device)
            row = torch.arange(end - start, device=device)
            target_score = scores[row, target].unsqueeze(1)
            ranks = 1 + (scores > target_score).sum(1)
            ranks += ((scores == target_score) & (item_ids < target.unsqueeze(1))).sum(1)
            ranks = ranks.float()
            for k in (10, 20):
                hits = ranks <= k
                sums[split][f"recall@{k}"] += hits.sum().item()
                sums[split][f"ndcg@{k}"] += torch.where(
                    hits, 1.0 / torch.log2(ranks + 1.0), torch.zeros_like(ranks)).sum().item()
    return {split: {key: value / num_users for key, value in metrics.items()}
            for split, metrics in sums.items()}


@torch.inference_mode()
def auxiliary_metrics(hidden, positives, negative_pairs, num_users, device, batch_size=262144):
    """Global AUC/AP over the frozen positive and auxiliary-negative pairs."""
    user_emb = hidden[:num_users].to(device)
    item_emb = hidden[num_users:].to(device)

    def score_pairs(pairs):
        output = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), batch_size):
            end = min(start + batch_size, len(pairs))
            users = torch.as_tensor(pairs[start:end, 0], device=device)
            items = torch.as_tensor(pairs[start:end, 1], device=device)
            output[start:end] = (user_emb[users] * item_emb[items]).sum(1).cpu().numpy()
        return output

    positive_scores = score_pairs(positives[:, :2])
    negative_scores = score_pairs(negative_pairs[:, :2])
    scores = np.concatenate([positive_scores, negative_scores])
    labels = np.concatenate([np.ones(len(positive_scores), dtype=np.uint8),
                             np.zeros(len(negative_scores), dtype=np.uint8)])
    order = np.argsort(scores, kind="stable")
    ordered_scores, ordered_labels = scores[order], labels[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(ordered_scores)) + 1, len(scores)]
    positives_by_group = np.add.reduceat(ordered_labels.astype(np.int64), boundaries[:-1])
    totals_by_group = np.diff(boundaries)
    negatives_by_group = totals_by_group - positives_by_group
    negatives_before = np.r_[0, np.cumsum(negatives_by_group)[:-1]]
    n_positive, n_negative = len(positive_scores), len(negative_scores)
    auc = np.sum(positives_by_group * (negatives_before + 0.5 * negatives_by_group))
    auc /= n_positive * n_negative
    descending_positive = positives_by_group[::-1]
    descending_total = totals_by_group[::-1]
    cumulative_positive = np.cumsum(descending_positive)
    cumulative_total = np.cumsum(descending_total)
    ap = np.sum((cumulative_positive / cumulative_total) * descending_positive) / n_positive
    return {"auc": float(auc), "ap": float(ap)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--neighbors", type=int, nargs="+", default=[15, 10])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed); torch.use_deterministic_algorithms(True, warn_only=True)
    manifest = json.loads((args.protocol_dir / "manifest.json").read_text())
    num_users, num_items = manifest["num_users"], manifest["num_items"]
    train = load_pairs(args.protocol_dir / "train.csv")
    validation = load_pairs(args.protocol_dir / "validation.csv")
    test = load_pairs(args.protocol_dir / "test.csv")
    negatives = np.load(args.protocol_dir / f"negatives_seed_{args.seed}.npz")["train"]
    if len(negatives) != len(train):
        raise ValueError("frozen training negatives do not align with positives")
    data = make_graph(args.features, train, num_users, num_items)
    positive_edges = np.stack([train[:, 0], train[:, 1] + num_users])
    negative_edges = np.stack([negatives[:, 0], negatives[:, 1] + num_users])
    label_index = torch.from_numpy(np.concatenate([positive_edges, negative_edges], axis=1)).long()
    labels = torch.cat([torch.ones(len(train)), torch.zeros(len(train))])
    loader = LinkNeighborLoader(data, num_neighbors=args.neighbors,
                                edge_label_index=label_index, edge_label=labels,
                                batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, directed=True)
    model = OursMinLink(1024, args.hidden_dim, num_heads=args.heads,
                        num_layers=args.layers, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best_value, stale, history, start_epoch = -1.0, 0, [], 1
    last_path = args.output_dir / "last_checkpoint.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        best_value, stale, history = state["best_value"], state["stale"], state["history"]
        start_epoch = state["epoch"] + 1
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
        print(json.dumps({"resumed_from_epoch": state["epoch"]}), flush=True)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); total_loss = total_examples = 0
        for batch in loader:
            batch = batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            hidden = model(batch.x.float(), batch.node_type, batch.edge_index,
                           batch.edge_type, batch.edge_time)
            logits = model.score_pairs(hidden, batch.edge_label_index)
            loss = F.binary_cross_entropy_with_logits(logits, batch.edge_label)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * batch.edge_label.numel()
            total_examples += batch.edge_label.numel()
        all_hidden = infer_all(model, data, args.device, args.inference_batch_size, args.workers)
        ranked = ranking_metrics(all_hidden, train,
                                 {"validation": validation[:, 1]},
                                 num_users, args.device)
        record = {"epoch": epoch, "train_loss": total_loss / total_examples,
                  "validation": ranked["validation"]}
        history.append(record); print(json.dumps(record), flush=True)
        value = ranked["validation"]["ndcg@10"]
        if value > best_value:
            best_value, stale = value, 0
            torch.save({"model_state": model.state_dict(), "args": vars(args),
                        "trainable_parameters": trainable, "epoch": epoch},
                       args.output_dir / "best_model.pt")
            np.save(args.output_dir / "best_hidden.npy", all_hidden.numpy().astype(np.float16))
        else:
            stale += 1
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "best_value": best_value,
                    "stale": stale, "history": history, "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                   last_path)
        if stale >= args.patience:
            break
    model_path = args.output_dir / "best_model.pt"
    metrics_path = args.output_dir / "ranking_metrics.json"
    best_hidden = torch.from_numpy(np.load(args.output_dir / "best_hidden.npy")).float()
    final_metrics = ranking_metrics(
        best_hidden, train,
        {"validation": validation[:, 1], "test": test[:, 1]},
        num_users, args.device)
    frozen = np.load(args.protocol_dir / f"negatives_seed_{args.seed}.npz", mmap_mode="r")
    auxiliary = frozen["auxiliary_eval"]
    per_split = len(validation) * (len(auxiliary) // (len(validation) + len(test)))
    final_metrics["validation"].update(auxiliary_metrics(
        best_hidden, validation, auxiliary[:per_split], num_users, args.device))
    final_metrics["test"].update(auxiliary_metrics(
        best_hidden, test, auxiliary[per_split:], num_users, args.device))
    metrics_path.write_text(json.dumps(final_metrics, indent=2) + "\n", encoding="utf-8")
    result = {
        "experiment_id": f"ours-min__amazon-sports-2018-5core-v1__seed-{args.seed}",
        "protocol_version": manifest["protocol_version"], "data_version": manifest["data_version"],
        "model_name": "Ours-min", "seed": args.seed, "hyperparameters": vars(args),
        "trainable_parameters": trainable,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "training_time_seconds": time.perf_counter() - started,
        "model_file": str(model_path), "model_sha256": sha256(model_path),
        "result_file": str(metrics_path), "result_sha256": sha256(metrics_path),
        "validation": final_metrics["validation"], "test": final_metrics["test"],
        "history": history,
    }
    (args.output_dir / "experiment_record.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
