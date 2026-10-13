"""Train Type-Aware Prompt Encoding + standard HGT on the frozen protocol."""

import argparse
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
from ours_min.standard_hgt import StandardHGTConv
from scripts.train_hgt_link import auxiliary_metrics, load_pairs, ranking_metrics, sha256
from type_aware.link_model import TypeAwareHGTLink
from ours_min.pure_hgt import PureHGTLink


def make_graph(features_path, metadata_path, train, num_users, num_items, num_categories):
    if features_path is None:
        # Pure HGT must never access any BGE/text feature file.
        features = np.zeros((num_users + num_items + num_categories, 1), dtype=np.float32)
    else:
        features = np.load(features_path, mmap_mode="r")
        expected = (num_users + num_items + num_categories, 1024)
        if features.shape != expected:
            raise ValueError(f"unexpected feature shape {features.shape}, expected {expected}")
    metadata = np.load(metadata_path)
    source_parts, target_parts, relation_parts = [], [], []

    users = train[:, 0]
    items = train[:, 1] + num_users
    purchase_time = train[:, 2] if train.shape[1] > 2 else np.zeros(len(train), dtype=np.int64)
    source_parts += [users, items]
    target_parts += [items, users]
    relation_parts += [np.zeros(len(train), dtype=np.int64), np.zeros(len(train), dtype=np.int64)]
    time_parts = [purchase_time, purchase_time]

    item_category = metadata["item_category"]
    item_nodes = item_category[:, 0] + num_users
    category_nodes = item_category[:, 1] + num_users + num_items
    source_parts += [item_nodes, category_nodes]
    target_parts += [category_nodes, item_nodes]
    relation_parts += [np.ones(len(item_category), dtype=np.int64),
                       np.ones(len(item_category), dtype=np.int64)]
    time_parts += [np.zeros(len(item_category), dtype=np.int64),
                   np.zeros(len(item_category), dtype=np.int64)]

    co_purchase = metadata["co_purchase"]
    left = co_purchase[:, 0] + num_users
    right = co_purchase[:, 1] + num_users
    source_parts += [left, right]
    target_parts += [right, left]
    relation_parts += [np.full(len(co_purchase), 2, dtype=np.int64),
                       np.full(len(co_purchase), 2, dtype=np.int64)]
    time_parts += [np.zeros(len(co_purchase), dtype=np.int64),
                   np.zeros(len(co_purchase), dtype=np.int64)]

    edge_index = torch.from_numpy(np.stack([
        np.concatenate(source_parts), np.concatenate(target_parts)]))
    edge_type = torch.from_numpy(np.concatenate(relation_parts).astype(np.int64))
    edge_time = torch.from_numpy(np.concatenate(time_parts).astype(np.int64))
    node_type = torch.cat([
        torch.zeros(num_users, dtype=torch.long),
        torch.ones(num_items, dtype=torch.long),
        torch.full((num_categories,), 2, dtype=torch.long),
    ])
    return Data(x=torch.from_numpy(features), edge_index=edge_index.long(),
                node_type=node_type, edge_type=edge_type,
                edge_time=edge_time)


@torch.inference_mode()
def infer_all(model, data, device, batch_size, workers):
    model.eval()
    node_count = data.num_nodes
    hidden = torch.empty((node_count, model.hidden_dim), dtype=torch.float32)
    for start in range(0, node_count, batch_size):
        end = min(start + batch_size, node_count)
        ids = torch.arange(start, end, device=device)
        hidden[start:end] = model.initial_hidden(
            data.x[start:end].to(device).float(), data.node_type[start:end].to(device), ids).cpu()
    for layer in model.layers:
        layer_data = Data(x=hidden, edge_index=data.edge_index, node_type=data.node_type,
                          edge_type=data.edge_type, edge_time=data.edge_time)
        loader = NeighborLoader(layer_data, input_nodes=None, num_neighbors=[-1],
                                batch_size=batch_size, shuffle=False,
                                num_workers=workers, directed=True)
        next_hidden = torch.empty_like(hidden)
        layer.eval()
        for batch in loader:
            batch = batch.to(device)
            out = layer(batch.x, batch.node_type, batch.edge_index,
                        batch.edge_type, batch.edge_time)
            next_hidden[batch.n_id[:batch.batch_size]] = out[:batch.batch_size].cpu()
        hidden = next_hidden
    return hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path)
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
    parser.add_argument("--experiment-name", default="type-aware-prompt-hgt",
                        choices=("type-aware-prompt-hgt", "unprompted-bge-m3-hgt", "pure-hgt"))
    args = parser.parse_args()
    pure = args.experiment_name == "pure-hgt"
    if pure and args.features is not None:
        parser.error("pure-hgt forbids --features: all representations are trainable node IDs")
    if not pure and args.features is None:
        parser.error("BGE experiments require --features")
    started = time.perf_counter()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed); torch.use_deterministic_algorithms(True, warn_only=True)
    manifest = json.loads((args.protocol_dir / "manifest.json").read_text())
    prepared = json.loads((args.prepared_dir / "manifest.json").read_text())
    num_users, num_items = manifest["num_users"], manifest["num_items"]
    num_categories = prepared["num_categories"]
    protocol_hashes = {}
    if pure:
        for filename, expected_hash in manifest["files"].items():
            actual = sha256(args.protocol_dir / filename)
            if actual != expected_hash:
                raise ValueError(f"frozen protocol checksum mismatch: {filename}")
            protocol_hashes[filename] = actual
        if sha256(args.prepared_dir / "graph_metadata.npz") != "a6d799cc33ef287d007b2258964770c298510651f01da5e9f120042d8faa7b85":
            raise ValueError("graph differs from frozen five-domain baseline")
        candidates = json.loads((args.protocol_dir / "candidate_items.json").read_text())
        if len(candidates) != num_items or len(set(candidates)) != num_items:
            raise ValueError("candidate mapping must cover every item exactly once")
    train = np.atleast_2d(np.loadtxt(args.protocol_dir / "train.csv", delimiter=",",
                                    skiprows=1, usecols=(0, 1, 2), dtype=np.int64))
    validation = load_pairs(args.protocol_dir / "validation.csv")
    test = load_pairs(args.protocol_dir / "test.csv")
    if pure:
        if np.any(np.diff(train[:, 0]) < 0):
            raise ValueError("shared ranking evaluator requires user-sorted training data")
        if not all(np.array_equal(pairs[:, 0], np.arange(num_users)) for pairs in (validation, test)):
            raise ValueError("shared evaluator requires one held-out item per ordered user")
    negatives = np.load(args.protocol_dir / f"negatives_seed_{args.seed}.npz")["train"]
    data = make_graph(args.features, args.prepared_dir / "graph_metadata.npz", train,
                      num_users, num_items, num_categories)
    positive = np.stack([train[:, 0], train[:, 1] + num_users])
    negative = np.stack([negatives[:, 0], negatives[:, 1] + num_users])
    label_index = torch.from_numpy(np.concatenate([positive, negative], axis=1)).long()
    labels = torch.cat([torch.ones(len(train)), torch.zeros(len(train))])
    loader = LinkNeighborLoader(data, num_neighbors=args.neighbors,
                                edge_label_index=label_index, edge_label=labels,
                                batch_size=args.batch_size, shuffle=True,
                                num_workers=args.workers, directed=True)
    if pure:
        model = PureHGTLink(num_users + num_items + num_categories, args.hidden_dim,
                            args.heads, args.layers, 3, args.dropout).to(args.device)
    else:
        model = TypeAwareHGTLink(1024, args.hidden_dim, num_users, args.heads,
                                 args.layers, 3, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if pure:
        run_config = {"hyperparameters": vars(args), "protocol_file_hashes": protocol_hashes,
                      "trainable_parameters": trainable, "uses_language_model": False,
                      "representation": "trainable IDs for User, Item, Category; normal(0, 0.02)",
                      "node_counts": {"user": num_users, "item": num_items, "category": num_categories},
                      "directed_edges": data.num_edges,
                      "graph_metadata_sha256": sha256(args.prepared_dir / "graph_metadata.npz")}
        config_path = args.output_dir / "run_config.json"
        normalized_config = json.loads(json.dumps(run_config, default=str))
        if config_path.exists() and json.loads(config_path.read_text()) != normalized_config:
            raise ValueError("cannot resume with a changed pure-HGT configuration")
        config_path.write_text(json.dumps(normalized_config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "initialized", "experiment": args.experiment_name,
                      "seed": args.seed, "trainable_parameters": trainable,
                      "num_nodes": data.num_nodes, "directed_edges": data.num_edges,
                      "uses_language_model": not pure}), flush=True)
    best_value, stale, history, start_epoch = -1.0, 0, [], 1
    previous_elapsed, previous_peak = 0.0, 0.0
    last_path = args.output_dir / "last_checkpoint.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model_state"]); optimizer.load_state_dict(state["optimizer_state"])
        best_value, stale, history = state["best_value"], state["stale"], state["history"]
        start_epoch = state["epoch"] + 1
        previous_elapsed = state.get("elapsed_seconds", 0.0)
        previous_peak = state.get("peak_gpu_memory_mb", 0.0)
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        print(json.dumps({"resumed_from_epoch": state["epoch"]}), flush=True)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); total_loss = total_examples = 0
        for batch_index, batch in enumerate(loader, 1):
            batch = batch.to(args.device); optimizer.zero_grad(set_to_none=True)
            hidden = model(batch.x.float(), batch.node_type, batch.edge_index,
                           batch.edge_type, batch.edge_time, batch.n_id)
            logits = model.score_pairs(hidden, batch.edge_label_index)
            loss = F.binary_cross_entropy_with_logits(logits, batch.edge_label)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * batch.edge_label.numel(); total_examples += batch.edge_label.numel()
            if pure and batch_index % 250 == 0:
                print(json.dumps({"event": "training_progress", "seed": args.seed, "epoch": epoch,
                                  "batch": batch_index, "total_batches": len(loader)}), flush=True)
        all_hidden = infer_all(model, data, args.device, args.inference_batch_size, args.workers)
        ranked = ranking_metrics(all_hidden[:num_users + num_items], train,
                                 {"validation": validation[:, 1]}, num_users, args.device)
        record = {"epoch": epoch, "train_loss": total_loss / total_examples,
                  "validation": ranked["validation"]}
        history.append(record); print(json.dumps(record), flush=True)
        value = record["validation"]["ndcg@10"]
        if value > best_value:
            best_value, stale = value, 0
            torch.save({"model_state": model.state_dict(), "args": vars(args),
                        "trainable_parameters": trainable, "epoch": epoch},
                       args.output_dir / "best_model.pt")
            np.save(args.output_dir / "best_hidden.npy", all_hidden[:num_users + num_items].numpy().astype(np.float16))
        else:
            stale += 1
        checkpoint_tmp = last_path.with_suffix(".tmp")
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "best_value": best_value,
                    "stale": stale, "history": history, "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
                    "peak_gpu_memory_mb": max(previous_peak, torch.cuda.max_memory_allocated() / 2**20)}, checkpoint_tmp)
        checkpoint_tmp.replace(last_path)
        if stale >= args.patience:
            break
    best_hidden = torch.from_numpy(np.load(args.output_dir / "best_hidden.npy")).float()
    final = ranking_metrics(best_hidden, train, {"validation": validation[:, 1], "test": test[:, 1]},
                            num_users, args.device)
    frozen = np.load(args.protocol_dir / f"negatives_seed_{args.seed}.npz", mmap_mode="r")
    auxiliary = frozen["auxiliary_eval"]
    per_split = len(validation) * (len(auxiliary) // (len(validation) + len(test)))
    final["validation"].update(auxiliary_metrics(best_hidden, validation, auxiliary[:per_split], num_users, args.device))
    final["test"].update(auxiliary_metrics(best_hidden, test, auxiliary[per_split:], num_users, args.device))
    metrics_path = args.output_dir / "ranking_metrics.json"
    metrics_path.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    model_path = args.output_dir / "best_model.pt"
    result = {"experiment_id": f"{args.experiment_name}__{manifest['data_version']}__seed-{args.seed}",
              "protocol_version": manifest["protocol_version"], "data_version": manifest["data_version"],
              "model_name": ("Type-Aware Prompt HGT" if args.experiment_name == "type-aware-prompt-hgt"
                             else "Pure HGT (trainable node IDs only)" if pure
                             else "Unprompted BGE-M3 HGT"), "seed": args.seed,
              "hyperparameters": vars(args), "trainable_parameters": trainable,
              "features_sha256": sha256(args.features) if args.features is not None else None,
              "uses_language_model": not pure,
              "protocol_file_hashes": protocol_hashes,
              "graph_metadata_sha256": sha256(args.prepared_dir / "graph_metadata.npz"),
              "peak_gpu_memory_mb": round(max(previous_peak, torch.cuda.max_memory_allocated() / 2**20), 2),
              "training_time_seconds": previous_elapsed + time.perf_counter() - started,
              "model_file": str(model_path), "model_sha256": sha256(model_path),
              "result_file": str(metrics_path), "result_sha256": sha256(metrics_path),
              "validation": final["validation"], "test": final["test"], "history": history}
    (args.output_dir / "experiment_record.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
