import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ours_min.io import load_graph, write_json
from ours_min.model import OursMin


def parse_args():
    parser = argparse.ArgumentParser(description="Train Ours-min: frozen BGE-M3 features + standard HGT")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="CPU intra-op threads (1 avoids severe overhead on small graphs)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-rte", action="store_true")
    return parser.parse_args()


def score(logits, labels, mask):
    truth = labels[mask].cpu().numpy()
    pred = logits[mask].argmax(-1).cpu().numpy()
    return {"accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0))}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    graph = {key: value.to(args.device) for key, value in load_graph(args.graph).items()}
    labeled = graph["labels"][graph["train_mask"]]
    if torch.any(labeled < 0):
        raise ValueError("all training nodes must have non-negative labels")
    num_classes = int(labeled.max().item()) + 1
    num_types = int(graph["node_type"].max().item()) + 1
    num_relations = int(graph["edge_type"].max().item()) + 1
    model = OursMin(graph["features"].size(1), args.hidden_dim, num_types,
                    num_relations, num_classes, args.heads, args.layers,
                    args.dropout, args.use_rte).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, stale = -1.0, -1, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(graph["features"], graph["node_type"], graph["edge_index"],
                       graph["edge_type"], graph["edge_time"])
        loss = F.cross_entropy(logits[graph["train_mask"]], graph["labels"][graph["train_mask"]])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(graph["features"], graph["node_type"], graph["edge_index"],
                           graph["edge_type"], graph["edge_time"])
            validation = score(logits, graph["labels"], graph["val_mask"])
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "validation": validation})
        if validation["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = validation["macro_f1"], epoch, 0
            torch.save({"model_state": model.state_dict(), "config": vars(args),
                        "num_types": num_types, "num_relations": num_relations,
                        "num_classes": num_classes, "input_dim": graph["features"].size(1)},
                       args.output_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        logits = model(graph["features"], graph["node_type"], graph["edge_index"],
                       graph["edge_type"], graph["edge_time"])
    metrics = {"best_epoch": best_epoch, "validation": score(logits, graph["labels"], graph["val_mask"]),
               "test": score(logits, graph["labels"], graph["test_mask"]), "history": history}
    write_json(args.output_dir / "metrics.json", metrics)
    np.savez_compressed(args.output_dir / "predictions.npz",
                        logits=logits.cpu().numpy(), predictions=logits.argmax(-1).cpu().numpy())
    print(metrics)


if __name__ == "__main__":
    main()
