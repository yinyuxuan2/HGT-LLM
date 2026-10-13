import json
from pathlib import Path

import numpy as np
import torch


REQUIRED_GRAPH_KEYS = {
    "features", "node_type", "edge_index", "edge_type", "edge_time",
    "labels", "train_mask", "val_mask", "test_mask",
}


def load_graph(path: Path):
    data = np.load(path, allow_pickle=False)
    missing = REQUIRED_GRAPH_KEYS - set(data.files)
    if missing:
        raise ValueError(f"graph file is missing keys: {sorted(missing)}")
    graph = {key: torch.from_numpy(data[key]) for key in REQUIRED_GRAPH_KEYS}
    graph["features"] = graph["features"].float()
    graph["edge_index"] = graph["edge_index"].long()
    for key in ("node_type", "edge_type", "edge_time", "labels"):
        graph[key] = graph[key].long()
    for key in ("train_mask", "val_mask", "test_mask"):
        graph[key] = graph[key].bool()
    validate_graph(graph)
    return graph


def validate_graph(graph):
    n = graph["features"].size(0)
    if graph["edge_index"].ndim != 2 or graph["edge_index"].size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if graph["node_type"].numel() != n or graph["labels"].numel() != n:
        raise ValueError("node-level arrays must match number of feature rows")
    if graph["edge_index"].numel() and (graph["edge_index"].min() < 0 or graph["edge_index"].max() >= n):
        raise ValueError("edge_index contains an invalid node index")
    for name in ("node_type", "edge_type"):
        values = torch.unique(graph[name]).cpu().tolist()
        if values != list(range(len(values))):
            raise ValueError(f"{name} IDs must be contiguous from zero; got {values}")
    for split in ("train_mask", "val_mask", "test_mask"):
        if not torch.any(graph[split]):
            raise ValueError(f"{split} is empty")
    overlap = (graph["train_mask"].int() + graph["val_mask"].int() + graph["test_mask"].int()) > 1
    if torch.any(overlap):
        raise ValueError("train/val/test masks overlap")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

