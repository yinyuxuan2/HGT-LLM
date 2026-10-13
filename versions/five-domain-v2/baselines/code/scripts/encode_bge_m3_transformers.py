"""Resumable frozen BGE-M3 dense encoding with Hugging Face Transformers."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_state(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Windows indexers/antivirus can briefly hold the destination file open.
    # Retry the atomic replace instead of losing a completed encoding batch.
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.output_dir / "features.npy"
    ids_path = args.output_dir / "node_ids.txt"
    state_path = args.output_dir / "encoding_state.json"
    manifest_path = args.output_dir / "features_manifest.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"processed": 0, "elapsed_seconds": 0.0}
    processed = int(state["processed"])
    fingerprint = {"nodes_sha256": file_sha256(args.nodes),
                   "model_weights_sha256": file_sha256(args.model / "pytorch_model.bin"),
                   "num_nodes": args.num_nodes, "max_length": args.max_length,
                   "batch_size": args.batch_size, "pooling": "normalized_cls"}
    if processed and state.get("input_fingerprint") != fingerprint:
        raise ValueError("encoding resume inputs/settings are different or unverified")

    if features_path.exists():
        features = np.lib.format.open_memmap(features_path, mode="r+")
        if features.shape != (args.num_nodes, 1024):
            raise ValueError(f"existing feature shape {features.shape} is not {(args.num_nodes, 1024)}")
    else:
        if processed:
            raise ValueError("state exists but features.npy does not")
        features = np.lib.format.open_memmap(features_path, mode="w+", dtype=np.float16,
                                             shape=(args.num_nodes, 1024))
        ids_path.write_text("", encoding="utf-8")
    existing_ids = sum(1 for _ in ids_path.open(encoding="utf-8")) if ids_path.exists() else 0
    if existing_ids != processed:
        raise ValueError(f"node_ids has {existing_ids} rows but state says {processed}")

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True, torch_dtype=dtype).to(args.device).eval()
    started = time.perf_counter()
    batch_texts, batch_ids = [], []
    id_stream = ids_path.open("a", encoding="utf-8")

    def encode_batch():
        nonlocal processed
        if not batch_texts:
            return
        tokens = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=args.max_length, return_tensors="pt")
        tokens = {key: value.to(args.device) for key, value in tokens.items()}
        with torch.inference_mode():
            dense = model(**tokens).last_hidden_state[:, 0]
            dense = torch.nn.functional.normalize(dense.float(), p=2, dim=-1)
        end = processed + len(batch_texts)
        if end > args.num_nodes or not torch.isfinite(dense).all():
            raise ValueError("invalid encoding output or node count")
        features[processed:end] = dense.cpu().numpy().astype(np.float16)
        id_stream.writelines(node_id + "\n" for node_id in batch_ids)
        id_stream.flush(); features.flush()
        processed = end
        elapsed = float(state.get("elapsed_seconds", 0.0)) + (time.perf_counter() - started)
        save_state(state_path, {"processed": processed, "num_nodes": args.num_nodes,
                                "input_fingerprint": fingerprint,
                                "max_length": args.max_length, "batch_size": args.batch_size,
                                "elapsed_seconds": elapsed})
        batch_texts.clear(); batch_ids.clear()

    with args.nodes.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index < processed:
                continue
            row = json.loads(line)
            batch_ids.append(str(row["id"])); batch_texts.append(str(row["text"]))
            if len(batch_texts) == args.batch_size:
                encode_batch()
                if (processed // args.batch_size) % args.log_every == 0:
                    rate = (processed - int(state["processed"])) / max(time.perf_counter() - started, 1e-9)
                    print(f"encoded {processed:,}/{args.num_nodes:,} nodes ({rate:.1f} nodes/s)", flush=True)
        encode_batch()
    id_stream.close()
    if processed != args.num_nodes:
        raise ValueError(f"encoded {processed} nodes, expected {args.num_nodes}")
    manifest = {
        "model": str(args.model), "model_type": type(model).__name__, "pooling": "normalized_cls",
        "frozen": True, "dtype": "float16", "dimension": 1024,
        "max_length": args.max_length, "batch_size": args.batch_size,
        "num_nodes": processed, "nodes_sha256": file_sha256(args.nodes),
        "model_weights_sha256": file_sha256(args.model / "pytorch_model.bin"),
        "features_sha256": file_sha256(features_path), "node_ids_sha256": file_sha256(ids_path),
        "elapsed_seconds": json.loads(state_path.read_text())["elapsed_seconds"],
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2) if args.device.startswith("cuda") else 0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
