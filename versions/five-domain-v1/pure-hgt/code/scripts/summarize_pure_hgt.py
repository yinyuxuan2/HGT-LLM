"""Summarize completed formal seeds, refusing missing or corrupt artifacts."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    records, artifacts = [], {}
    for seed in (42, 43, 44):
        run = args.root / "runs-pure-hgt" / f"seed-{seed}"
        record = json.loads((run / "experiment_record.json").read_text())
        assert record["seed"] == seed and record["uses_language_model"] is False
        for filename, field in (("best_model.pt", "model_sha256"),
                                ("ranking_metrics.json", "result_sha256")):
            assert sha256(run / filename) == record[field], filename
        for path in sorted(run.iterdir()):
            if path.is_file():
                artifacts[str(path.relative_to(args.root))] = sha256(path)
        records.append(record)
    assert all(r["protocol_file_hashes"] == records[0]["protocol_file_hashes"] for r in records)
    metrics = ("recall@10", "recall@20", "ndcg@10", "ndcg@20", "auc", "ap")
    stats = {key: {"mean": statistics.mean(r["test"][key] for r in records),
                   "sample_std": statistics.stdev(r["test"][key] for r in records)} for key in metrics}
    summary = {"experiment": "pure-hgt", "data_version": records[0]["data_version"],
               "test_statistics": stats, "runs": records, "artifact_sha256": artifacts,
               "model_transfer_status": "server_complete_local_verification_pending"}
    (args.root / "pure_hgt_results_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Pure HGT: five-domain Amazon Reviews 2018", "",
             "No language model, text feature, pretrained embedding or distillation.", "",
             "Test metrics: three seeds, mean +/- sample standard deviation (ddof=1).", "",
             "| Metric | Mean | Sample std |", "|---|---:|---:|"]
    lines += [f"| {k} | {v['mean']:.9f} | {v['sample_std']:.9f} |" for k, v in stats.items()]
    lines += ["", "Full frozen candidate ranking; training interactions masked.",
              "Checkpoint selection uses validation NDCG@10 only."]
    (args.root / "PURE_HGT_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
