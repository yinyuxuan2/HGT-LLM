"""Archive correction source, configurations and checksums under EXPERIMENT_ROOT."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = ["scripts/train_hgt_corrected_v2.py", "scripts/train_hgt_link.py",
             "scripts/revise_protocol_v2.py", "scripts/revise_graph_v2.py",
             "scripts/prepare_multidomain_type_aware_graph.py", "scripts/prepare_uniform_prompt_nodes.py",
             "scripts/audit_frozen_experiment.py", "scripts/smoke_corrected_v2.py",
             "scripts/build_type_aware_features.py", "scripts/encode_bge_m3_transformers.py",
             "scripts/package_experiment_corrections.py", "scripts/test_training_evaluation_v2.py", "tests/test_experiment_corrections.py",
             "ours_min/protocol.py", "ours_min/__init__.py", "ours_min/model.py",
             "ours_min/standard_hgt.py", "ours_min/link_model.py", "ours_min/pure_hgt.py",
             "type_aware/__init__.py", "type_aware/link_model.py"]
    for name in files:
        target = args.output / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    for name in ("EXPERIMENT_AUDIT_2026_09_24.md", "corrected_baselines_v2.yaml"):
        shutil.copyfile(root / "config" / name, args.output / name)
    hashes = {str(path.relative_to(args.output)): digest(path)
              for path in sorted((args.output / "code").rglob("*")) if path.is_file()}
    for name in ("EXPERIMENT_AUDIT_2026_09_24.md", "corrected_baselines_v2.yaml",
                 "frozen_v1_audit.json", "frozen_v2_audit.json"):
        hashes[name] = digest(args.output / name)
    (args.output / "correction_bundle_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({"source_files": len(files), "output": str(args.output), "status": "corrections_packaged_not_formally_trained"}))


if __name__ == "__main__":
    main()
