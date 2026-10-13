"""Fail closed if regenerated server inputs differ from locally frozen v2."""
import argparse
import hashlib
import json
from pathlib import Path
from protocol_array_hashes import array_hashes


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--record", type=Path)
    args = p.parse_args()
    if args.record:
        record = json.loads(args.record.read_text())
        for prefix in ("model", "result"):
            if sha(Path(record[prefix+"_file"])) != record[prefix+"_sha256"]:
                raise ValueError("completed run checksum mismatch")
        print("VERIFIED", args.record)
        return
    expected = json.loads(Path("inputs/expected_protocol_manifest.json").read_text())
    actual = json.loads(Path("protocol/manifest.json").read_text())
    logical_expected = json.loads(Path("inputs/expected_negative_array_hashes.json").read_text())
    if array_hashes(Path("protocol")) != logical_expected:
        raise ValueError("server/local negative arrays differ")
    for name, digest in expected["files"].items():
        # NPZ compressed bytes can differ between zlib versions; array bytes
        # above must match exactly. Every server container is also verified.
        if sha(Path("protocol") / name) != actual["files"][name]:
            raise ValueError("server frozen container checksum mismatch: " + name)
        if not name.endswith(".npz") and sha(Path("protocol") / name) != digest:
            raise ValueError("protocol checksum mismatch: " + name)
    prepared = json.loads(Path("prepared/manifest.json").read_text())
    if sha(Path("prepared/graph_metadata.npz")) != prepared["files"]["graph_metadata.npz"]:
        raise ValueError("graph checksum mismatch")
    for name, feature in (("type-aware", "combined_features.npy"), ("unprompted", "unprompted_combined_features.npy")):
        manifest = json.loads(Path(f"inputs/{name}.json").read_text())
        if sha(Path("../features") / feature) != manifest["features_sha256"]:
            raise ValueError("feature checksum mismatch: " + name)
    print("VERIFIED frozen protocol, corrected graph, both reused feature caches")


if __name__ == "__main__":
    main()
