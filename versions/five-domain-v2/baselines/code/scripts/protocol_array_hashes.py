"""Cross-platform semantic hashes in addition to archive SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def array_hashes(root):
    result = {}
    for seed in (42, 43, 44):
        name = f"negatives_seed_{seed}.npz"
        with np.load(root / name) as archive:
            result[name] = {}
            for key in archive.files:
                array = np.ascontiguousarray(archive[key], dtype="<i8")
                result[name][key] = {"shape": list(array.shape), "dtype": "<i8",
                                     "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(array_hashes(args.protocol), indent=2) + "\n")
