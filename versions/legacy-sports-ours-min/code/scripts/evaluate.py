import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ours_min.evaluation import evaluate_scores


def main():
    parser = argparse.ArgumentParser(description="Unified full-candidate ranking and auxiliary evaluation")
    parser.add_argument("--scores", type=Path, required=True,
                        help="NPZ containing scores[num_users, num_candidate_items]")
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = np.load(args.scores, allow_pickle=False)["scores"]
    metrics = evaluate_scores(scores, args.protocol_dir, args.seed, args.split)
    payload = {"split": args.split, "seed": args.seed, **metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

