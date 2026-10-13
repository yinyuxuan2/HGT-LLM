"""Synthetic correctness checks only; never produces experimental scores."""
import sys
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ours_min.pure_hgt import PureHGTLink


def main():
    torch.manual_seed(42)
    model = PureHGTLink(6, hidden_dim=8, num_heads=2, dropout=0.0)
    ids = torch.arange(6)
    types = torch.tensor([0, 0, 1, 1, 2, 2])
    edges = torch.tensor([[0, 2, 1, 3, 2, 4, 3, 5, 2, 3],
                          [2, 0, 3, 1, 4, 2, 5, 3, 3, 2]])
    relations = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    times = torch.zeros(10, dtype=torch.long)
    # Even non-finite fake text cannot affect this model.
    with patch("numpy.load", side_effect=AssertionError("no feature loading allowed")):
        first = model(torch.zeros(6, 1), types, edges, relations, times, ids)
        second = model(torch.full((6, 1024), float("nan")), types, edges, relations, times, ids)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.isfinite(first).all()
    logits = model.score_pairs(first, torch.tensor([[0, 1], [2, 3]]))
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0.])).backward()
    assert model.node_id.weight.grad is not None
    assert torch.isfinite(model.node_id.weight.grad).all()
    assert model.node_id.weight.grad.abs().sum() > 0
    assert all(layer.relation_att.grad is not None for layer in model.layers)
    # PyG may import optional transformers utilities; no such model is instantiated.
    assert not any(type(module).__module__.startswith(("transformers", "FlagEmbedding"))
                   for module in model.modules())
    print("PASS: ID-only features, no text dependency, finite forward/backward, HGT gradients")


if __name__ == "__main__":
    main()
