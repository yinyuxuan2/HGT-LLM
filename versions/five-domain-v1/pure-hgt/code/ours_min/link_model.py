import torch
from torch import nn

from .standard_hgt import StandardHGT


class OursMinLink(nn.Module):
    """Frozen BGE-M3 features, standard HGT, and parameter-free dot product."""

    def __init__(self, input_dim, hidden_dim, num_types=2, num_relations=2,
                 num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.hgt = StandardHGT(input_dim, hidden_dim, num_types, num_relations,
                               num_heads, num_layers, dropout, use_rte=False)

    @staticmethod
    def score_pairs(hidden, pair_index):
        return (hidden[pair_index[0]] * hidden[pair_index[1]]).sum(-1)

    def forward(self, features, node_type, edge_index, edge_type, edge_time):
        return self.hgt(features, node_type, edge_index, edge_type, edge_time)

