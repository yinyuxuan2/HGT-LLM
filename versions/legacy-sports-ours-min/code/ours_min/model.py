from torch import nn

from .standard_hgt import StandardHGT


class OursMin(nn.Module):
    """BGE-M3 features -> standard HGT -> linear node classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, num_types: int,
                 num_relations: int, num_classes: int, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.2,
                 use_rte: bool = False):
        super().__init__()
        self.hgt = StandardHGT(input_dim, hidden_dim, num_types, num_relations,
                               num_heads, num_layers, dropout, use_rte)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, node_type, edge_index, edge_type, edge_time):
        hidden = self.hgt(features, node_type, edge_index, edge_type, edge_time)
        return self.classifier(hidden)

