"""Structure-only HGT: no text encoder, pretrained weights or text features."""

from torch import nn

from ours_min.standard_hgt import StandardHGTConv


class PureHGTLink(nn.Module):
    def __init__(self, num_nodes, hidden_dim=128, num_heads=4, num_layers=2,
                 num_relations=3, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Global IDs occupy disjoint User, Item, Category ranges.
        self.node_id = nn.Embedding(num_nodes, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            StandardHGTConv(hidden_dim, 3, num_relations, num_heads,
                            dropout=dropout, use_norm=True, use_rte=False)
            for _ in range(num_layers)
        )
        nn.init.normal_(self.node_id.weight, mean=0.0, std=0.02)

    def initial_hidden(self, features, node_type, node_ids):
        # features is an unused one-column placeholder for the shared loader.
        return self.input_dropout(self.node_id(node_ids))

    def forward(self, features, node_type, edge_index, edge_type, edge_time, node_ids):
        hidden = self.initial_hidden(features, node_type, node_ids)
        for layer in self.layers:
            hidden = layer(hidden, node_type, edge_index, edge_type, edge_time)
        return hidden

    @staticmethod
    def score_pairs(hidden, pair_index):
        return (hidden[pair_index[0]] * hidden[pair_index[1]]).sum(-1)
