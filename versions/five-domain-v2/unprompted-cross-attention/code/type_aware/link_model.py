"""Type-aware prompt HGT link predictor."""

import torch
from torch import nn

from ours_min.standard_hgt import StandardHGTConv


class TypeAwareHGTLink(nn.Module):
    """Frozen Item/Category text features plus User ID and behavior features."""

    def __init__(self, input_dim, hidden_dim, num_users, num_heads=4,
                 num_layers=2, num_relations=3, dropout=0.2):
        super().__init__()
        self.num_users = num_users
        self.hidden_dim = hidden_dim
        self.type_adapters = nn.ModuleList(nn.Linear(input_dim, hidden_dim) for _ in range(3))
        self.user_id = nn.Embedding(num_users, hidden_dim)
        self.user_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            StandardHGTConv(hidden_dim, 3, num_relations, num_heads,
                            dropout=dropout, use_norm=True, use_rte=False)
            for _ in range(num_layers)
        )
        nn.init.normal_(self.user_id.weight, std=0.02)

    def initial_hidden(self, features, node_type, node_ids):
        hidden = features.new_zeros((features.size(0), self.hidden_dim))
        for type_id, adapter in enumerate(self.type_adapters):
            mask = node_type == type_id
            if torch.any(mask):
                hidden[mask] = torch.tanh(adapter(features[mask]))
        user_mask = node_type == 0
        if torch.any(user_mask):
            global_user_ids = node_ids[user_mask]
            hidden[user_mask] = self.user_norm(hidden[user_mask] + self.user_id(global_user_ids))
        return self.input_dropout(hidden)

    def forward(self, features, node_type, edge_index, edge_type, edge_time, node_ids):
        hidden = self.initial_hidden(features, node_type, node_ids)
        for layer in self.layers:
            hidden = layer(hidden, node_type, edge_index, edge_type, edge_time)
        return hidden

    @staticmethod
    def score_pairs(hidden, pair_index):
        return (hidden[pair_index[0]] * hidden[pair_index[1]]).sum(-1)
