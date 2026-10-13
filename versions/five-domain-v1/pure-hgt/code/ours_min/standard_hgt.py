"""Standard HGT layer, matching the equations/code structure of acbull/pyHGT.

This local form uses PyG's public scatter/softmax primitives directly so the
baseline remains runnable with current torch-geometric releases. It does not add
an architectural component to HGT.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import softmax


class RelTemporalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 240):
        super().__init__()
        position = torch.arange(float(max_len)).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        weight = torch.empty(max_len, dim)
        weight[:, 0::2] = torch.sin(position * div) / math.sqrt(dim)
        weight[:, 1::2] = torch.cos(position * div) / math.sqrt(dim)
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=True)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, edge_time: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.embedding(edge_time))


class StandardHGTConv(nn.Module):
    def __init__(self, dim: int, num_types: int, num_relations: int,
                 num_heads: int, dropout: float = 0.2,
                 use_norm: bool = True, use_rte: bool = False):
        super().__init__()
        if dim % num_heads:
            raise ValueError("hidden dimension must be divisible by number of heads")
        self.dim = dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sqrt_dk = math.sqrt(self.head_dim)
        self.use_norm = use_norm
        self.use_rte = use_rte

        self.k_linears = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_types))
        self.q_linears = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_types))
        self.v_linears = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_types))
        self.a_linears = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_types))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(num_types))
        self.relation_pri = nn.Parameter(torch.ones(num_relations, num_heads))
        self.relation_att = nn.Parameter(
            torch.empty(num_relations, num_heads, self.head_dim, self.head_dim))
        self.relation_msg = nn.Parameter(
            torch.empty(num_relations, num_heads, self.head_dim, self.head_dim))
        self.skip = nn.Parameter(torch.ones(num_types))
        self.dropout = nn.Dropout(dropout)
        self.rte = RelTemporalEncoding(dim) if use_rte else None
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, x: torch.Tensor, node_type: torch.Tensor,
                edge_index: torch.Tensor, edge_type: torch.Tensor,
                edge_time: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        edge_count = source.numel()
        attention = x.new_zeros((edge_count, self.num_heads))
        messages = x.new_zeros((edge_count, self.num_heads, self.head_dim))

        for source_type in range(self.num_types):
            source_mask = node_type[source] == source_type
            for target_type in range(self.num_types):
                meta_mask = source_mask & (node_type[target] == target_type)
                for relation in range(self.num_relations):
                    mask = meta_mask & (edge_type == relation)
                    if not torch.any(mask):
                        continue
                    src_x = x[source[mask]]
                    if self.rte is not None:
                        src_x = self.rte(src_x, edge_time[mask])
                    dst_x = x[target[mask]]
                    q = self.q_linears[target_type](dst_x).view(-1, self.num_heads, self.head_dim)
                    k = self.k_linears[source_type](src_x).view(-1, self.num_heads, self.head_dim)
                    v = self.v_linears[source_type](src_x).view(-1, self.num_heads, self.head_dim)
                    k = torch.einsum("ehd,hdf->ehf", k, self.relation_att[relation])
                    v = torch.einsum("ehd,hdf->ehf", v, self.relation_msg[relation])
                    attention[mask] = (q * k).sum(-1) * self.relation_pri[relation] / self.sqrt_dk
                    messages[mask] = v

        weights = softmax(attention, target)
        weighted = (messages * weights.unsqueeze(-1)).reshape(edge_count, self.dim)
        aggregated = x.new_zeros(x.shape)
        aggregated.index_add_(0, target, weighted)
        aggregated = F.gelu(aggregated)

        output = x.new_zeros(x.shape)
        for target_type in range(self.num_types):
            mask = node_type == target_type
            if not torch.any(mask):
                continue
            transformed = self.dropout(self.a_linears[target_type](aggregated[mask]))
            alpha = torch.sigmoid(self.skip[target_type])
            value = transformed * alpha + x[mask] * (1.0 - alpha)
            output[mask] = self.norms[target_type](value) if self.use_norm else value
        return output


class StandardHGT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_types: int,
                 num_relations: int, num_heads: int, num_layers: int,
                 dropout: float = 0.2, use_rte: bool = False):
        super().__init__()
        self.type_adapters = nn.ModuleList(
            nn.Linear(input_dim, hidden_dim) for _ in range(num_types))
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            StandardHGTConv(hidden_dim, num_types, num_relations, num_heads,
                            dropout=dropout, use_norm=True, use_rte=use_rte)
            for _ in range(num_layers)
        )

    def forward(self, features: torch.Tensor, node_type: torch.Tensor,
                edge_index: torch.Tensor, edge_type: torch.Tensor,
                edge_time: torch.Tensor) -> torch.Tensor:
        hidden = features.new_zeros((features.size(0), self.layers[0].dim))
        for type_id, adapter in enumerate(self.type_adapters):
            mask = node_type == type_id
            if torch.any(mask):
                hidden[mask] = torch.tanh(adapter(features[mask]))
        hidden = self.input_dropout(hidden)
        for layer in self.layers:
            hidden = layer(hidden, node_type, edge_index, edge_type, edge_time)
        return hidden

