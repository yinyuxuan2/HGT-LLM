"""Neighborhood semantic-key / structural-query cross attention.

Only attention probabilities are fused. HGT values/messages remain unchanged.
MHA value/output projections are unused and frozen (excluded from trainable count).
No node-wise single-key softmax and no all-graph dense attention are used.
"""
import math
import torch
from torch import nn
from ours_min.standard_hgt import StandardHGTConv
from type_aware.link_model import TypeAwareHGTLink


class SemanticStructureHGTConv(StandardHGTConv):
    def __init__(self, dim, num_types, num_relations, num_heads, dropout=.2,
                 semantic_dim=1024, cross_init=.5, max_padded_edges=8192):
        super().__init__(dim, num_types, num_relations, num_heads, dropout,
                         use_norm=True, use_rte=False)
        if not 0 < cross_init < 1 or semantic_dim == dim or max_padded_edges < 1:
            raise ValueError('Require 0<cross_init<1, different semantic/hidden dims and positive packing budget')
        self.semantic_dim = semantic_dim
        self.max_padded_edges = max_padded_edges
        self.cross_enabled = True  # ablation/testing switch; production config uses True
        self.cross = nn.MultiheadAttention(dim, num_heads, kdim=semantic_dim,
                                           vdim=dim, dropout=0., bias=False, batch_first=True)
        self.cross.v_proj_weight.requires_grad_(False)
        self.cross.out_proj.weight.requires_grad_(False)
        self.cross_logit = nn.Parameter(torch.tensor(math.log(cross_init/(1-cross_init))))

    def cross_weights(self, x, edge_index, semantic):
        if semantic is None or semantic.shape != (x.size(0), self.semantic_dim):
            raise ValueError('Every layer requires unchanged node-aligned raw semantic features')
        source, target = edge_index
        if not source.numel():
            return x.new_empty((0, self.num_heads))
        # Group by destination, then bucket by degree. Chunk destinations, NEVER
        # split or truncate one destination's neighborhood (softmax denominator).
        order = torch.argsort(target, stable=True)
        sorted_target = target[order]
        nodes, counts = torch.unique_consecutive(sorted_target, return_counts=True)
        starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
        degree_order = torch.argsort(counts, stable=True)
        sorted_degrees = counts[degree_order].detach().cpu().tolist()
        result = x.new_zeros((source.numel(), self.num_heads))
        offset = 0
        while offset < len(sorted_degrees):
            width = 1 << (sorted_degrees[offset]-1).bit_length()
            stop = offset
            while stop < len(sorted_degrees) and sorted_degrees[stop] <= width:
                stop += 1
            step = max(1, self.max_padded_edges // width)
            for begin in range(offset, stop, step):
                selected = degree_order[begin:min(begin+step, stop)]
                length = max(sorted_degrees[begin:min(begin+step, stop)])
                positions = torch.arange(length, device=x.device)[None, :]
                valid = positions < counts[selected, None]
                slots = (starts[selected, None] + positions).clamp_max(source.numel()-1)
                edge_ids = order[slots]
                keys = semantic.detach()[source[edge_ids]].to(dtype=x.dtype)
                keys = keys.masked_fill(~valid[..., None], 0.)
                queries = x[nodes[selected]][:, None, :]
                dummy_values = x.new_zeros((selected.numel(), length, self.dim))
                _, probabilities = self.cross(queries, keys, dummy_values,
                    key_padding_mask=~valid, need_weights=True, average_attn_weights=False)
                # [destinations, heads, 1, neighbors] -> original [edges, heads]
                result[edge_ids[valid]] = probabilities[:, :, 0, :].transpose(1, 2)[valid]
            offset = stop
        return result

    def attention_weights(self, logits, target, x, edge_index, semantic=None):
        structural = super().attention_weights(logits, target, x, edge_index)
        if not self.cross_enabled:
            return structural
        cross = self.cross_weights(x, edge_index, semantic)
        mix = self.cross_logit.sigmoid()
        return (1-mix)*structural + mix*cross


class SemanticStructureHGTLink(TypeAwareHGTLink):
    def __init__(self, input_dim, hidden_dim, num_users, num_heads=4,
                 num_layers=2, num_relations=3, dropout=.2,
                 cross_init=.5, max_padded_edges=8192):
        super().__init__(input_dim, hidden_dim, num_users, num_heads,
                         num_layers, num_relations, dropout)
        layers = []
        for original in self.layers:
            layer = SemanticStructureHGTConv(hidden_dim, 3, num_relations, num_heads,
                      dropout, input_dim, cross_init, max_padded_edges)
            missing, unexpected = layer.load_state_dict(original.state_dict(), strict=False)
            assert not unexpected and all(n.startswith('cross') for n in missing)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, features, node_type, edge_index, edge_type, edge_time, node_ids):
        semantic = features.detach()
        hidden = self.initial_hidden(semantic, node_type, node_ids)
        for layer in self.layers:
            hidden = layer(hidden, node_type, edge_index, edge_type, edge_time, semantic)
        return hidden
