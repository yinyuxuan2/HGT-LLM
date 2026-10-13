"""Unit tests for edge probabilities, freezing, ablation and exact inference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch_geometric.data import Data
from type_aware.cross_attention import SemanticStructureHGTConv, SemanticStructureHGTLink
from ours_min.standard_hgt import StandardHGTConv
from scripts.train_hgt_cross_attention_v2 import infer_all


def run(device):
    torch.manual_seed(1)
    n, dim, semantic_dim = 12, 8, 16
    edge = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8], [4, 4, 4, 5, 5, 0, 0, 0, 0]], device=device)
    x = torch.randn(n, dim, device=device, requires_grad=True)
    semantic = torch.randn(n, semantic_dim, device=device, requires_grad=True)
    types = torch.arange(n, device=device) % 3
    relations = torch.arange(edge.size(1), device=device) % 3
    times = torch.zeros(edge.size(1), device=device, dtype=torch.long)
    layer = SemanticStructureHGTConv(dim, 3, 3, 2, dropout=0., semantic_dim=semantic_dim,
                                      max_padded_edges=3).to(device)
    logits = torch.randn(edge.size(1), 2, device=device)
    cross = layer.cross_weights(x, edge, semantic)
    mixed = layer.attention_weights(logits, edge[1], x, edge, semantic)
    structural = StandardHGTConv.attention_weights(layer, logits, edge[1], x, edge)
    torch.testing.assert_close(mixed, .5*cross+.5*structural)
    for target in edge[1].unique():
        mask = edge[1] == target
        torch.testing.assert_close(cross[mask].sum(0), torch.ones(2, device=device))
        torch.testing.assert_close(mixed[mask].sum(0), torch.ones(2, device=device))
        _, reference = layer.cross(x[target][None,None], semantic.detach()[edge[0,mask]][None],
            torch.zeros(1, int(mask.sum()), dim, device=device), need_weights=True, average_attn_weights=False)
        torch.testing.assert_close(cross[mask], reference[0,:,0].T)
    changed = layer.cross_weights(x, edge, semantic.flip(0))
    assert not torch.allclose(cross, changed)
    permutation = torch.randperm(edge.size(1), device=device)
    torch.testing.assert_close(layer.cross_weights(x, edge[:,permutation], semantic), cross[permutation])
    assert torch.isfinite(layer(x, types, edge, relations, times, semantic)).all()
    baseline = StandardHGTConv(dim, 3, 3, 2, dropout=0.).to(device)
    baseline.load_state_dict({k:v for k,v in layer.state_dict().items() if not k.startswith('cross')})
    layer.cross_enabled = False
    torch.testing.assert_close(layer(x,types,edge,relations,times,semantic), baseline(x,types,edge,relations,times), rtol=0,atol=0)
    layer.cross_enabled = True
    loss = (layer(x, types, edge, relations, times, semantic)*torch.randn_like(x)).sum()
    loss.backward()
    assert semantic.grad is None
    for param in (layer.cross.q_proj_weight, layer.cross.k_proj_weight, layer.cross_logit):
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum()>0
    assert layer.cross.v_proj_weight.grad is None and layer.cross.out_proj.weight.grad is None
    empty = edge[:,:0]
    assert torch.isfinite(layer(x,types,empty,relations[:0],times[:0],semantic)).all()
    model = SemanticStructureHGTLink(semantic_dim, dim, 4, 2, 2, 3, 0., max_padded_edges=3).to(device).eval()
    data = Data(x=semantic.detach().cpu(), node_type=torch.tensor([0]*4+[1]*5+[2]*3),
                edge_index=edge.cpu(), edge_type=relations.cpu(), edge_time=times.cpu())
    with torch.inference_mode():
        direct = model(data.x.to(device), data.node_type.to(device), edge, relations, times, torch.arange(n,device=device)).cpu()
        inferred = infer_all(model, data, device, 3, 0)
    torch.testing.assert_close(direct, inferred, atol=2e-6, rtol=2e-6)
    print('PASS',device,': per-edge MHA reference, normalized convex fusion, semantic sensitivity, edge permutation, frozen inputs, gradients, exact disabled-HGT, empty edges, full-neighbor inference',flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    run('cpu')
    if torch.cuda.is_available():run('cuda')
