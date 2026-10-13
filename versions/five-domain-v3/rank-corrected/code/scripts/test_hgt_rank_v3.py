"""Small integrity checks for paired sampling and per-batch edge masking."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_hgt_rank_v3 import paired_seed_edges, validate_pairs, mask_current_positive_edges


def run(device):
    train=np.array([[0,0],[0,1],[1,1],[1,2]],dtype=np.int64)
    negative=np.array([[0,2],[0,2],[1,0],[1,0]],dtype=np.int64)
    permutation=np.array([2,0,3,1])
    source=torch.tensor([0,0,1,1,3,4,4,5,3,4,5,6])
    target=torch.tensor([3,4,4,5,0,0,1,1,6,6,6,3])
    edge_type=torch.tensor([0]*8+[1]*4)
    graph=Data(x=torch.zeros(7,1),edge_index=torch.stack((source,target)),
               edge_type=edge_type,edge_time=torch.zeros(len(source),dtype=torch.long),
               node_type=torch.tensor([0,0,0,1,1,1,2]))
    edges=paired_seed_edges(train,negative,permutation,3)
    loader=LinkNeighborLoader(graph,num_neighbors=[-1,-1],edge_label_index=edges,
                              batch_size=4,shuffle=False)
    removed_total=0
    for batch in loader:
        rows=validate_pairs(batch,train,negative,permutation,3)
        assert len(rows)==2
        before=batch.edge_index.clone();types=batch.edge_type.clone()
        batch=batch.to(device)
        removed=mask_current_positive_edges(batch)
        removed_total+=removed
        assert removed>0
        assert (batch.edge_type==1).sum().item()==(types==1).sum().item()
        assert batch.edge_index.shape[1]==before.shape[1]-removed
        logits=torch.tensor([1.2,0.3,0.5,1.1],device=device,requires_grad=True)
        loss=F.softplus(logits[1::2]-logits[0::2]).mean()
        expected=(F.softplus(torch.tensor(-.9))+F.softplus(torch.tensor(.6)))/2
        torch.testing.assert_close(loss.cpu(),expected)
        loss.backward()
        assert logits.grad[0]<0 and logits.grad[1]>0
    assert removed_total>0
    # Batch edge labels must not be reordered; fail closed if that happens.
    bad=next(iter(LinkNeighborLoader(graph,num_neighbors=[-1],edge_label_index=edges,
                                     batch_size=4,shuffle=False)))
    bad.input_id=bad.input_id.flip(0)
    try: validate_pairs(bad,train,negative,permutation,3)
    except ValueError: pass
    else: raise AssertionError('scrambled supervision order was accepted')
    print(f'PASS {device}: paired frozen negatives, purchase edge mask, preserved other edges, BPR gradient')


if __name__=='__main__':
    torch.set_num_threads(2)
    run('cpu')
    if torch.cuda.is_available(): run('cuda')
