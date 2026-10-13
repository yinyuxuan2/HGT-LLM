"""Read-only numeric checks; synthetic values are never experiment results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from scripts.train_hgt_link import ranking_metrics, auxiliary_metrics
from scripts.train_hgt_corrected_v2 import infer_all
from ours_min.pure_hgt import PureHGTLink


def run(device):
    rng = np.random.default_rng(7)
    users, items = 4, 40
    h = torch.tensor(rng.integers(-2, 3, (users + items, 8)), dtype=torch.float32)
    train = np.array([[u, u] for u in range(users)], dtype=np.int64)
    target = np.array([10, 20, 30, 39], dtype=np.int64)
    actual = ranking_metrics(h, train, {"test": target}, users, device, 2)["test"]
    scores = h[:users].numpy() @ h[users:].numpy().T
    scores[np.arange(users), train[:, 1]] = -np.inf
    ranks = [np.where(np.lexsort((np.arange(items), -scores[u])) == target[u])[0][0]+1
             for u in range(users)]
    for k in (10, 20):
        np.testing.assert_allclose(actual[f"recall@{k}"], np.mean(np.array(ranks) <= k))
        expected = np.mean([1/np.log2(r+1) if r <= k else 0 for r in ranks])
        np.testing.assert_allclose(actual[f"ndcg@{k}"], expected, atol=1e-7)
    pos = np.column_stack([np.arange(users), target])
    neg = np.array([[u, i] for u in range(users) for i in range(8)])
    aux = auxiliary_metrics(h, pos, neg, users, device, 5)
    pairs = np.concatenate([pos, neg]); labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    sc = (h[pairs[:, 0]] * h[users+pairs[:, 1]]).sum(1).numpy()
    np.testing.assert_allclose(aux["auc"], roc_auc_score(labels, sc))
    np.testing.assert_allclose(aux["ap"], average_precision_score(labels, sc))
    for bad in (float("nan"), float("inf")):
        corrupt = h.clone(); corrupt[0, 0] = bad
        try:
            ranking_metrics(corrupt, train, {"test": target}, users, device)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite representations accepted")
    torch.manual_seed(42)
    n = 15
    edges = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [3, 3, 4, 0, 1, 9, 9, 10]])
    edges = torch.cat([edges, edges.flip(0)], 1)
    data = Data(x=torch.zeros(n, 1), node_type=torch.arange(n) % 3,
                edge_index=edges, edge_type=torch.arange(edges.shape[1]) % 3,
                edge_time=torch.zeros(edges.shape[1], dtype=torch.long))
    model = PureHGTLink(n, 8, 2, 2, 3, .2).to(device).eval()
    with torch.inference_mode():
        direct = model(data.x.to(device), data.node_type.to(device), data.edge_index.to(device),
                       data.edge_type.to(device), data.edge_time.to(device), torch.arange(n, device=device)).cpu()
        layerwise = infer_all(model, data, device, 4, 0)
    torch.testing.assert_close(direct, layerwise, atol=2e-6, rtol=2e-6)
    print(f"PASS {device}: full-ranking/mask/ties, sklearn AUC/AP, NaN/Inf rejection, full-neighbor inference (including isolates)", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    run("cpu")
    if torch.cuda.is_available():
        run("cuda")
