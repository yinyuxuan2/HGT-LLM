"""Ranking-aligned HGT correction on the immutable five-domain v2 protocol.

The graph, node features, HGT layers and full-catalog evaluator are imported
from v2. The only training changes are paired BPR supervision and excluding
the current positive purchase edge (both directions) from sampled messages.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch_geometric.loader import LinkNeighborLoader

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from scripts.train_hgt_corrected_v2 import make_graph, infer_all
from scripts.train_hgt_link import load_pairs, ranking_metrics, auxiliary_metrics, sha256
from ours_min.pure_hgt import PureHGTLink
from type_aware.link_model import TypeAwareHGTLink


REVISION = 'rank-supervision-edge-mask-v3'


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')
    tmp.replace(path)


def atomic_torch(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def paired_seed_edges(train, negatives, permutation, num_users):
    """Even columns are positives; immediately following odds are paired negatives."""
    n = len(permutation)
    edges = np.empty((2, n * 2), dtype=np.int64)
    edges[0, 0::2] = train[permutation, 0]
    edges[1, 0::2] = train[permutation, 1] + num_users
    edges[0, 1::2] = negatives[permutation, 0]
    edges[1, 1::2] = negatives[permutation, 1] + num_users
    return torch.from_numpy(edges)


def validate_pairs(batch, train, negatives, permutation, num_users):
    """Fail closed if PyG changes the order of interleaved supervision edges."""
    input_ids = batch.input_id.cpu().numpy()
    if len(input_ids) % 2 or not np.array_equal(input_ids[0::2] + 1, input_ids[1::2]):
        raise ValueError('loader did not preserve contiguous positive/negative pairs')
    rows = permutation[input_ids[0::2] // 2]
    global_edges = batch.n_id[batch.edge_label_index].cpu().numpy()
    if not (np.array_equal(global_edges[0,0::2], train[rows,0]) and
            np.array_equal(global_edges[1,0::2], train[rows,1] + num_users) and
            np.array_equal(global_edges[0,1::2], negatives[rows,0]) and
            np.array_equal(global_edges[1,1::2], negatives[rows,1] + num_users)):
        raise ValueError('frozen negatives no longer match their positive pairs')
    return rows


def mask_current_positive_edges(batch):
    """Drop every sampled purchase edge matching a current positive, in both directions."""
    positive = batch.edge_label_index[:, 0::2]
    if positive.shape[1] == 0:
        raise ValueError('empty positive batch')
    n = batch.num_nodes
    forward = positive[0] * n + positive[1]
    reverse = positive[1] * n + positive[0]
    edge = batch.edge_index
    key = edge[0] * n + edge[1]
    remove = (batch.edge_type == 0) & torch.isin(key, torch.cat([forward, reverse]))
    kept = ~remove
    # Keep all other purchase interactions and all category/co-purchase edges.
    batch.edge_index = edge[:, kept]
    batch.edge_type = batch.edge_type[kept]
    batch.edge_time = batch.edge_time[kept]
    if torch.any((batch.edge_type == 0) & torch.isin(batch.edge_index[0]*n+batch.edge_index[1],
                                                     torch.cat([forward, reverse]))):
        raise AssertionError('supervision purchase edge remained in graph')
    return int(remove.sum().item())


def verify_protocol(args):
    protocol = json.loads((args.protocol_dir / 'manifest.json').read_text())
    prepared = json.loads((args.prepared_dir / 'manifest.json').read_text())
    if protocol['protocol_version'] != 'multidomain-temporal-leave-two-out-v2':
        raise ValueError('requires frozen v2 protocol')
    if prepared['graph_revision'] != 'exact-edge-dedup-v2':
        raise ValueError('requires frozen deduplicated graph')
    for name, expected in protocol['files'].items():
        if sha256(args.protocol_dir/name) != expected:
            raise ValueError(f'protocol checksum mismatch: {name}')
    if sha256(args.prepared_dir/'graph_metadata.npz') != prepared['files']['graph_metadata.npz']:
        raise ValueError('graph metadata checksum mismatch')
    candidates = json.loads((args.protocol_dir/'candidate_items.json').read_text())
    if len(candidates) != protocol['num_items'] or len(set(candidates)) != len(candidates):
        raise ValueError('candidate set does not cover all frozen items')
    if args.experiment_name == 'pure-hgt':
        if args.features is not None:
            raise ValueError('pure HGT cannot receive text features')
        feature_hash = None
    else:
        if args.features is None or args.features_manifest is None:
            raise ValueError('type-aware HGT requires verified frozen BGE features')
        feature_info = json.loads(args.features_manifest.read_text())
        if feature_info['variant'] != 'type-aware-prompt-hgt-multidomain-v1':
            raise ValueError('unexpected BGE feature variant')
        feature_hash = sha256(args.features)
        if feature_hash != feature_info['features_sha256']:
            raise ValueError('BGE feature checksum mismatch')
    return protocol, prepared, feature_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--features-manifest', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--experiment-name', choices=('pure-hgt','type-aware-prompt-hgt'), required=True)
    parser.add_argument('--seed', type=int, choices=(42,43,44), required=True)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=.2)
    parser.add_argument('--neighbors', type=int, nargs='+', default=[15,10])
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--inference-batch-size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight-decay', type=float, default=.0001)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke-batches', type=int, default=0,
                        help='diagnostic only: stop after this many full-data batches')
    args = parser.parse_args()
    if args.batch_size % 2 or args.batch_size < 2 or len(args.neighbors) != args.layers:
        parser.error('even batch size and one neighbor fanout per layer required')
    if args.epochs < 1 or args.patience < 1:
        parser.error('positive epochs and patience required')
    started = time.perf_counter()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    protocol, prepared, feature_hash = verify_protocol(args)
    num_users, num_items = protocol['num_users'], protocol['num_items']
    train = np.atleast_2d(np.loadtxt(args.protocol_dir/'train.csv', delimiter=',',
                                    skiprows=1, usecols=(0,1,2), dtype=np.int64))
    validation, test = [load_pairs(args.protocol_dir/(name+'.csv')) for name in ('validation','test')]
    negatives = np.load(args.protocol_dir/f'negatives_seed_{args.seed}.npz')['train']
    if (len(train) != protocol['num_edges']['train'] or len(negatives) != len(train) or
        not np.array_equal(negatives[:,0], train[:,0]) or np.any(np.diff(train[:,0]) < 0)):
        raise ValueError('training pairs or frozen negatives inconsistent')
    for held in (validation,test):
        if not np.array_equal(held[:,0], np.arange(num_users)):
            raise ValueError('held-out rows must be ordered one per user')
    train_keys = train[:,0]*num_items+train[:,1]
    if (np.intersect1d(train_keys, negatives[:,0]*num_items+negatives[:,1]).size or
        np.intersect1d(train_keys, validation[:,0]*num_items+validation[:,1]).size or
        np.intersect1d(train_keys, test[:,0]*num_items+test[:,1]).size):
        raise ValueError('train/negative/evaluation pairs overlap')
    data = make_graph(args.features, args.prepared_dir/'graph_metadata.npz', train,
                      num_users, num_items, prepared['num_categories'])
    if args.experiment_name == 'pure-hgt':
        model = PureHGTLink(data.num_nodes,args.hidden_dim,args.heads,args.layers,3,args.dropout).to(args.device)
    else:
        model = TypeAwareHGTLink(1024,args.hidden_dim,num_users,args.heads,args.layers,3,args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {str(p.relative_to(SOURCE)):sha256(p) for p in [Path(__file__),
        SOURCE/'scripts/train_hgt_corrected_v2.py', SOURCE/'scripts/train_hgt_link.py',
        SOURCE/'ours_min/standard_hgt.py', SOURCE/'ours_min/pure_hgt.py', SOURCE/'type_aware/link_model.py']}
    config = {'revision':REVISION,'hyperparameters':vars(args),'code_sha256':hashes,
              'protocol_file_hashes':protocol['files'],'graph_sha256':prepared['files']['graph_metadata.npz'],
              'features_sha256':feature_hash,'trainable_parameters':sum(p.numel() for p in model.parameters()),
              'training':{'objective':'paired BPR','supervision_edge_mask':'current positive purchase edge, both directions',
                          'negative_source':'frozen per-seed train array','graph':'original v2 3-relation graph'},
              'evaluation':'original v2 full catalog and auxiliary AUC/AP'}
    serial = json.loads(json.dumps(config, default=str))
    config_path = args.output_dir/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != serial:
        raise ValueError('cannot resume with a changed configuration')
    atomic_json(config_path,serial)
    (args.output_dir/'hyperparameters.yaml').write_text(yaml.safe_dump(serial,sort_keys=False),encoding='utf-8')
    best,stale,history,start_epoch = -1.,0,[],1
    elapsed_prior, peak_prior = 0.,0.
    last = args.output_dir/'last_checkpoint.pt'
    if last.exists():
        saved = torch.load(last,map_location=args.device,weights_only=False)
        if saved['revision'] != REVISION:
            raise ValueError('legacy checkpoint cannot be resumed')
        model.load_state_dict(saved['model_state']); optimizer.load_state_dict(saved['optimizer_state'])
        best,stale,history = saved['best'],saved['stale'],saved['history']
        start_epoch=saved['epoch']+1
        elapsed_prior=saved['elapsed_seconds'];peak_prior=saved['peak_gpu_memory_mb']
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        print(json.dumps({'event':'resumed','seed':args.seed,'epoch':saved['epoch']}),flush=True)
    if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    print(json.dumps({'event':'initialized','seed':args.seed,'experiment':args.experiment_name,
                      'train_rows':len(train),'graph_directed_edges':data.num_edges}),flush=True)
    for epoch in range(start_epoch,args.epochs+1):
        if stale>=args.patience: break
        permutation=np.random.permutation(len(train))
        seeds=paired_seed_edges(train,negatives,permutation,num_users)
        loader=LinkNeighborLoader(data,num_neighbors=args.neighbors,edge_label_index=seeds,
                                  batch_size=args.batch_size,shuffle=False,num_workers=args.workers)
        model.train();total_loss=total_pairs=removed=0
        for batch_number,batch in enumerate(loader,1):
            validate_pairs(batch,train,negatives,permutation,num_users)
            batch=batch.to(args.device)
            removed+=mask_current_positive_edges(batch)
            optimizer.zero_grad(set_to_none=True)
            hidden=model(batch.x.float(),batch.node_type,batch.edge_index,
                         batch.edge_type,batch.edge_time,batch.n_id)
            scores=model.score_pairs(hidden,batch.edge_label_index)
            loss=F.softplus(scores[1::2]-scores[0::2]).mean()
            if not torch.isfinite(loss): raise ValueError('non-finite BPR loss')
            loss.backward();optimizer.step()
            pair_count=len(scores)//2
            total_loss+=loss.item()*pair_count;total_pairs+=pair_count
            if batch_number%250==0:
                print(json.dumps({'event':'batch','seed':args.seed,'epoch':epoch,
                                  'batch':batch_number,'total_batches':len(loader),'removed_edges':removed}),flush=True)
            if args.smoke_batches and batch_number>=args.smoke_batches:
                report={'status':'passed','diagnostic_only':True,'seed':args.seed,
                        'experiment':args.experiment_name,'verified_training_pairs':total_pairs,
                        'removed_supervision_edges':removed,'loss':float(loss.item()),
                        'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/2**20
                        if args.device.startswith('cuda') else 0.}
                atomic_json(args.output_dir/'smoke_report.json',report)
                print(json.dumps(report),flush=True)
                return
        if total_pairs != len(train): raise ValueError('not all frozen training pairs were used')
        hidden_all=infer_all(model,data,args.device,args.inference_batch_size,args.workers)
        valid=ranking_metrics(hidden_all[:num_users+num_items],train,
                              {'validation':validation[:,1]},num_users,args.device)['validation']
        record={'epoch':epoch,'train_loss':total_loss/total_pairs,
                'removed_supervision_edges':removed,'validation':valid}
        history.append(record);print(json.dumps(record),flush=True)
        score=valid['ndcg@10']
        if score>best:
            best,stale=score,0
            atomic_torch(args.output_dir/'best_model.pt',{'model_state':model.state_dict(),
                         'args':vars(args),'trainable_parameters':config['trainable_parameters'],
                         'epoch':epoch,'revision':REVISION})
        else: stale+=1
        saved={'revision':REVISION,'epoch':epoch,'model_state':model.state_dict(),
               'optimizer_state':optimizer.state_dict(),'best':best,'stale':stale,'history':history,
               'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
               'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               'elapsed_seconds':elapsed_prior+time.perf_counter()-started,
               'peak_gpu_memory_mb':max(peak_prior,torch.cuda.max_memory_allocated()/2**20
                                        if args.device.startswith('cuda') else 0.)}
        atomic_torch(last,saved)
        atomic_json(args.output_dir/'progress.json',record)
    selected=torch.load(args.output_dir/'best_model.pt',map_location=args.device,weights_only=False)
    if selected['epoch'] != max(history,key=lambda x:x['validation']['ndcg@10'])['epoch']:
        raise ValueError('best model and validation history disagree')
    model.load_state_dict(selected['model_state'])
    best_hidden=infer_all(model,data,args.device,args.inference_batch_size,args.workers)[:num_users+num_items]
    final=ranking_metrics(best_hidden,train,{'validation':validation[:,1],'test':test[:,1]},
                          num_users,args.device)
    auxiliary=np.load(args.protocol_dir/f'negatives_seed_{args.seed}.npz')['auxiliary_eval']
    if len(auxiliary) != 100*(len(validation)+len(test)):
        raise ValueError('auxiliary negatives mismatch')
    boundary=100*len(validation)
    final['validation'].update(auxiliary_metrics(best_hidden,validation,auxiliary[:boundary],num_users,args.device))
    final['test'].update(auxiliary_metrics(best_hidden,test,auxiliary[boundary:],num_users,args.device))
    result_path=args.output_dir/'ranking_metrics.json';atomic_json(result_path,final)
    model_path=args.output_dir/'best_model.pt'
    result={'revision':REVISION,'experiment':args.experiment_name,'seed':args.seed,
            'best_epoch':selected['epoch'],'history':history,'validation':final['validation'],'test':final['test'],
            'hyperparameters':vars(args),'trainable_parameters':config['trainable_parameters'],
            'model_sha256':sha256(model_path),'result_sha256':sha256(result_path),
            'code_sha256':hashes,'protocol_file_hashes':protocol['files'],
            'graph_sha256':config['graph_sha256'],'features_sha256':feature_hash,
            'training_time_seconds':elapsed_prior+time.perf_counter()-started,
            'peak_gpu_memory_mb':max(peak_prior,torch.cuda.max_memory_allocated()/2**20
                                     if args.device.startswith('cuda') else 0.)}
    atomic_json(args.output_dir/'experiment_record.json',result)
    print(json.dumps({'event':'completed','seed':args.seed,'experiment':args.experiment_name,
                      'test':result['test'],'model_sha256':result['model_sha256']}),flush=True)


if __name__=='__main__': main()
