"""Aggregate only complete, hash-verified v3 runs; sample standard deviation."""
import argparse
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_hgt_link import sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    result={'revision':'rank-supervision-edge-mask-v3','status':'complete',
            'scope':'frozen five-domain protocol v2; 3 relation graph; paired BPR and sampled edge masking',
            'methods':{}}
    for method in ('type-aware-prompt-hgt','pure-hgt'):
        records=[]
        for seed in (42,43,44):
            folder=args.root/method/f'seed-{seed}'
            record=json.loads((folder/'experiment_record.json').read_text())
            assert record['revision']==result['revision'] and record['seed']==seed
            assert record['experiment']==method
            assert sha256(folder/'best_model.pt')==record['model_sha256']
            assert sha256(folder/'ranking_metrics.json')==record['result_sha256']
            assert json.loads((folder/'ranking_metrics.json').read_text())['test']==record['test']
            records.append(record)
        summary={}
        for key in records[0]['test']:
            values=[r['test'][key] for r in records]
            summary[key]={'mean':statistics.mean(values),'sample_std':statistics.stdev(values)}
        result['methods'][method]={'seeds':[42,43,44],'test':summary,
                                   'records':records}
    destination=args.root/'results_summary.json'
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],
                      'test':{k:v['test'] for k,v in result['methods'].items()}},indent=2))


if __name__=='__main__':main()
