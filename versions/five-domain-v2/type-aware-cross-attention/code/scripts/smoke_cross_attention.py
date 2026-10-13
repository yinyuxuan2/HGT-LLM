"""Synthetic train/eval/resume tests for both feature variants, not paper results."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Use a new synthetic test output directory')
    args.output.mkdir(parents=True)
    for name in ('protocol','prepared'):
        shutil.copytree(args.fixtures/name,args.output/name)
    shutil.copyfile(args.fixtures/'features.npy',args.output/'features.npy')
    features=args.output/'features.npy'
    trainer=Path(__file__).with_name('train_hgt_cross_attention_v2.py')
    for variant,feature_variant in (('type-aware-cross-hgt','type-aware-prompt-hgt-multidomain-v1'),
                                    ('unprompted-cross-hgt','unprompted-bge-m3-hgt-multidomain-v1')):
        fm=args.output/(variant+'-features.json')
        fm.write_text(json.dumps({'variant':feature_variant,'features_sha256':hashlib.sha256(features.read_bytes()).hexdigest()}))
        out=args.output/variant
        cmd=[sys.executable,'-B',str(trainer),'--protocol-dir',str(args.output/'protocol'),
             '--prepared-dir',str(args.output/'prepared'),'--features',str(features),'--features-manifest',str(fm),
             '--output-dir',str(out),'--seed','42','--experiment-name',variant,'--hidden-dim','8',
             '--heads','2','--layers','1','--neighbors','2','--batch-size','8',
             '--inference-batch-size','32','--epochs','1','--device','cpu','--cross-pack-budget','32']
        subprocess.run(cmd,check=True)
        record=json.loads((out/'experiment_record.json').read_text())
        assert record['trainer_revision']=='semantic-structure-cross-v1'
        assert record['validation']['ndcg@10']==record['history'][0]['validation']['ndcg@10']
        assert np.load(out/'best_hidden.npy').dtype==np.float32
        old=(out/'ranking_metrics.json').read_bytes()
        subprocess.run(cmd,check=True)
        assert old==(out/'ranking_metrics.json').read_bytes()
        print('PASS synthetic train/eval/FP32/checkpoint resume:',variant,flush=True)


if __name__=='__main__':main()
