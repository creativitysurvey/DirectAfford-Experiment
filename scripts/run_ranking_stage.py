"""Validation calibration of prior best AUC, then paired BCE/ranking training.
No test split is read. Completed experiments are not modified.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-screen',required=True,type=Path)
    parser.add_argument('--data-dir',default='data/laso_dataset',type=Path)
    parser.add_argument('--output-root',required=True,type=Path)
    parser.add_argument('--gpu',default='1')
    parser.add_argument('--epochs',default=50,type=int)
    parser.add_argument('--seed',default=42,type=int)
    args=parser.parse_args()
    if args.epochs<=0:parser.error('epochs must be positive')
    repo=Path(__file__).resolve().parents[1]
    source,root,data=args.source_screen.resolve(),args.output_root.resolve(),args.data_dir.resolve()
    if root==source or source in root.parents or root in source.parents:
        raise ValueError('Choose an output root separate from the source experiment')
    summary=source/'screen_summary.json'
    if not summary.is_file():raise ValueError('Source sampling screen must be complete')
    checkpoint=source/'epoch85_lr1e-4/best_auc.pth'
    base=data/'LASO_dataset' if (data/'LASO_dataset').is_dir() else data
    inputs=[base/f'{kind}_{split}.pkl' for kind in ('anno','objects') for split in ('train','val')]
    inputs.append(base/'Affordance-Question.csv')
    plan={'source_screen':str(source),'summary_sha256':digest(summary),
          'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
          'data_hashes':{str(p):digest(p) for p in inputs},'epochs':args.epochs,'seed':args.seed,'gpu':args.gpu,
          'source_hashes':{str(p.relative_to(repo)):digest(p) for d in ('src','scripts') for p in (repo/d).glob('*.py')}}
    root.mkdir(parents=True,exist_ok=True)
    environment={**os.environ,'PYTHONNOUSERSITE':'1','CUDA_VISIBLE_DEVICES':args.gpu}
    with open(root/'stage.lock','a') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Ranking stage already running')
        manifest=root/'plan.json'
        if manifest.exists() and json.loads(manifest.read_text())!=plan:
            raise ValueError('Inputs/configuration/source changed; use a new root')
        manifest.write_text(json.dumps(plan,indent=2))
        def status(state,step):
            temp=root/'status.json.tmp';temp.write_text(json.dumps({'status':state,'step':step},indent=2))
            os.replace(temp,root/'status.json')
        def launch(step,command):
            print(json.dumps({'event':'launch','step':step,'command':command}),flush=True)
            with open(root/f'{step}.log','a') as log:
                subprocess.run(command,cwd=repo,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        step='start'
        try:
            step='calibration_export';status('running',step)
            export=root/'validation_export'
            if not (export/'metrics.json').exists():
                if export.exists() and any(export.iterdir()):
                    raise ValueError('Partial validation export: inspect log and use a new root')
                launch(step,[sys.executable,'-u',str(repo/'src/evaluate.py'),'--checkpoint',str(checkpoint),
                             '--data-dir',str(data),'--split','val','--audit-queries','--device','cuda',
                             '--output-dir',str(export)])
            step='calibration_fit';status('running',step)
            calibration=root/'calibration'
            if not (calibration/'diagnostic.json').exists():
                launch(step,[sys.executable,'-u',str(repo/'scripts/diagnose_calibration.py'),
                             '--predictions-dir',str(export),'--output-dir',str(calibration),
                             '--objective','mae','--seed',str(args.seed)])
            step='ranking_screen';status('running',step)
            if not (root/'screen/screen_summary.json').exists():
                launch(step,[sys.executable,'-u',str(repo/'scripts/run_ranking_screen.py'),
                             '--data-dir',str(data),'--output-root',str(root/'screen'),
                             '--gpu',args.gpu,'--epochs',str(args.epochs),'--seed',str(args.seed)])
            report={'uncalibrated_validation':json.loads((export/'metrics.json').read_text()),
                    'calibration_diagnostic':json.loads((calibration/'diagnostic.json').read_text()),
                    'ranking_screen':json.loads((root/'screen/screen_summary.json').read_text()),
                    'protocol':'Existing pooled AUC/MAE joint selection; selected-epoch macro AUC also reported.',
                    'limitation':'Validation development only. Calibration check is exploratory because checkpoint selection used full val. No independent test claim.'}
            (root/'summary.json').write_text(json.dumps(report,indent=2))
            status('complete','complete')
            print(json.dumps(report,indent=2),flush=True)
        except BaseException:
            status('failed',step);raise


if __name__=='__main__':main()
