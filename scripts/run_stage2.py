"""Validation calibration diagnostics followed by a paired architecture screen.
No test split is evaluated or used for selection by this pipeline.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-screen',required=True,type=Path)
    parser.add_argument('--output-root',required=True,type=Path)
    parser.add_argument('--data-dir',default='data/laso_dataset')
    parser.add_argument('--gpu',default='1')
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.epochs<=0:parser.error('epochs must be positive')
    repo=Path(__file__).resolve().parents[1]
    source=args.source_screen.resolve();root=args.output_root.resolve();data=Path(args.data_dir).resolve()
    root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(repo/'src'))
    from real_data import file_hash
    checkpoints={name:source/name/filename for name,filename in
                 [('bce','best_joint.pth'),('combined_01','best_joint.pth'),('combined_1','best_auc.pth')]}
    data_base=data/'LASO_dataset' if (data/'LASO_dataset').is_dir() else data
    files=[data_base/f'{kind}_{split}.pkl' for split in ('train','val') for kind in ('anno','objects')]
    files.append(data_base/'Affordance-Question.csv')
    environment={**os.environ,'PYTHONNOUSERSITE':'1','CUDA_VISIBLE_DEVICES':args.gpu}
    with open(root/'stage2.lock','a') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Stage 2 already running for this root')
        plan={'source_screen':str(source),'data_dir':str(data),'gpu':args.gpu,'epochs':args.epochs,'seed':args.seed,
              'checkpoint_hashes':{name:file_hash(p) for name,p in checkpoints.items()},
              'data_hashes':{str(p):file_hash(p) for p in files},
              'source_hashes':{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for directory in ('src','scripts') for p in (repo/directory).glob('*.py')}}
        manifest=root/'stage2_plan.json'
        if manifest.exists() and json.loads(manifest.read_text())!=plan:
            raise SystemExit('Input/code/configuration changed; choose a new output root')
        manifest.write_text(json.dumps(plan,indent=2))
        if (root/'stage2_summary.json').exists() and (root/'status.json').exists() and json.loads((root/'status.json').read_text())['status']=='complete':
            print('Stage 2 already complete for this root',flush=True)
            return
        def status(state,step):
            temporary=root/'status.json.tmp'
            temporary.write_text(json.dumps({'status':state,'step':step},indent=2))
            os.replace(temporary,root/'status.json')
        def launch(command,log):
            print(json.dumps({'event':'launch','command':command,'log':str(log)}),flush=True)
            with open(log,'a') as handle:
                subprocess.run(command,cwd=repo,env=environment,stdout=handle,stderr=subprocess.STDOUT,check=True)
        step='start'
        try:
            for name,checkpoint in checkpoints.items():
                step=f'validation_diagnostic:{name}';status('running',step)
                base=root/'calibration_diagnostics'/name;base.mkdir(parents=True,exist_ok=True)
                export=base/'validation_export'
                if not (export/'metrics.json').exists():
                    if export.exists() and any(export.iterdir()):
                        raise SystemExit(f'Incomplete export at {export}; inspect it before retrying')
                    launch([sys.executable,'-u',str(repo/'src/evaluate.py'),'--checkpoint',str(checkpoint),
                            '--data-dir',str(data),'--split','val','--audit-queries','--device','cuda',
                            '--output-dir',str(export)],base/'export.log')
                calibrated=base/'calibration'
                if not (calibrated/'diagnostic.json').exists():
                    launch([sys.executable,'-u',str(repo/'scripts/diagnose_calibration.py'),
                            '--predictions-dir',str(export),'--output-dir',str(calibrated),
                            '--objective','mae','--seed',str(args.seed)],base/'fit.log')
            step='architecture_screen';status('running',step)
            launch([sys.executable,'-u',str(repo/'scripts/run_structure_screen.py'),'--gpu',args.gpu,
                    '--data-dir',str(data),'--epochs',str(args.epochs),'--seed',str(args.seed),
                    '--output-root',str(root/'structure')],root/'structure.log')
            report={'structure':json.loads((root/'structure/screen_summary.json').read_text()),
                    'calibration_diagnostics':{name:json.loads((root/'calibration_diagnostics'/name/'calibration/diagnostic.json').read_text()) for name in checkpoints},
                    'limitation':'Validation exploration only; no independent test claim or guarantee of target attainment.'}
            (root/'stage2_summary.json').write_text(json.dumps(report,indent=2))
            status('complete','complete');print(json.dumps(report,indent=2),flush=True)
        except BaseException:
            status('failed',step)
            raise


if __name__=='__main__':main()
