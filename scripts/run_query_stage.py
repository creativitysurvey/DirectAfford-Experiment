"""Query diagnostics, capacity probes and a paired token-decoder screen.

Sequential GPU use, fresh output root, no test access. Reuse completed steps
with identical inputs/source; partial diagnostic steps need inspection/new root.
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
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-stage2',required=True,type=Path)
    parser.add_argument('--data-dir',default='data/laso_dataset',type=Path)
    parser.add_argument('--output-root',required=True,type=Path)
    parser.add_argument('--gpu',default='1')
    parser.add_argument('--epochs',default=50,type=int)
    parser.add_argument('--probe-steps',default=300,type=int)
    parser.add_argument('--seed',default=42,type=int)
    args=parser.parse_args()
    if min(args.epochs,args.probe_steps)<=0: parser.error('Budgets must be positive')
    repo=Path(__file__).resolve().parents[1]
    source,root,data=args.source_stage2.resolve(),args.output_root.resolve(),args.data_dir.resolve()
    if root==source or source in root.parents or root in source.parents:
        raise ValueError('Use a new root separate from stage2')
    if json.loads((source/'status.json').read_text()).get('status')!='complete':
        raise ValueError('Stage2 must be complete')
    checkpoint=source/'structure/query_local_bce/best_joint.pth'
    base=data/'LASO_dataset' if (data/'LASO_dataset').is_dir() else data
    inputs=[base/f'{kind}_{split}.pkl' for kind in ('anno','objects') for split in ('train','val')]
    inputs.append(base/'Affordance-Question.csv')
    plan={'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
          'data_hashes':{str(p):digest(p) for p in inputs},'epochs':args.epochs,
          'probe_steps':args.probe_steps,'seed':args.seed,'gpu':args.gpu,
          'source_hashes':{str(p.relative_to(repo)):digest(p) for d in ('src','scripts') for p in (repo/d).glob('*.py')}}
    root.mkdir(parents=True,exist_ok=True)
    environment={**os.environ,'PYTHONNOUSERSITE':'1','CUDA_VISIBLE_DEVICES':args.gpu}
    with open(root/'stage.lock','a') as lock:
        try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Query stage already running')
        manifest=root/'plan.json'
        if manifest.exists() and json.loads(manifest.read_text())!=plan:
            raise ValueError('Configuration/source/inputs changed; use a new output root')
        manifest.write_text(json.dumps(plan,indent=2))
        def status(state,step):
            temp=root/'status.json.tmp';temp.write_text(json.dumps({'status':state,'step':step},indent=2))
            os.replace(temp,root/'status.json')
        def launch(name,command):
            print(json.dumps({'event':'launch','step':name,'command':command}),flush=True)
            with open(root/f'{name}.log','a') as log:
                subprocess.run(command,cwd=repo,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        step='start'
        try:
            diagnostics={}
            for name,mode,architecture in [('same_object','queries','query_local'),
                                           ('overfit_pooled','overfit','query_local'),
                                           ('overfit_tokens','overfit','query_tokens')]:
                step=name;status('running',step);out=root/name
                if not (out/'diagnostic.json').exists():
                    if out.exists() and any(out.iterdir()):
                        raise ValueError(f'Partial diagnostic {out}; inspect log and use new root')
                    command=[sys.executable,'-u',str(repo/'scripts/diagnose_queries.py'),'--mode',mode,
                             '--data-dir',str(data),'--architecture',architecture,'--seed',str(args.seed),
                             '--steps',str(args.probe_steps),'--max-shapes','128' if mode=='queries' else '8',
                             '--output-dir',str(out)]
                    if mode=='queries':command+=['--checkpoint',str(checkpoint)]
                    launch(name,command)
                diagnostics[name]=json.loads((out/'diagnostic.json').read_text())
            # Probes diagnose capacity; no arbitrary probe threshold decides full-run selection.
            step='token_screen';status('running',step)
            if not (root/'screen/screen_summary.json').exists():
                launch(step,[sys.executable,'-u',str(repo/'scripts/run_token_screen.py'),'--gpu',args.gpu,
                             '--data-dir',str(data),'--epochs',str(args.epochs),'--seed',str(args.seed),
                             '--output-root',str(root/'screen')])
            report={'diagnostics':diagnostics,'screen':json.loads((root/'screen/screen_summary.json').read_text()),
                    'limitation':'Validation development only. Probe scores are training-subset diagnostics; no test or accuracy guarantee.'}
            (root/'summary.json').write_text(json.dumps(report,indent=2))
            status('complete','complete')
            print(json.dumps({'event':'complete','summary':str(root/'summary.json'),'screen':report['screen']},indent=2),flush=True)
        except BaseException:
            status('failed',step);raise


if __name__=='__main__':main()
