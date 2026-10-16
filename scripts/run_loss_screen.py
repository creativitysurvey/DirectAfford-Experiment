"""Sequential paired validation screening on a dedicated GPU; never evaluates test data."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

CANDIDATES=[('alignment','alignment',1.0),('bce','bce',0.0),
            ('combined_01','alignment_bce',0.1),('combined_1','alignment_bce',1.0)]


def main(candidates=None,study="loss",candidate_options=None):
    candidates = candidates or [(name,mode,weight,"legacy") for name,mode,weight in CANDIDATES]
    candidate_options = candidate_options or {}
    if set(candidate_options)-{c[0] for c in candidates}:
        raise ValueError('Options refer to unknown candidates')
    for options in candidate_options.values():
        if set(options)-{'lr','train-sampling','train-keep-ratio','ranking-weight','ranking-points'}:
            raise ValueError('Unsupported candidate option')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',default='1')
    parser.add_argument('--data-dir',default='data/laso_dataset')
    parser.add_argument('--output-root',required=True,type=Path)
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.epochs <= 0: parser.error('epochs must be positive')
    repo=Path(__file__).resolve().parents[1]
    data=Path(args.data_dir).resolve()
    sys.path.insert(0,str(repo/"src"))
    from real_data import file_hash
    data_base=data/"LASO_dataset" if (data/"LASO_dataset").is_dir() else data
    data_files=[data_base/f"{kind}_{split}.pkl" for split in ("train","val") for kind in ("anno","objects")]
    data_files.append(data_base/"Affordance-Question.csv")
    data_hashes={str(p):file_hash(p) for p in data_files}
    root=args.output_root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':args.gpu,'PYTHONNOUSERSITE':'1'}
    with open(root/'screen.lock','a') as lock:
        try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('This screen is already running')
        plan={'data_dir':str(data),'epochs':args.epochs,'seed':args.seed,'gpu':args.gpu,
              'data_hashes':data_hashes,'study':study,'candidates':candidates,'candidate_options':candidate_options,
              'selection':'joint','scheduler':'plateau',
              'source_hashes':{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for directory in ('src','scripts') for p in (repo/directory).glob('*.py')}}
        plan=json.loads(json.dumps(plan))
        plan_path=root/'screen_plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise SystemExit('Screen configuration/source changed: choose a new output root')
        plan_path.write_text(json.dumps(plan,indent=2))
        for name,mode,weight,architecture in candidates:
            out=root/name
            out.mkdir(exist_ok=True)
            history_path=out/'history.json'
            if history_path.exists() and json.loads(history_path.read_text())[-1]['epoch'] >= args.epochs:
                print(f'Skipping completed candidate {name}',flush=True)
                continue
            command=[sys.executable,'-u',str(repo/'src/train.py'),'--dataset','laso','--data-dir',str(data),
                     '--device','cuda','--epochs',str(args.epochs),'--seed',str(args.seed),
                     '--batch-size','16','--loss',mode,'--alignment-weight',str(weight),
                     '--bce-weight','1','--architecture',architecture,'--scheduler','plateau','--lr-patience','5',
                     '--selection','joint','--output-dir',str(out)]
            for key,value in candidate_options.get(name,{}).items():
                command += ['--'+key,str(value)]
            if (out/'last.pth').exists(): command+=['--resume',str(out/'last.pth')]
            print(json.dumps({'event':'launch','candidate':name,'command':command}),flush=True)
            with open(out/'train.log','a') as log:
                subprocess.run(command,cwd=repo,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
        summaries=[]
        for name,_,_,architecture in candidates:
            rows=json.loads((root/name/'history.json').read_text())
            def joint(r):
                return max(0,0.871-r['auc'])/0.871+max(0,r['mae']-0.089)/0.089 if r['auc'] is not None else float('inf')
            selected=min(rows,key=joint)
            summaries.append({'candidate':name,'architecture':architecture,'selected_epoch':selected['epoch'],
                              'auc':selected['auc'],'macro_auc':selected.get('macro_auc'),
                              'mae':selected['mae'],'joint_shortfall':joint(selected),
                              'options':candidate_options.get(name,{}),
                              'both_validation_targets_met':selected['auc'] is not None and selected['auc']>0.871 and selected['mae']<0.089,
                              'checkpoint':str(root/name/'best_joint.pth')})
        (root/'screen_summary.json').write_text(json.dumps(summaries,indent=2))
        print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__': main()
