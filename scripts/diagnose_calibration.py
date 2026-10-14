"""Fit/check calibration by shape on exported raw LASO validation predictions.
This is exploratory: the checkpoint may already have been selected using full val.
It is not an unbiased estimate of independent test accuracy.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from calibration import apply_calibration,fit_calibration,split_groups
from metrics import compute_metrics
from real_data import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-dir',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--objective',choices=['mae','bce'],default='mae')
    args=parser.parse_args()
    metadata=json.loads((args.predictions_dir/'metrics.json').read_text())
    if metadata['split']!='val' or metadata.get('dataset')!='laso' or metadata.get('calibration_sha256'):
        parser.error('Requires raw LASO validation exports; test/calibrated data are forbidden')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    path=args.predictions_dir/'predictions.npz'
    with np.load(path,allow_pickle=False) as data:
        p,y,ids=data['probabilities'],data['masks'],data['shape_ids']
    if p.shape!=y.shape or len(ids)!=len(p): raise ValueError('Export shape mismatch')
    fit,check=split_groups(ids,args.seed)
    parameters=fit_calibration(p[fit],y[fit],args.objective,args.seed)
    parameters.update(source_split='val',checkpoint_sha256=metadata['checkpoint_sha256'],
                      validation_predictions_sha256=file_hash(path),
                      fit_shape_ids=sorted(set(ids[fit].tolist())),check_shape_ids=sorted(set(ids[check].tolist())),
                      purpose='exploratory grouped validation calibration; checkpoint selected on full val')
    report={'checkpoint_sha256':metadata['checkpoint_sha256'],'objective':args.objective,
            'fit_samples':int(fit.sum()),'check_samples':int(check.sum()),
            'check_raw':compute_metrics(p[check],y[check]),
            'check_calibrated':compute_metrics(apply_calibration(p[check],parameters['slope'],parameters['bias']),y[check]),
            'limitation':'Checkpoint selection used validation already; these are exploratory metrics, not independent test evidence.'}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'calibration.json').write_text(json.dumps(parameters,indent=2))
    (args.output_dir/'diagnostic.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
