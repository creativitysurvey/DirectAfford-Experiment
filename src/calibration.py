"""One global strictly positive logit slope and bias; validation fitting only."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def apply_calibration(probabilities,slope,bias):
    if not np.isfinite(slope) or slope <= 0 or not np.isfinite(bias):
        raise ValueError('Calibration requires a finite positive slope and finite bias')
    p=np.asarray(probabilities,dtype=np.float64)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid probabilities')
    # Preserve order of all interior probabilities. Exact 0/1 remain endpoints.
    result=np.empty_like(p)
    middle=(p>0)&(p<1)
    result[p==0]=0; result[p==1]=1
    result[middle]=expit(slope*(np.log(p[middle])-np.log1p(-p[middle]))+bias)
    return result


def split_groups(shape_ids,seed=42):
    ids=np.asarray(shape_ids).astype(str)
    groups=np.unique(ids)
    if len(groups)<4: raise ValueError('Need at least four shapes for grouped calibration diagnosis')
    groups=np.random.default_rng(seed).permutation(groups)
    fit=np.isin(ids,groups[:len(groups)//2])
    return fit,~fit


def fit_calibration(probabilities,targets,objective='mae',seed=42,max_points=100000):
    p=np.asarray(probabilities,dtype=np.float64).reshape(-1)
    y=np.asarray(targets,dtype=np.float64).reshape(-1)
    if objective not in {'mae','bce'}: raise ValueError('Unknown objective')
    if p.shape!=y.shape or not p.size or not np.isfinite(y).all() or ((y<0)|(y>1)).any():
        raise ValueError('Invalid targets')
    apply_calibration(p,1,0)
    if p.size>max_points:
        indices=np.random.default_rng(seed).choice(len(p),max_points,replace=False)
        p,y=p[indices],y[indices]
    def loss(parameters):
        q=apply_calibration(p,float(np.exp(parameters[0])),float(parameters[1]))
        if objective=='mae': return float(np.abs(q-y).mean())
        q=np.clip(q,1e-12,1-1e-12)
        return float(-(y*np.log(q)+(1-y)*np.log1p(-q)).mean())
    solution=minimize(loss,[0.,0.],method='Powell',bounds=[(-3.,3.),(-10.,10.)],options={'maxiter':200})
    if not solution.success or not np.isfinite(solution.fun):
        raise RuntimeError(f'Calibration fit failed: {solution.message}')
    # Identity is always eligible; optimization cannot worsen fitting objective.
    parameters=solution.x if solution.fun<loss([0,0]) else np.array([0.,0.])
    return {'slope':float(np.exp(parameters[0])),'bias':float(parameters[1]),
            'fit_objective':loss(parameters),'identity_objective':loss([0,0]),
            'objective':objective,'fit_points':len(p),'seed':seed}
