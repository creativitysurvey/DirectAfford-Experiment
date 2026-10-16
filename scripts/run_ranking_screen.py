"""BCE versus small within-query ranking weights; frozen validation protocol."""
from run_loss_screen import main

CANDIDATES=[(name,'bce',0.0,'query_tokens') for name in ('bce_control','rank002','rank005')]
OPTIONS={name:{'lr':1e-4,'train-sampling':'epoch','train-keep-ratio':0.85,
               'ranking-weight':weight,'ranking-points':64}
         for name,weight in [('bce_control',0.0),('rank002',0.02),('rank005',0.05)]}

if __name__=='__main__':
    main(candidates=CANDIDATES,study='ranking',candidate_options=OPTIONS)
