"""Fixed 0.05/0.10/0.20 ranking-strength comparison; no test evaluation."""
from run_loss_screen import main

CANDIDATES=[(name,'bce',0.0,'query_tokens') for name in ('rank005','rank010','rank020')]
OPTIONS={name:{'lr':1e-4,'train-sampling':'epoch','train-keep-ratio':0.85,
               'ranking-weight':weight,'ranking-points':64}
         for name,weight in [('rank005',0.05),('rank010',0.10),('rank020',0.20)]}

if __name__=='__main__':
    main(candidates=CANDIDATES,study='ranking_strength',candidate_options=OPTIONS)
