"""Paired 50-epoch architecture screen; all candidates use pure BCE."""
from run_loss_screen import main

if __name__ == '__main__':
    main(candidates=[('legacy_bce','bce',0.0,'legacy'),
                     ('local_bce','bce',0.0,'local'),
                     ('query_local_bce','bce',0.0,'query_local')],study='structure')
