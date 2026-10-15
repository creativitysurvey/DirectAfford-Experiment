"""Paired pooled-query versus token-query experiment. No test selection."""
from run_loss_screen import main

if __name__ == "__main__":
    main(candidates=[('query_local_bce','bce',0.0,'query_local'),
                     ('query_tokens_bce','bce',0.0,'query_tokens')],study='text_tokens')
