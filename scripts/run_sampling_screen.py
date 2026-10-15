"""2x2 training-sampling/LR ablation. Same token model, BCE and fixed validation."""
from run_loss_screen import main

CANDIDATES = [(name,'bce',0.0,'query_tokens') for name in (
    'fixed_lr1e-4','epoch85_lr1e-4','fixed_lr5e-5','epoch85_lr5e-5')]
OPTIONS = {
    'fixed_lr1e-4': {'lr':1e-4,'train-sampling':'fixed','train-keep-ratio':1.0},
    'epoch85_lr1e-4': {'lr':1e-4,'train-sampling':'epoch','train-keep-ratio':0.85},
    'fixed_lr5e-5': {'lr':5e-5,'train-sampling':'fixed','train-keep-ratio':1.0},
    'epoch85_lr5e-5': {'lr':5e-5,'train-sampling':'epoch','train-keep-ratio':0.85},
}

if __name__ == '__main__':
    main(candidates=CANDIDATES,study='sampling_lr_2x2',candidate_options=OPTIONS)
