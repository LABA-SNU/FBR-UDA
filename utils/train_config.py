# utils/train_config.py
"""
Set training options for paper-release.
"""
import copy
from utils.transforms_utils import transform, augmentation

args = dict(
    save=True,
    checkpoint='exp/checkpoint.pth',
    save_name='exp_paper',
    save_path='exp',

    crop='apple',
    src_dataset={
        'kwargs': {
            'type': 'src_bg_augmented',  # or 'src_lab'
            'transform': transform,
        },
        'batch_size': 64,
        'num_workers': 4,
    },

    tgt_dataset={
        'kwargs': {
            'type': 'tgt',
            'transform': transform,
            'augmentation': augmentation,
        },
        'batch_size': 64,
        'num_workers': 4,
    },

    test_dataset={
        'kwargs': {
            'type': 'tst',
            'transform': transform,
        },
        'batch_size': 64,
        'num_workers': 4,
    },

    model={
        'name': 'dann',       # ddc | dcoral | dann | cdan | daln
        'backbone': 'resnet18',
        'n_class': 3,
    },

    n_epochs=300,
    early_stop=25,
)

def get_args():
    return copy.deepcopy(args)
