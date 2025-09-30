"""
Set training options.
"""
import copy
import os
import pickle

args = dict(
    save=True,
    checkpoint='exp/checkpoint.pth',
    save_name ='exp_test',
    save_path='exp',

    crop = 'tomato2',
    
    src_dataset = {
        'kwargs': {
            'type': 'src_bg_augmented',
        },
        'batch_size' : 64,
        'num_workers' : 4,
    },
    
    tgt_dataset = {
        'kwargs': {
            'type': 'tgt',
        },
        'batch_size' : 64,
        'num_workers' : 4,
    },

    test_dataset = {
        'kwargs': {
            'type': 'tst',
        },
        'batch_size' : 64,
        'num_workers' : 4,
    },

    model = {
        'name': 'adamatch',
        'backbone' : 'resnet18',
        'n_class' : 3,
    },

    n_epochs = 300,
    early_stop = 300,
)

def get_args():
    return copy.deepcopy(args)
