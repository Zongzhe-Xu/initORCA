import time, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import operator
from itertools import product
from functools import reduce, partial

# import data loaders, task-specific losses and metrics
from data_loaders import load_imagenet, load_text, load_cifar, load_mnist, load_deepsea, load_darcy_flow, load_psicov, load_music, load_ecg, load_satellite, load_ninapro, load_cosmic, load_spherical, load_fsd, load_modelnet, load_homology, load_listops, load_domainnet
from utils import FocalLoss, LpLoss, conv_init
from utils import mask, accuracy, accuracy_onehot, auroc, psicov_mae, ecg_f1, fnr, map_value, inv_auroc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(root, dataset, batch_size, valid_split, maxsize=None):
    data_kwargs = None

    if dataset == "your_new_task": # modify this to experiment with a new task
        train_loader, val_loader, test_loader = None, None, None
    elif dataset == "MODELNET":
        train_loader, val_loader, test_loader = load_modelnet(root, batch_size, valid_split=valid_split)
    elif dataset == "DOMAINNET":
        train_loader, val_loader, test_loader = load_domainnet(root, batch_size, valid_split=valid_split)
    elif dataset == "IMAGENET":
        train_loader, val_loader, test_loader = load_imagenet(root, batch_size, maxsize=maxsize)
    elif dataset == "text":
        train_loader, val_loader, test_loader = load_text(root, batch_size, maxsize=maxsize)
    elif dataset == "CIFAR10":
        train_loader, val_loader, test_loader = load_cifar(root, 10, batch_size, valid_split=valid_split, maxsize=maxsize)
    elif dataset == "CIFAR10-PERM":
        train_loader, val_loader, test_loader = load_cifar(root, 10, batch_size, permute=True, valid_split=valid_split, maxsize=maxsize)
    elif dataset == "CIFAR100":
        train_loader, val_loader, test_loader = load_cifar(root, 100, batch_size, valid_split=valid_split, maxsize=maxsize)
    elif dataset == "CIFAR100-PERM":
        train_loader, val_loader, test_loader = load_cifar(root, 100, batch_size, permute=True, valid_split=valid_split, maxsize=maxsize)
    elif dataset == "MNIST":
        train_loader, val_loader, test_loader = load_mnist(root, batch_size, valid_split=valid_split)
    elif dataset == "MNIST-PERM":
        train_loader, val_loader, test_loader = load_mnist(root, batch_size, permute=True, valid_split=valid_split)
    elif dataset == "SPHERICAL":
        train_loader, val_loader, test_loader = load_spherical(root, batch_size, valid_split=valid_split, maxsize=maxsize)
    elif dataset == "DEEPSEA":
        train_loader, val_loader, test_loader = load_deepsea(root, batch_size, valid_split=valid_split)
    elif dataset == "DARCY-FLOW-5":
        train_loader, val_loader, test_loader, y_normalizer = load_darcy_flow(root, batch_size, sub = 5, valid_split=valid_split)
        data_kwargs = {"decoder": y_normalizer}
    elif dataset == 'PSICOV':
        train_loader, val_loader, test_loader, _, _ = load_psicov(root, batch_size, valid_split=valid_split)
    elif dataset[:5] == 'MUSIC':
        if dataset[6] == 'J': length = 255
        elif dataset[6] == 'N': length = 513
        train_loader, val_loader, test_loader = load_music(root, batch_size, dataset[6:], length=length, valid_split=valid_split)
    elif dataset == "ECG":
        train_loader, val_loader, test_loader = load_ecg(root, batch_size, valid_split=valid_split)
    elif dataset == "SATELLITE":
        train_loader, val_loader, test_loader = load_satellite(root, batch_size, valid_split=valid_split)
    elif dataset == "NINAPRO":
        train_loader, val_loader, test_loader = load_ninapro(root, batch_size, valid_split=valid_split, maxsize=maxsize)
    elif dataset == "COSMIC":
        train_loader, val_loader, test_loader = load_cosmic(root, batch_size, valid_split=valid_split)
        data_kwargs = {'transform': mask}
    elif dataset == "FSD":
        train_loader, val_loader, test_loader = load_fsd(root, batch_size, valid_split=valid_split)
    elif dataset == "HOMOLOGY":
        train_loader, val_loader, test_loader = load_homology(root, batch_size, valid_split=valid_split)
    elif dataset == "LISTOPS":
        train_loader, val_loader, test_loader = load_listops(root, batch_size, valid_split=valid_split)

    n_train, n_val, n_test = len(train_loader), len(val_loader) if val_loader is not None else 0, len(test_loader)

    if not valid_split:
        val_loader = test_loader
        n_val = n_test

    return train_loader, val_loader, test_loader, n_train, n_val, n_test, data_kwargs


def get_config(dataset):
    config_kwargs = {'activation': None, 'remain_shape': False, 'embed_dim': 16, 'stride': 4, 'channel_matching': None, 'drop_out': 0}
    
    if dataset == "your_new_task": # modify this to experiment with a new task
        dims, num_classes = None, None
        loss = None

    elif dataset == "MODELNET":
        dims, sample_shape, num_classes = 1, (1, 3, 1024), 40
        loss = nn.CrossEntropyLoss()

    elif dataset == "DOMAINNET":
        dims, sample_shape, num_classes = 1, (1, 3, 224, 224), 40
        loss = nn.CrossEntropyLoss()

    elif dataset[:5] == "CIFAR":
        dims, sample_shape, num_classes = 2,  (1, 3, 32, 32), 10 if dataset in ['CIFAR10', 'CIFAR10-PERM'] else 100
        loss = nn.CrossEntropyLoss()

    elif dataset == 'SPHERICAL':
        dims, sample_shape, num_classes = 2, (1, 3, 60, 60), 100
        loss = nn.CrossEntropyLoss() 

    elif dataset == "DARCY-FLOW-5":
        dims, sample_shape, num_classes = 2, (1, 3, 85, 85), 1
        loss = LpLoss(size_average=False)

    elif dataset == "PSICOV":
        dims, sample_shape, num_classes = 2, (1, 57, 128, 128), 1
        loss = nn.MSELoss(reduction='mean')

    elif dataset == "NINAPRO": 
        dims, sample_shape, num_classes = 2, (1, 1, 16, 52), 18
        loss = FocalLoss(alpha=1)

    elif dataset == "COSMIC":
        dims, sample_shape, num_classes = 2, (1, 1, 128, 128), 1
        loss = nn.MSELoss(reduction='mean')

    elif dataset == 'FSD':
        dims, sample_shape, num_classes = 2, (1, 1, 96, 102), 200
        loss = nn.BCEWithLogitsLoss(pos_weight=10 * torch.ones((200, )))
        
    elif dataset[:5] == "MNIST":
        dims, sample_shape, num_classes = 1, (1, 1, 784), 10
        loss = F.nll_loss

    elif dataset[:5] == "MUSIC":
        if dataset[6] == 'J': length = 255 
        elif dataset[6] == 'N': length = 513
        dims, sample_shape, num_classes = 1, (1, length - 1, 88), 88
        loss = nn.BCELoss() 
    
    elif dataset == "ECG": 
        dims, sample_shape, num_classes = 1, (1, 1, 1000), 4
        loss = nn.CrossEntropyLoss()   

    elif dataset == "SATELLITE":
        dims, sample_shape, num_classes = 1, (1, 1, 46), 24
        loss = nn.CrossEntropyLoss()

    elif dataset == "DEEPSEA":
        dims, sample_shape, num_classes = 1, (1, 4, 1000), 36
        loss = nn.BCEWithLogitsLoss(pos_weight=4 * torch.ones((36, )))

    elif dataset == "HOMOLOGY":
        dims, sample_shape, num_classes = 1, (1, 29, 1419), 1195
        loss = nn.CrossEntropyLoss()

    elif dataset == "LISTOPS":
        dims, sample_shape, num_classes = 1, (1, 17, 1024), 10
        loss = nn.CrossEntropyLoss()

    return dims, sample_shape, num_classes, loss, config_kwargs


def get_metric(root, dataset):
    if dataset == "your_new_task": # modify this to experiment with a new task
        return accuracy, np.max
    if dataset[:5] == "CIFAR" or dataset[:5] == "MNIST" or dataset == "SATELLITE" or dataset == "SPHERICAL" or dataset == "MODELNET" or dataset == "HOMOLOGY" or dataset == "LISTOPS" or dataset == "DOMAINNET":
        return accuracy, np.max
    if dataset == "DEEPSEA":
        return auroc, np.max
    if dataset == "DARCY-FLOW-5":
        return LpLoss(size_average=True), np.min
    if dataset == 'PSICOV':
        return psicov_mae(root), np.min
    if dataset[:5] == 'MUSIC':
        return nn.BCELoss(), np.min
    if dataset == 'ECG':
        return ecg_f1, np.max
    if dataset == 'NINAPRO':
        return accuracy, np.max #accuracy_onehot
    if dataset == 'COSMIC':
        return inv_auroc, np.min
    if dataset == 'FSD':
        return map_value, np.max


def get_optimizer(name, params):
    if name == 'SGD':
        return partial(torch.optim.SGD, lr=params['lr'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    elif name == 'Adam':
        return partial(torch.optim.Adam, lr=params['lr'], betas=tuple(params['betas']), weight_decay=params['weight_decay'])
    elif name == 'AdamW':
        return partial(torch.optim.AdamW, lr=params['lr'], betas=tuple(params['betas']), weight_decay=params['weight_decay'])


def get_scheduler(name, params, epochs=200, n_train=None):
    if name == 'StepLR':
        sched = params['sched']

        def scheduler(epoch):    
            optim_factor = 0
            for i in range(len(sched)):
                if epoch > sched[len(sched) - 1 - i]:
                    optim_factor = len(sched) - i
                    break
                    
            return math.pow(params['base'], optim_factor)  

        lr_sched_iter = False

    elif name == 'WarmupLR':
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))

            current_decay_steps = total_steps - step
            total_decay_steps = total_steps - warmup_steps
            f = (current_decay_steps / total_decay_steps)

            return f  

    elif name == 'ExpLR':
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))

            current_decay_steps = total_steps - step
            total_decay_steps = total_steps - warmup_steps
            f = (current_decay_steps / total_decay_steps)

            return params['base'] * f  

    elif name == 'SinLR':

        cycles = 0.5
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            # progress after warmup
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1. + math.cos(math.pi * float(cycles) * 2.0 * progress)))

    return scheduler, lr_sched_iter

