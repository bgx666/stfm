import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def update_params(model, para_names, flag=True):
    paras = []
    for cur_name in para_names:
        for name, p in model.named_parameters():
            if cur_name in name:
                p.requires_grad = flag
                paras.append(p)
    return paras


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def setgpu(gpus):
    if gpus == 'all':
        gpus = '0,1,2,3'
    print('using gpu ' + gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))


def one_hot_embedding(labels, num_classes=10):
    y = torch.eye(num_classes)
    return y[labels]
