import os
import torch
from torch import nn
import pdb
import numpy as np
from PIL import Image
import cv2
import logging

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F


def update_params(model, para_names, flag=True):
    paras = []
    for cur_name in para_names:
        for name,p in model.named_parameters():
            if cur_name in name:
                print("update only", name)
                p.requires_grad = flag
                paras.append(p)
    return paras
    
class FocalLoss(nn.Module):
    def __init__(self, class_num=None, alpha=1.0, gamma=2.0, size_average=True):
        """
        Focal Loss for multi-class classification.
        
        Args:
            class_num: (deprecated, kept for compatibility)
            alpha: Weighting factor (default: 1.0)
            gamma: Focusing parameter, higher gamma down-weights easy examples more (default: 2.0)
            size_average: If True, return mean loss; otherwise return sum (default: True)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) raw logits
            targets: (N,) class indices
        """
        # Use cross_entropy with reduction='none' to get per-sample loss
        # F.cross_entropy internally uses log_softmax with numerical stability
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # (N,)
        
        # p_t = probability of the true class = exp(-ce_loss)
        p_t = torch.exp(-ce_loss)
        
        # Focal term: (1 - p_t)^gamma
        focal_term = torch.pow(1 - p_t, self.gamma)
        
        # Focal loss
        loss = self.alpha * focal_term * ce_loss
        
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
        

def get_metrics(gt, pred, average="binary"):
    acc = accuracy_score(gt, pred)
    pre = precision_score(gt, pred, average=average)
    rec = recall_score(gt, pred, average=average)
    f1 = f1_score(gt, pred, average=average)
    
    return pre, rec, f1, acc


def get_GradeBI(n_class, preds, targets):
    # BIRADS 分类
    formats = n_class
    results = np.zeros((formats, formats)) 
    for pred, target in zip(preds, targets):
        results[target, pred] += 1

    for i in range(formats):
        results[i] = np.round(results[i] / (sum(results[i])) , 4)
    return results


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]


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


def get_Metric(preds, targets):
    # 良恶性分类
    bresults = np.zeros((2, 2)) 
    for pred, target in zip(preds, targets):
        if pred > 1:
            pred = 1
        else:
            pred = 0
        if target > 1:
            target = 1
        else:
            target = 0
        
        bresults[target, pred] += 1
    for i in range(2):
        bresults[i] = np.round(bresults[i] / (sum(bresults[i])) , 4)

    # BIRADS 分类
    formats = max(targets) + 1
    results = np.zeros((formats, formats)) 
    for pred, target in zip(preds, targets):
        results[target, pred] += 1
    
    T = 0
    ALL = 0
    for i in range(formats):
        results[i] = np.round(results[i] / (sum(results[i])) , 4)
        T += results[i,i]
        ALL += sum(results[i])
    if formats <=2:
        errors = 0
    else:
        errors = sum(results[0,2:])+sum(results[1,2:])+sum(results[2,:2])+sum(results[3,:2])+sum(results[4,:2])
    return bresults, results, errors



def save_fig(fake, file_path):
#     plt.imshow(fake.detach().cpu().numpy()[0].transpose([1,2,0]))
    # plt.imsave(file_path, fake.detach().cpu().numpy()[0].transpose([1,2,0]))

    fake = fake.detach().cpu().numpy()[0].transpose([1,2,0])
    fake_t = np.copy(fake)
    fake_t[:,:,0] = fake[:,:,2]
    fake_t[:,:,2] = fake[:,:,0]

    cv2.imwrite(file_path, fake_t)
    
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def get_logger(log_path):
    parent_path = os.path.dirname(log_path)  # get parent path
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    logging.basicConfig(level=logging.INFO,
                    filename=log_path,
                    format='%(levelname)s:%(name)s:%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    logger = logging.getLogger()
    logger.addHandler(console)
    return logger