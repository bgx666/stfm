import argparse
import os
import time
import numpy as np
import sklearn.metrics as skmetrics
import sys
import logging

import torch

from torch.nn import DataParallel, Linear
from torch.backends import cudnn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_utils.dataloader import EchoSelectdata, EchoSelectSegdata, Echo2DdataTest
from utils import *
from losses import *
from models_enhanced import create_enhanced_stfm

###########################################################################
"""
        The main function of AI + EchoVideo Prototype Prediction Framework
                                 Python 3
                               pytorch 2.3.0
                              author: Tao He
                       Institution: Sichuan University
                         email: tao_he@scu.edu.cn
"""
###########################################################################

parser = argparse.ArgumentParser(description='PyTorch Classification for EchoVideo')
parser.add_argument('--model_name',  default='resnet50', type=str)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--start_epoch', default=1,  type=int)
parser.add_argument('-b', '--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.0001, type=float )
parser.add_argument('--resume',  default='',  type=str )
parser.add_argument('--weight_decay', default=0.05, type=float )
parser.add_argument('--save_dir', default='./save', type=str)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--patient', default=20, type=int)
parser.add_argument('--loss_name', default='reedl_loss', type=str)
parser.add_argument('--data_path', default='./data/EchoData/', type=str)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--test_flag', default=0, type=int)  # 0 for training and 1 for testing
parser.add_argument('--n_class', default=9, type=int)
# Enhanced STFM parameters
parser.add_argument('--use_enhanced', default=1, type=int, help='Use Enhanced STFM')
parser.add_argument('--temporal_hidden', default=512, type=int)
parser.add_argument('--temporal_layers', default=2, type=int)
parser.add_argument('--seed', default=666, type=int)
parser.add_argument('--over_sample', default="0_3_3_4_0_4_3_5_5", type=str) # setting in video-level

parser.add_argument('--frameNo', default=32, type=int)
parser.add_argument('--selective', default=1, type=int)  # 1 for True 0 for False
parser.add_argument('--uncertainty', default=1, type=int) # # 1 for True 0 for False
parser.add_argument('--subset_size', default=30, type=int)    # None
parser.add_argument('--clip_length', default=5, type=int)    # None
parser.add_argument('--clip_interval', default=5, type=int)  # clip sampling interval
parser.add_argument('--segment_size', default=20, type=int, help='>0 to use segment-level uncertainty (EchoSelectSegdata)')
parser.add_argument('--test_num_frames', default=10, type=int)
parser.add_argument('--epsilon', default=0.2, type=float, help='Epsilon-greedy exploration rate for selective sampling (0=full exploitation, 1=full random)')
parser.add_argument('--val_interval', default=1, type=int, help='validation interval')
parser.add_argument('--save_latest', default=1, type=int, help='Save model_latest.ckpt every epoch')
parser.add_argument('--eval_phase', default='Test', type=str, choices=['Train', 'Val', 'Test'], help='Dataset split for evaluation')
parser.add_argument('--lamb2', default=0.8, type=float, help='REEDL lambda2 parameter')
parser.add_argument('--fixed_center', default=0, type=int, help='Always pick center frame of segment instead of random')

DEVICE = torch.device("cuda" if True else "cpu")

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    setgpu(args.gpu)

    over_samplestr = args.over_sample.split("_")
    over_sample = [int(i) for i in over_samplestr]
    if len(over_sample)!=args.n_class:
        over_sample = None

    ############################ model testing here ###################################
    model_name_lower = args.model_name.lower()
    if model_name_lower == 'inception_v3':
        input_size = (299,299)
    elif model_name_lower in ['resnet18', 'resnet50', 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
                             'vit_tiny', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32',
                             'efficientnet_v2_s']:
        input_size = (224,224)
    elif model_name_lower in ['efficientnet_v2_m', 'efficientnet_v2_l']:
        input_size = (256,256)
    else:
        input_size = (224,224)

    # STFM
    if args.use_enhanced:
        logging.info("Using Enhanced STFM model")
        net = create_enhanced_stfm(
            num_classes=args.n_class,
            backbone=args.model_name,
            hidden_size=args.temporal_hidden,
            num_layers=args.temporal_layers,
            embed_dims=128
        )
    else:
        from models import STFM
        net = STFM(space_model_name=args.model_name, temporal_model_name="lstm", num_classes=args.n_class)

    if args.test_flag:
        net = net.eval()

    if args.loss_name == "crossentropy":
        loss = torch.nn.CrossEntropyLoss()
    elif args.loss_name == "edl_log":
        loss = edl_log_loss
    elif args.loss_name == "edl_digamma":
        loss = edl_digamma_loss
    elif args.loss_name == "edl_mse":
        loss = edl_mse_loss
    elif args.loss_name == "reedl_loss":
        loss = reedl_loss
    ##########################################################################################

    start_epoch = args.start_epoch
    logging.info(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['state_dict'])

    net = net.to(DEVICE)
    if len(args.gpu.split(',')) > 1 or args.gpu == 'all':
        net = DataParallel(net)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # GPU /
    val_gpu_norm = v2.Compose([
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean, std),
    ]).to(DEVICE)

    ################ testing here ######################

    if args.test_flag:
        test_transform = v2.Compose([
                        v2.Resize(256),
                        v2.CenterCrop(224),
                        v2.ToImage(),
        ])
        test_dataset = Echo2DdataTest(transforms=test_transform,
                                        phase=args.eval_phase,
                                        parent_dir=args.data_path)
        testloader = DataLoader(test_dataset,
                                batch_size=1, # must be 1 for testing
                                shuffle=False,
                                drop_last =False,
                                num_workers=args.num_workers,
                                pin_memory=True)
        test(testloader, net, loss, args.n_class, args.clip_length, args.clip_interval, args.test_num_frames, args.lamb2, gpu_norm=val_gpu_norm)
        return

    ################## training here ####################
    val_transform = v2.Compose([
                    v2.Resize(256),
                    v2.CenterCrop(224),
                    v2.ToImage(),
    ])
    val_dataset = Echo2DdataTest(transforms=val_transform, phase='Val',
            parent_dir=args.data_path)
    valloader = DataLoader(val_dataset,
                    batch_size=1,
                    shuffle=False,
                    drop_last =False,
                    num_workers=args.num_workers,
                    pin_memory=True)

    # Test dataset ()
    test_transform = v2.Compose([
                    v2.Resize(256),
                    v2.CenterCrop(224),
                    v2.ToImage(),
    ])
    test_dataset = Echo2DdataTest(transforms=test_transform,
                                    phase='Test',
                                    parent_dir=args.data_path)
    testloader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            drop_last=False,
                            num_workers=args.num_workers,
                            pin_memory=True)

    # CPU transform (DataLoader  resize)
    cpu_transform = v2.Compose([
                    v2.Resize(256),
                    v2.ToImage(),                # PIL → uint8 (C, H, W)
    ])
    # GPU transform clip v2  T
    gpu_transform = v2.Compose([
                    v2.ToDtype(torch.float32, scale=True),
                    v2.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
                    v2.RandomRotation(degrees=10),
                    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
                    v2.Normalize(mean, std),
    ]).to(DEVICE)
    if args.segment_size > 0:
        train_dataset = EchoSelectSegdata(transforms=cpu_transform, phase='Train', selective=bool(args.selective),
                over_sample=over_sample,
                parent_dir=args.data_path,
                clip_length=args.clip_length,
                clip_interval=args.clip_interval,
                segment_size=args.segment_size,
                fixed_center=bool(args.fixed_center))
    else:
        train_dataset = EchoSelectdata(transforms=cpu_transform, phase='Train', selective=bool(args.selective),
                over_sample=over_sample,
                parent_dir=args.data_path,
                subset_size=args.subset_size,
                clip_length=args.clip_length,
                clip_interval=args.clip_interval)
    trainloader = DataLoader(train_dataset,
             batch_size=args.batch_size,
             shuffle=True,
             num_workers=args.num_workers,
             pin_memory=True,
             persistent_workers=True)

    initloader = DataLoader(train_dataset,
             batch_size=1,

             num_workers=args.num_workers,
             pin_memory=True,
             persistent_workers=True)

    initloader = DataLoader(train_dataset,
             batch_size=1,
             shuffle=False,
             num_workers=12,
             pin_memory=True,
             persistent_workers=True)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=0)
    schedule = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs])
    break_flag = 0.
    max_acc = 0.

    if args.selective:
        pass
        # initialize the uncertainty bank first
        # initloader.dataset.set_init(False)
        # logging.info("###initialize the uncertainty bank first###")
        # init_uncertainty(initloader, net, args.n_class)
        # initloader.dataset.set_init(True)

    # Epoch 1~5: full random exploration; epoch 6+: bank-guided
    if args.segment_size > 0:
        if start_epoch <= 5:
            trainloader.dataset.set_epsilon(1.0)
        else:
            trainloader.dataset.set_epsilon(args.epsilon)

    #  batch
    batch_stats = {
        'batch_idx': [],
        'probs_mean': [],
        'probs_std': [],
        'uncertainty_mean': [],
        'uncertainty_std': []
    }

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == 6 and args.segment_size > 0:
            trainloader.dataset.set_epsilon(args.epsilon)
        cur_acc, break_flag = train(trainloader,
                        valloader, net, loss, epoch, optimizer,
                        args.save_dir,
                        max_acc, args.n_class, bool(args.selective), bool(args.uncertainty),
                        break_flag, batch_stats, args.lamb2, gpu_transform, val_gpu_norm)
        schedule.step()
        if cur_acc > max_acc:
            max_acc = cur_acc
        if break_flag > args.patient:
            logging.info(f'Early stopping at epoch {epoch}')
            break

    # Post-training: load best checkpoint and run test
    best_ckpt = os.path.join(args.save_dir, 'model.ckpt')
    if os.path.exists(best_ckpt):
        logging.info(f'Loading best checkpoint for test: {best_ckpt}')
        ckpt = torch.load(best_ckpt, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', ckpt)
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        net.load_state_dict(sd)
        net = net.to(DEVICE)
        test(testloader, net, loss, args.n_class, args.clip_length, args.clip_interval, args.test_num_frames, args.lamb2, gpu_norm=val_gpu_norm)
        logging.info(f'Test evaluation done.')

def init_uncertainty(initloader, net, n_class, lamb2=0.8):
    start_time = time.time()
    net.eval()

    with torch.no_grad():
        for i, sample in enumerate(initloader):
            data = sample[0] # batch, image_num, channel, w, h
            label_ = sample[1]
            data = data.to(DEVICE)
            data = data.view(data.size(0)*data.size(1), data.size(2), data.size(3), data.size(4))
            y = one_hot_embedding(label_, n_class)
            y = y.to(DEVICE)
            logits = net(data)
            #probs = torch.softmax(logits,1)
            probs = get_expectedprob(logits, label_, lamb2=lamb2)
            cur_uncertainty = get_uncertainty( logits, n_class, lamb2=lamb2)
            inthe_bank = probs[range(data.size(0)),label_] + cur_uncertainty.squeeze(1)
            select_index = sample[2]
            initloader.dataset.init_uncertainty(select_index, inthe_bank.detach().cpu())

def log_metrics(targets, preds, prefix='Metrics'):
    """ metrics"""
    acc = skmetrics.accuracy_score(targets, preds)
    macro_recall = skmetrics.recall_score(targets, preds, average='macro')
    micro_recall = skmetrics.recall_score(targets, preds, average='micro')
    macro_f1 = skmetrics.f1_score(targets, preds, average='macro')
    micro_f1 = skmetrics.f1_score(targets, preds, average='micro')
    confusion = skmetrics.confusion_matrix(targets, preds)

    logging.info(
        '%s --> Accuracy: [%.6f], Macro F1: [%.6f], Micro F1: [%.6f], Macro Recall [%.6f], Micro Recall [%.6f]'
        % (prefix, acc, macro_f1, micro_f1, macro_recall, micro_recall))
    logging.info('%s Confuse matrix =======>', prefix)
    logging.info(confusion)
    return acc

def train(trainloader, valloader, net, loss, epoch, optimizer, save_dir, max_acc, n_class, selective, uncertainty, break_flag=0, batch_stats=None, lamb2=0.8, gpu_transform=None, val_gpu_norm=None):
    start_time = time.time()
    net.train()

    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    all_preds = []
    all_labels = []

    for i, sample in enumerate(trainloader):
        counter += 1
        # sampleclip
        if len(sample) == 6:  # (center_img, clip_tensor, label, video_idx, frame_idx, clip_indices)
            data, clip, label_, video_idx, frame_idx, clip_indices = sample
            clip = clip.to(DEVICE, non_blocking=True)
            label = label_.to(DEVICE, non_blocking=True)
            # GPU transform: v2  T
            if gpu_transform is not None:
                clip = torch.stack([gpu_transform(clip[b]) for b in range(clip.shape[0])], 0)
                frame = clip[:, args.clip_length // 2]
                data = (frame, clip)
            else:
                data = data.to(DEVICE, non_blocking=True)
                data = (data, clip)
        elif len(sample) == 5:    # (center_img, clip_tensor, label, video_idx, frame_idx)
            data, clip, label_, video_idx, frame_idx = sample
            clip = clip.to(DEVICE, non_blocking=True)
            label = label_.to(DEVICE, non_blocking=True)
            if gpu_transform is not None:
                clip = torch.stack([gpu_transform(clip[b]) for b in range(clip.shape[0])], 0)
                frame = clip[:, args.clip_length // 2]
                data = (frame, clip)
            clip_indices = None
        else:  # (img, label, video_idx, frame_idx)
            data, label_, video_idx, frame_idx = sample
            data = data.to(DEVICE, non_blocking=True)
            label = label_.to(DEVICE, non_blocking=True)
            clip_indices = None

        optimizer.zero_grad()

        if uncertainty:
            y = one_hot_embedding(label_, n_class)
            y = y.to(DEVICE)
            logits = net(data)
            cur_loss = loss(
                logits, y.float(), epoch, n_class, 50, DEVICE
            )
        else:
            logits = net(data)
            cur_loss = loss(logits, label)

        train_running_loss += cur_loss.item()
        scores, preds = torch.max(logits.data, 1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(label.cpu().numpy())

        if selective:
            probs = get_expectedprob(logits, label_, lamb2=lamb2)
            cur_uncertainty = get_uncertainty(logits, n_class, lamb2=lamb2)
            inthe_bank = 1.0 - cur_uncertainty.squeeze(1)

            if args.segment_size > 0:
                trainloader.dataset.reset_segment_uncertainty(
                    video_idx, frame_idx, inthe_bank.detach().cpu()
                )
            elif clip_indices is not None:
                #  EMA  clip
                trainloader.dataset.reset_uncertainty_ema(
                    video_idx, clip_indices, inthe_bank.detach().cpu()
                )
            else:
                #  clip
                trainloader.dataset.reset_uncertainty(
                    video_idx, frame_idx, inthe_bank.detach().cpu()
                )

            last_video_idx = int(video_idx[0]) if isinstance(video_idx, torch.Tensor) else video_idx[0]
            last_frame_idx = int(frame_idx[0]) if isinstance(frame_idx, torch.Tensor) else frame_idx[0]
            last_batch_probs = probs[range(label.size(0)), label_].detach().cpu().tolist()
            last_batch_uncertainty = cur_uncertainty.squeeze(1).detach().cpu().tolist()

            #  batch  probs  uncertainty
            if batch_stats is not None:
                real_class_probs = probs[range(label.size(0)), label_].detach()
                global_batch_idx = len(batch_stats['batch_idx']) + 1
                batch_stats['batch_idx'].append(global_batch_idx)
                batch_stats['probs_mean'].append(real_class_probs.mean().item())
                batch_stats['probs_std'].append(real_class_probs.std().item())
                batch_stats['uncertainty_mean'].append(cur_uncertainty.mean().item())
                batch_stats['uncertainty_std'].append(cur_uncertainty.std().item())
        train_running_correct += (preds == label).sum().item()

        cur_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=20.0)
        optimizer.step()

    all_preds = np.concatenate(all_preds, 0)
    all_labels = np.concatenate(all_labels, 0)

    if epoch % args.val_interval == 0:
        train_acc = log_metrics(all_labels, all_preds, prefix='Train')
    else:
        train_acc = skmetrics.accuracy_score(all_labels, all_preds)

    logging.info(
        'Train --> Epoch[%d], lr[%.6f], classify loss: [%.6f], acc: [%.2f %%], time: %.1f s!'
        % (epoch, optimizer.param_groups[0]['lr'], train_running_loss/counter, 100.0*train_acc, time.time() - start_time))

    if selective and args.segment_size > 0:
        seg_list = trainloader.dataset.uncertainties[last_video_idx]
        seg_idx = last_frame_idx // args.segment_size
        marked = [f"*{round(float(seg_list[i]), 4)}*" if i == seg_idx else str(round(float(seg_list[i]), 4))
                  for i in range(len(seg_list))]
        logging.info('Epoch[%d] Video[%d] segment uncertainties (%d segs, sel=seg#%d): [%s]',
                     epoch, last_video_idx, len(seg_list), seg_idx,
                     ', '.join(marked))
        logging.info('Epoch[%d] Video[%d] expected_prob: %.4f  cur_uncertainty: %.4f',
                     epoch, last_video_idx,
                     round(float(last_batch_probs[0]), 4),
                     round(float(last_batch_uncertainty[0]), 4))

    start_time = time.time()
    if epoch % args.val_interval == 0:
        val_t0 = time.time()
        cur_val_acc = test(valloader, net, loss, n_class, args.clip_length, args.clip_interval, args.test_num_frames, args.lamb2, gpu_norm=val_gpu_norm)
        val_time = time.time() - val_t0
        logging.info('Val --> Epoch[%d], time: %.1f s!', epoch, val_time)
        if cur_val_acc > max_acc:
            max_acc = cur_val_acc
            break_flag = 0
            if len(args.gpu.split(',')) > 1 or args.gpu == 'all':
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()
            torch.save(
                {
                    'epoch': epoch,
                    'save_dir': save_dir,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'args': args
                }, os.path.join(save_dir, 'model.ckpt'))
            logging.info(
                '***********************model saved successful************************* !\n'
            )
        else:
            break_flag += 1
    else:
        cur_val_acc = max_acc
    if args.save_latest:
        if len(args.gpu.split(',')) > 1 or args.gpu == 'all':
            state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()
        torch.save(
            {
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'args': args
            }, os.path.join(save_dir, 'model_latest.ckpt'))

    #  epoch  batch
    if selective and batch_stats is not None and len(batch_stats['batch_idx']) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f'Cumulative Batch Statistics (Epoch {epoch})', fontsize=14)

        x = batch_stats['batch_idx']

        axes[0, 0].plot(x, batch_stats['probs_mean'], 'b-', alpha=0.7, linewidth=0.8)
        axes[0, 0].set_title('Probs Mean')
        axes[0, 0].set_xlabel('Batch')
        axes[0, 0].set_ylabel('Mean')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(x, batch_stats['probs_std'], 'r-', alpha=0.7, linewidth=0.8)
        axes[0, 1].set_title('Probs Std')
        axes[0, 1].set_xlabel('Batch')
        axes[0, 1].set_ylabel('Std')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(x, batch_stats['uncertainty_mean'], 'g-', alpha=0.7, linewidth=0.8)
        axes[1, 0].set_title('Uncertainty Mean')
        axes[1, 0].set_xlabel('Batch')
        axes[1, 0].set_ylabel('Mean')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(x, batch_stats['uncertainty_std'], 'm-', alpha=0.7, linewidth=0.8)
        axes[1, 1].set_title('Uncertainty Std')
        axes[1, 1].set_xlabel('Batch')
        axes[1, 1].set_ylabel('Std')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.join(save_dir, 'batch_stats'), exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'batch_stats', 'cumulative_batch_stats.png'), dpi=150)
        plt.close()

    return cur_val_acc, break_flag

def evaluation(valloader, net, loss, n_class):
    start_time = time.time()
    net.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for i, sample in enumerate(valloader):
            data = sample[0]
            label = sample[1]
            b, f, c, w, h = data.size(0), data.size(1), data.size(2), data.size(3), data.size(4)
            data = data.view(b*f, c, w, h)
            targets.append(label.numpy())
            data = data.to(DEVICE)
            label = label.to(DEVICE)

            logits = net(data)
            logits = logits.view(b, f, logits.size(1))
            _, cur_pred = torch.max(torch.sum(logits,1).data, 1)
            preds.append(cur_pred.cpu().numpy())

        preds = np.concatenate(preds, 0)
        targets = np.concatenate(targets, 0)

        return log_metrics(targets, preds, prefix='Metrics')

def test(testloader, net, loss, n_class, clip_length=10, clip_interval=1, num_frames=5, lamb2=0.8, gpu_norm=None):
    start_time = time.time()
    net.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for i, sample in enumerate(testloader):

            data = sample[0] # batch, image_num, channel, w, h
            data = data.to(DEVICE)
            if gpu_norm is not None:
                data = gpu_norm(data)
            label = sample[1]
            b, image_num, c, w, h = data.size()

            targets.append(label.numpy())

            # num_frames
            if image_num >= num_frames:
                key_indices = torch.linspace(0, image_num - 1, num_frames).long()
            else:
                # num_frames
                key_indices = torch.linspace(0, image_num - 1, min(num_frames, image_num)).long()
                if len(key_indices) < num_frames:
                    key_indices = torch.cat([key_indices, torch.full((num_frames - len(key_indices),), image_num - 1).long()])

            #  clip
            half_len = (clip_length // 2) * clip_interval
            all_frames = []
            all_clips = []
            for idx in key_indices:
                idx = idx.item()
                start_idx = idx - half_len

                clip_idx_list = []
                for i in range(clip_length):
                    cur_idx = start_idx + i * clip_interval
                    if cur_idx < 0:
                        clip_idx_list.append(0)
                    elif cur_idx >= image_num:
                        clip_idx_list.append(image_num - 1)
                    else:
                        clip_idx_list.append(cur_idx)

                all_frames.append(data[:, idx, :, :, :])
                all_clips.append(data[:, clip_idx_list, :, :, :])

            #  batch forward
            frames_batch = torch.cat(all_frames, dim=0)  # (L, C, H, W)
            clips_batch = torch.cat(all_clips, dim=0)    # (L, T, C, H, W)
            output = net((frames_batch, clips_batch))    # (L, n_class)

            evidence = F.softplus(output)
            alpha = evidence + lamb2
            total_alpha = alpha.sum(dim=0, keepdim=True)  # (1, n_class)
            S = total_alpha.sum(dim=1, keepdim=True)
            final_prob = total_alpha / S

            _, final_pred = torch.max(final_prob, 1)
            preds.append(final_pred.cpu().numpy())

        preds = np.concatenate(preds, 0)
        targets = np.concatenate(targets, 0)

        acc = skmetrics.accuracy_score(targets, preds)
        macro_recall = skmetrics.recall_score(targets, preds, average='macro')
        micro_recall = skmetrics.recall_score(targets, preds, average='micro')
        macro_f1 = skmetrics.f1_score(targets, preds, average='macro')
        micro_f1 = skmetrics.f1_score(targets, preds, average='micro')
        confusion = skmetrics.confusion_matrix(targets, preds)

        logging.info(
        'Metrics --> Accuracy: [%.6f], \n Macro F1: [%.6f], \n Micro F1: [%.6f], \n Macro Recall [%.6f], \n Micro Recall [%.6f], \n time: %.1f s!'
        % (acc, macro_f1, micro_f1, macro_recall, micro_recall,  time.time() - start_time))
        start_time = time.time()
        logging.info('Confuse matrix =======>')
        logging.info(confusion)

        return acc

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, args.model_name, args.loss_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s,%(lineno)d: %(message)s\n',
                        datefmt='%Y-%m-%d(%a)%H:%M:%S',
                        filename=os.path.join(args.save_dir, 'log.txt'),
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    main(args)

