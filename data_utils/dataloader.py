import os
import numpy as np
from torch.utils.data import Dataset
import time
import csv
from PIL import Image
import torch

INDEXOFLABEL = {'PLHLA': 0, 'PMASA': 1, 'PMVLSA': 2, 'PASA': 3, 'A4C': 4, 'A5C': 5, 'PMPALA': 6, 'PPMLSA': 7, 'SC4C': 8}


class Echo2DdataTest(Dataset):
    # video-level testing
    def __init__(self, transforms=None, phase='Test', parent_dir=None):
        self.phase = phase    
        self.transforms = transforms 
        data_files = []
        labels = []
        label_dist = np.zeros((9,))
        if phase=="Test":
            cur_datadir = os.path.join(parent_dir, 'test.txt')
        elif phase=="Val":
            cur_datadir = os.path.join(parent_dir, 'validation.txt')
        elif phase=="Train":
            cur_datadir = os.path.join(parent_dir, 'train.txt')
        
        count = 0
        label_dir = os.path.join(parent_dir, "labels.csv")
        subpar_dir = os.path.join(parent_dir, "Images")

        label_dict = {}
        with open(label_dir, mode='r', encoding='utf-8') as file:
            csv_dict_reader = csv.DictReader(file)
            for row in csv_dict_reader:
                label_dict[row['names']] = row['labels']

        with open(cur_datadir, "r") as f:
            xs = f.readlines()
            for x in xs:
                video_name = x.split("\n")[0]
                cur_files = []
                for file_ in sorted(os.listdir(os.path.join(subpar_dir, video_name))):
                    cur_files.append(os.path.join(subpar_dir, video_name, file_))
                data_files.append(cur_files)
                labels.append(INDEXOFLABEL[label_dict[video_name]])
                label_dist[INDEXOFLABEL[label_dict[video_name]]] +=1
                count += 1
        print("EchoVideo dataset have ", count, " ", phase, " videos")                  

        self.data_files = data_files
        self.labels = labels

        print('the data length is %d, for %s' % (len(self.data_files), phase))
        for i, name in enumerate(['PLHLA','PMASA','PMVLSA','PASA','A4C','A5C','PMPALA','PPMLSA','SC4C']):
            print(f"  class {name} -> {int(label_dist[i])}")

    def __len__(self):
        L = len(self.data_files)
        return L

    def __getitem__(self, index):        
        _label = self.labels[index]
        cur_files = self.data_files[index]
        if len(cur_files) > 300:
            cur_files = cur_files[:300]
        cur_images = []
        for file_ in cur_files:
            pil_img = Image.open(file_).convert("RGB")
            if self.transforms is not None:
                pil_img = self.transforms(pil_img)
            cur_images.append(pil_img)
        cur_imgs = torch.stack(cur_images, 0)
        return cur_imgs, _label


class EchoSelectdata(Dataset):
    def __init__(self, transforms=None, phase='Train', parent_dir=None, over_sample=None, selective=True, subset_size=None, clip_length=None, clip_interval=1):
        self.phase = phase    
        self.transforms = transforms 
        self.selective = selective
        self.subset_size = subset_size
        self.clip_length = clip_length
        self.clip_interval = clip_interval
        uncertainties = []
        data_files = []
        labels = []
        label_dist = np.zeros((9,))
        
        if phase=="Train":
            cur_datadir = os.path.join(parent_dir, 'train.txt')
        else:
            assert False, 'the dataset type not surport !'

        count = 0
        label_dir = os.path.join(parent_dir, "labels.csv")
        subpar_dir = os.path.join(parent_dir, "Images")

        label_dict = {}
        with open(label_dir, mode='r', encoding='utf-8') as file:
            csv_dict_reader = csv.DictReader(file)
            for row in csv_dict_reader:
                label_dict[row['names']] = row['labels']

        with open(cur_datadir, "r") as f:
            xs = f.readlines()
            for x in xs:
                video_name = x.split("\n")[0]
                cur_files = []
                cur_thresh = []
                for file_ in sorted(os.listdir(os.path.join(subpar_dir, video_name))):
                    cur_files.append(os.path.join(subpar_dir, video_name, file_))
                n_frames = len(cur_files)
                for _ in range(n_frames):
                    cur_thresh.append(-0.1)
                data_files.append(cur_files)
                uncertainties.append(cur_thresh)
                labels.append(INDEXOFLABEL[label_dict[video_name]])
                label_dist[INDEXOFLABEL[label_dict[video_name]]] +=1
                count += 1
        print("EchoVideo dataset have %d %s videos" % (count, phase))

        self.data_files = data_files
        self.labels = labels
        self.uncertainties = uncertainties
        self.epsilon = 0.2
        self.init = True
        if over_sample is not None:
            for idx in range(len(data_files)):
                for i in range(over_sample[labels[idx]]):
                    self.data_files.append(data_files[idx])
                    self.uncertainties.append(uncertainties[idx])
                    self.labels.append(labels[idx])
                    label_dist[labels[idx]] += 1   

        print('the data length is %d, for %s' % (len(self.data_files), phase))
        for i, name in enumerate(['PLHLA','PMASA','PMVLSA','PASA','A4C','A5C','PMPALA','PPMLSA','SC4C']):
            print(f"  class {name} -> {int(label_dist[i])}")

    def __len__(self):
        return len(self.data_files)

    def _sample_subset(self, n_frames):
        if self.subset_size is None or self.subset_size >= n_frames:
            return np.arange(n_frames)
        return np.random.choice(n_frames, self.subset_size, replace=False)

    def _get_clip_indices(self, center_idx, n_frames):
        half_len = (self.clip_length // 2) * self.clip_interval
        start_idx = center_idx - half_len
        indices = []
        for i in range(self.clip_length):
            cur = start_idx + i * self.clip_interval
            cur = max(0, min(cur, n_frames - 1))
            indices.append(cur)
        return indices

    def reset_uncertainty(self, index, frame_indices, uncertainty):
        for idx, threshold in enumerate(uncertainty):
            self.uncertainties[index][int(frame_indices[idx])] = float(threshold)

    def reset_uncertainty_ema(self, video_indices, clip_indices_list, new_uncertainties, min_weight=0.2):
        for vid, clip_indices, new_val in zip(video_indices, clip_indices_list, new_uncertainties):
            vid = int(vid)
            new_val = float(new_val)
            unique_indices = sorted(set(clip_indices.tolist()))
            for idx in unique_indices:
                old = self.uncertainties[vid][idx]
                if old < 0:
                    self.uncertainties[vid][idx] = new_val
                else:
                    self.uncertainties[vid][idx] = new_val * 0.3 + old * 0.7

    def init_uncertainty(self, index, uncertainty):
        for idx, threshold in enumerate(uncertainty):
            self.uncertainties[index][idx] = float(threshold)

    def __getitem__(self, index):
        if self.init:
            _label = self.labels[index]
            cur_files = self.data_files[index]
            n_frames = len(cur_files)

            if self.selective:
                uncertainties = np.array(self.uncertainties[index])
                if np.random.random() < self.epsilon:
                    cur_index = np.random.randint(n_frames)
                else:
                    subset = self._sample_subset(n_frames)
                    subset_unc = uncertainties[subset]
                    if subset_unc.min() >= 0:
                        cur_index = subset[subset_unc.argmin()]
                    else:
                        best_idx = np.where(subset_unc < 0)[0]
                        cur_index = subset[np.random.choice(best_idx)]
            else:
                cur_index = np.random.randint(n_frames)

            if self.clip_length is not None:
                clip_indices = self._get_clip_indices(cur_index, n_frames)
                clip_imgs = []
                for idx in clip_indices:
                    pil_img = Image.open(cur_files[idx]).convert("RGB")
                    if self.transforms is not None:
                        pil_img = self.transforms(pil_img)
                    clip_imgs.append(pil_img)
                clip_tensor = torch.stack(clip_imgs, 0)
                center_img = clip_imgs[self.clip_length // 2]
                return center_img, clip_tensor, _label, index, cur_index, clip_indices
            else:
                pil_img = Image.open(cur_files[cur_index]).convert("RGB")
                if self.transforms is not None:
                    pil_img = self.transforms(pil_img)
                return pil_img, _label, index, cur_index
        else:
            assert self.selective == True
            _label = self.labels[index]
            cur_files = self.data_files[index]
            cur_images = []
            for file_ in cur_files:
                pil_img = Image.open(file_).convert("RGB")
                if self.transforms is not None:
                    pil_img = self.transforms(pil_img)
                cur_images.append(pil_img)
            cur_imgs = torch.stack(cur_images, 0)
            return cur_imgs, _label, index


class EchoSelectSegdata(Dataset):
    def __init__(self, transforms=None, phase='Train', parent_dir=None, over_sample=None, selective=True, clip_length=None, clip_interval=1, segment_size=20, fixed_center=False):
        self.phase = phase    
        self.transforms = transforms 
        self.selective = selective
        self.clip_length = clip_length
        self.clip_interval = clip_interval
        self.segment_size = segment_size
        self.fixed_center = fixed_center
        uncertainties = []
        data_files = []
        labels = []
        label_dist = np.zeros((9,))
        
        if phase=="Train":
            cur_datadir = os.path.join(parent_dir, 'train.txt')
        elif phase=="Val":
            cur_datadir = os.path.join(parent_dir, 'validation.txt')
        else:
            assert False, 'the dataset type not surport !'

        count = 0
        label_dir = os.path.join(parent_dir, "labels.csv")
        subpar_dir = os.path.join(parent_dir, "Images")

        label_dict = {}
        with open(label_dir, mode='r', encoding='utf-8') as file:
            csv_dict_reader = csv.DictReader(file)
            for row in csv_dict_reader:
                label_dict[row['names']] = row['labels']

        with open(cur_datadir, "r") as f:
            xs = f.readlines()
            for x in xs:
                video_name = x.split("\n")[0]
                cur_files = []
                cur_thresh = []
                for file_ in sorted(os.listdir(os.path.join(subpar_dir, video_name))):
                    cur_files.append(os.path.join(subpar_dir, video_name, file_))
                n_frames = len(cur_files)
                n_segments = max(1, (n_frames + segment_size - 1) // segment_size)
                for _ in range(n_segments):
                    cur_thresh.append(-0.1)
                data_files.append(cur_files)
                uncertainties.append(cur_thresh)
                labels.append(INDEXOFLABEL[label_dict[video_name]])
                label_dist[INDEXOFLABEL[label_dict[video_name]]] +=1
                count += 1
        print("EchoVideo dataset have %d %s videos (segment_size=%d)" % (count, phase, segment_size))

        self.data_files = data_files
        self.labels = labels
        self.uncertainties = uncertainties
        self.epsilon = 0.2
        self.init = True
        if over_sample is not None:
            for idx in range(len(data_files)):
                for i in range(over_sample[labels[idx]]):
                    self.data_files.append(data_files[idx])
                    self.uncertainties.append(uncertainties[idx])
                    self.labels.append(labels[idx])
                    label_dist[labels[idx]] += 1   

        print('the data length is %d, for %s' % (len(self.data_files), phase))
        for i, name in enumerate(['PLHLA','PMASA','PMVLSA','PASA','A4C','A5C','PMPALA','PPMLSA','SC4C']):
            print(f"  class {name} -> {int(label_dist[i])}")
        print("uncertainty length is %d" % (len(uncertainties)))

    def __len__(self):
        L = len(self.data_files)
        return L

    def _random_frame_in_segment(self, seg_idx, n_frames):
        if self.fixed_center:
            start = seg_idx * self.segment_size
            return start + self.segment_size // 2
        start = seg_idx * self.segment_size
        end = min(start + self.segment_size, n_frames)
        if end <= start:
            return start
        return np.random.randint(start, end)

    def _get_clip_indices(self, center_idx, n_frames):
        half_len = (self.clip_length // 2) * self.clip_interval
        start_idx = center_idx - half_len
        indices = []
        for i in range(self.clip_length):
            cur = start_idx + i * self.clip_interval
            cur = max(0, min(cur, n_frames - 1))
            indices.append(cur)
        return indices

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def init_uncertainty(self, index, uncertainty):
        for idx, threshold in enumerate(uncertainty):
            seg_idx = min(idx // self.segment_size, len(self.uncertainties[index]) - 1)
            self.uncertainties[index][seg_idx] = float(threshold)

    def reset_segment_uncertainty(self, video_indices, center_frame_indices, new_uncertainties):
        ema_weight = 0.3
        for vid, frame_idx, new_val in zip(video_indices, center_frame_indices, new_uncertainties):
            vid = int(vid)
            frame_idx = int(frame_idx)
            new_val = float(new_val)
            seg_idx = frame_idx // self.segment_size
            seg_idx = min(seg_idx, len(self.uncertainties[vid]) - 1)
            old_val = self.uncertainties[vid][seg_idx]
            if old_val < 0:
                self.uncertainties[vid][seg_idx] = new_val
            else:
                self.uncertainties[vid][seg_idx] = new_val * ema_weight + old_val * (1 - ema_weight)

    def __getitem__(self, index):
        if self.init:
            _label = self.labels[index]
            cur_files = self.data_files[index]
            cur_index = 0

            if self.selective:
                uncertainties = np.array(self.uncertainties[index])
                if np.random.random() < self.epsilon:
                    cur_seg = np.random.randint(len(uncertainties))
                else:
                    cur_seg = uncertainties.argmax()
            else:
                n_segments = len(cur_files) // self.segment_size + 1
                cur_seg = np.random.randint(n_segments)

            n_frames = len(cur_files)
            cur_index = self._random_frame_in_segment(cur_seg, n_frames)

            if self.clip_length is not None:
                clip_indices = self._get_clip_indices(cur_index, n_frames)
                clip_imgs = []
                crop_transform = None
                crop_params = None
                if hasattr(self.transforms, 'transforms'):
                    for t in self.transforms.transforms:
                        if hasattr(t, 'random_params_for_img'):
                            crop_transform = t
                            ref_img = Image.open(cur_files[clip_indices[0]]).convert("RGB")
                            crop_params = crop_transform.random_params_for_img(ref_img)
                            break

                for idx in clip_indices:
                    pil_img = Image.open(cur_files[idx]).convert("RGB")
                    if self.transforms is not None:
                        if crop_params is not None:
                            for t in self.transforms.transforms:
                                if t is crop_transform:
                                    pil_img = t(pil_img, params=crop_params)
                                else:
                                    pil_img = t(pil_img)
                        else:
                            pil_img = self.transforms(pil_img)
                    clip_imgs.append(pil_img)
                clip_tensor = torch.stack(clip_imgs, 0)
                center_img = clip_imgs[len(clip_imgs) // 2]
                return center_img, clip_tensor, _label, index, cur_index, clip_indices
            else:
                pil_img = Image.open(cur_files[cur_index]).convert("RGB")
                if self.transforms is not None:
                    pil_img = self.transforms(pil_img)
                return pil_img, _label, index, cur_index
        else:
            assert self.selective == True
            _label = self.labels[index]
            cur_files = self.data_files[index]
            cur_images = []
            for file_ in cur_files:
                pil_img = Image.open(file_).convert("RGB")
                if self.transforms is not None:
                    pil_img = self.transforms(pil_img)
                cur_images.append(pil_img)
            cur_imgs = torch.stack(cur_images, 0)
            return cur_imgs, _label, index
