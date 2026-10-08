import os
import numpy as np
from torch.utils.data import Dataset
import pdb
import time
import cv2
import scipy
import csv
from PIL import Image
import torch
 
INDEXOFLABEL = {'PLHLA': 0, 'PMASA': 1, 'PMVLSA': 2, 'PASA': 3, 'A4C': 4, 'A5C': 5, 'PMPALA': 6, 'PPMLSA': 7, 'SC4C': 8}

class Echo2Ddata(Dataset):
    # image-level training and validation
    def __init__(self, transforms=None, phase='Train', parent_dir=None, over_sample=None):
        self.phase = phase    
        self.transforms = transforms 
        data_files = []
        labels = []
        label_dist = np.zeros((9,))
        
        if phase=="Train":
            cur_datadir = os.path.join(parent_dir, 'train.txt')
        
        else:
            assert False, 'the dataset type not surport !'

        count = 0
        label_dir = os.path.join(parent_dir, "labels.csv")
        subpar_dir = os.path.join(parent_dir, "Images") # convert videos into images first

        # processing the csv label file 
        label_dict = {}
        with open(label_dir, mode='r', encoding='utf-8') as file:
            csv_dict_reader = csv.DictReader(file)
        
            for row in csv_dict_reader:
                label_dict[row['names']] = row['labels']
        
        with open(cur_datadir, "r") as f:
            xs = f.readlines()
            for x in xs:
                video_name = x.split("\n")[0]
                for file_ in sorted(os.listdir(os.path.join(subpar_dir, video_name))):
                    labels.append(INDEXOFLABEL[label_dict[video_name]])
                    data_files.append(os.path.join(subpar_dir, video_name, file_))
                    label_dist[INDEXOFLABEL[label_dict[video_name]]] +=1
                    count += 1
        print("EchoVideo dataset have ", count, " ", phase, " images")                  

        ###################### OVER SAMPLING HERE ############################
        self.data_files = data_files
        self.labels = labels

        if over_sample is not None:
            for idx in range(len(data_files)):
                for i in range(over_sample[labels[idx]]):
                    self.data_files.append(data_files[idx])
                    self.labels.append(labels[idx])
                    label_dist[labels[idx]] += 1          

        print('the data length is %d, for %s' % (len(self.data_files), phase))
        print("class PLHLA->%d" % (label_dist[0]))
        print("class PMASA->%d" % (label_dist[1]))
        print("class PMVLSA->%d" % (label_dist[2]))
        print("class PASA->%d" % (label_dist[3]))
        print("class A4C->%d" % (label_dist[4]))
        print("class A5C->%d" % (label_dist[5]))
        print("class PMPALA->%d" % (label_dist[6]))
        print("class PPMLSA->%d" % (label_dist[7]))
        print("class SC4C->%d" % (label_dist[8]))

    def __len__(self):
        L = len(self.data_files)
        return L
    

    def __getitem__(self, index):        
        _label = self.labels[index]
        pil_img = Image.open(self.data_files[index]).convert("RGB")
        if self.transforms is not None:
            pil_img = self.transforms(pil_img)
        return pil_img, _label
    def __str__(self):
        pass



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
        subpar_dir = os.path.join(parent_dir, "Images") # convert videos into images first

        # processing the csv label file 
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

        ###################### OVER SAMPLING HERE ############################
        self.data_files = data_files
        self.labels = labels

        print('the data length is %d, for %s' % (len(self.data_files), phase))
        print("class PLHLA->%d" % (label_dist[0]))
        print("class PMASA->%d" % (label_dist[1]))
        print("class PMVLSA->%d" % (label_dist[2]))
        print("class PASA->%d" % (label_dist[3]))
        print("class A4C->%d" % (label_dist[4]))
        print("class A5C->%d" % (label_dist[5]))
        print("class PMPALA->%d" % (label_dist[6]))
        print("class PPMLSA->%d" % (label_dist[7]))
        print("class SC4C->%d" % (label_dist[8]))

    def __len__(self):
        L = len(self.data_files)
        return L
    

    def __getitem__(self, index):        
        _label = self.labels[index]
        cur_files = self.data_files[index]
        cur_images = []
        if len(cur_files) > 300: # video larger than 10s was truncated
            cur_files = cur_files[:300]
        for file_ in cur_files:
            pil_img = Image.open(file_).convert("RGB")
            if self.transforms is not None:
                pil_img = self.transforms(pil_img)
            cur_images.append(pil_img)
        cur_imgs = torch.stack(cur_images, 0)
        return cur_imgs, _label
    def __str__(self):
        pass


class Echo2DdataTestFast(Dataset):
    """Lazy evaluation loader (opt-in).

    Loads ONLY the frames that trainSelective.test() actually consumes
    (the `test_num_frames` key frames plus each key frame's clip frames,
    <= ~50 unique frames) instead of decoding every frame. The returned
    tensor has the identical shape and identical values at all used
    indices as Echo2DdataTest, so model inputs are bit-identical and
    metrics are unchanged. Deterministic val/test transforms only.
    """
    def __init__(self, transforms=None, phase='Test', parent_dir=None,
                 test_num_frames=10, clip_length=5, clip_interval=5,
                 frame_cap=300):
        self.transforms = transforms
        self.test_num_frames = test_num_frames
        self.clip_length = clip_length
        self.clip_interval = clip_interval
        self.frame_cap = frame_cap
        data_files = []
        labels = []
        if phase == "Test":
            cur_datadir = os.path.join(parent_dir, 'test.txt')
        elif phase == "Val":
            cur_datadir = os.path.join(parent_dir, 'validation.txt')
        elif phase == "Train":
            cur_datadir = os.path.join(parent_dir, 'train.txt')
        else:
            raise ValueError(phase)
        label_dir = os.path.join(parent_dir, "labels.csv")
        subpar_dir = os.path.join(parent_dir, "Images")
        label_dict = {}
        with open(label_dir, mode='r', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                label_dict[row['names']] = row['labels']
        with open(cur_datadir, "r") as f:
            for x in f.readlines():
                video_name = x.split("\n")[0]
                cur_files = [os.path.join(subpar_dir, video_name, fn)
                             for fn in sorted(os.listdir(os.path.join(subpar_dir, video_name)))]
                data_files.append(cur_files)
                labels.append(INDEXOFLABEL[label_dict[video_name]])
        self.data_files = data_files
        self.labels = labels
        print("EchoVideo dataset have %d %s videos (fast eval)" % (len(data_files), phase))

    def __len__(self):
        return len(self.data_files)

    def _needed_indices(self, n_frames):
        num_frames = self.test_num_frames
        if n_frames >= num_frames:
            key = torch.linspace(0, n_frames - 1, num_frames).long()
        else:
            key = torch.linspace(0, n_frames - 1, min(num_frames, n_frames)).long()
            if len(key) < num_frames:
                pad = torch.full((num_frames - len(key),), n_frames - 1).long()
                key = torch.cat([key, pad])
        half_len = (self.clip_length // 2) * self.clip_interval
        needed = set()
        for k in key.tolist():
            needed.add(k)
            for i in range(self.clip_length):
                cur = k - half_len + i * self.clip_interval
                if cur < 0:
                    cur = 0
                elif cur >= n_frames:
                    cur = n_frames - 1
                needed.add(cur)
        return needed

    def __getitem__(self, index):
        _label = self.labels[index]
        cur_files = self.data_files[index]
        if len(cur_files) > self.frame_cap:
            cur_files = cur_files[:self.frame_cap]
        n_frames = len(cur_files)
        needed = self._needed_indices(n_frames)
        out = None
        for i in sorted(needed):
            pil_img = Image.open(cur_files[i]).convert("RGB")
            if self.transforms is not None:
                pil_img = self.transforms(pil_img)
            if out is None:
                out = torch.zeros((n_frames, *pil_img.shape), dtype=torch.uint8)
            out[i] = pil_img
        if out is None:
            raise RuntimeError(f'empty video index={index}')
        return out, _label


class Echo3Ddata(Dataset):
    # video-level training
    def __init__(self, transforms=None, phase='Train', parent_dir=None, over_sample=None, frameNo=32,size=112):
        self.phase = phase    
        self.frameNo = frameNo
        self.transforms = transforms 
        self.size = size
        
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
        subpar_dir = os.path.join(parent_dir, "Images") # convert videos into images first

        # processing the csv label file 
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

        ###################### OVER SAMPLING HERE ############################
        self.data_files = data_files
        self.labels = labels

        if over_sample is not None:
            for idx in range(len(data_files)):
                for i in range(over_sample[labels[idx]]):
                    self.data_files.append(data_files[idx])
                    self.labels.append(labels[idx])
                    label_dist[labels[idx]] += 1   

        print('the data length is %d, for %s' % (len(self.data_files), phase))
        print("class PLHLA->%d" % (label_dist[0]))
        print("class PMASA->%d" % (label_dist[1]))
        print("class PMVLSA->%d" % (label_dist[2]))
        print("class PASA->%d" % (label_dist[3]))
        print("class A4C->%d" % (label_dist[4]))
        print("class A5C->%d" % (label_dist[5]))
        print("class PMPALA->%d" % (label_dist[6]))
        print("class PPMLSA->%d" % (label_dist[7]))
        print("class SC4C->%d" % (label_dist[8]))

    def __len__(self):
        L = len(self.data_files)
        return L
    

    def __getitem__(self, index):        
        _label = self.labels[index]
        cur_files = self.data_files[index]
        while len(cur_files)<self.frameNo:      #if the video is shorter than frameNo, then repeat it
            cur_files = cur_files+cur_files
        # np.random.shuffle(cur_files)
        cur_files = cur_files[:self.frameNo]
        cur_images = []
        for file_ in cur_files:
            pil_img = Image.open(file_).convert("RGB")
            pil_img = pil_img.resize((self.size,self.size))
            if self.transforms is not None:
                pil_img = self.transforms(pil_img)
            cur_images.append(pil_img)
        cur_imgs = torch.stack(cur_images, 0)       # (frameNo, 3, size, size)
        return cur_imgs, _label
    def __str__(self):
        pass


class EchoSelectdata(Dataset):
    def __init__(self, transforms=None, phase='Train', parent_dir=None, over_sample=None, selective=True, subset_size=None, clip_length=None, clip_interval=1):
        self.phase = phase    
        self.transforms = transforms 
        self.selective = selective
        self.subset_size = subset_size  # 子集大小，None表示使用所有帧
        self.clip_length = clip_length  # 片段长度，None表示只返回单帧，否则返回前后共L帧的片段
        self.clip_interval = clip_interval  # 片段采样间隔，1表示连续帧，2表示隔1帧采样
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
        subpar_dir = os.path.join(parent_dir, "Images") # convert videos into images first

        # processing the csv label file 
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
                    cur_thresh.append(0)
                data_files.append(cur_files)
                uncertainties.append(cur_thresh)
                labels.append(INDEXOFLABEL[label_dict[video_name]])
                label_dist[INDEXOFLABEL[label_dict[video_name]]] +=1
                count += 1
        print("EchoVideo dataset have ", count, " ", phase, " videos")                  

        ###################### OVER SAMPLING HERE ############################
        self.data_files = data_files
        self.labels = labels
        self.uncertainties = uncertainties
        self.init = True
        # 存储每个视频当前的子集索引映射: {video_idx: [original_indices]}
        self.subset_indices = {}
        if over_sample is not None:
            for idx in range(len(data_files)):
                for i in range(over_sample[labels[idx]]):
                    self.data_files.append(data_files[idx])
                    self.uncertainties.append(uncertainties[idx])
                    self.labels.append(labels[idx])
                    label_dist[labels[idx]] += 1   

        print('the data length is %d, for %s' % (len(self.data_files), phase))
        print("class PLHLA->%d" % (label_dist[0]))
        print("class PMASA->%d" % (label_dist[1]))
        print("class PMVLSA->%d" % (label_dist[2]))
        print("class PASA->%d" % (label_dist[3]))
        print("class A4C->%d" % (label_dist[4]))
        print("class A5C->%d" % (label_dist[5]))
        print("class PMPALA->%d" % (label_dist[6]))
        print("class PPMLSA->%d" % (label_dist[7]))
        print("class SC4C->%d" % (label_dist[8]))

    def __len__(self):
        L = len(self.data_files)
        return L    

    def init_uncertainty(self, index, uncertainty):
        for idx, threshold in enumerate(uncertainty):
            self.uncertainties[index][idx] = float(threshold)
            
    def reset_uncertainty(self, index, frame_indices, uncertainty):
        """
        更新不确定性值
        Args:
            index: 视频索引列表
            frame_indices: 每个视频中被选中的原始帧索引列表
            uncertainty: 对应的不确定性值列表
        """
        for idx, frame_idx, threshold in zip(index, frame_indices, uncertainty):
            # 直接使用原始帧索引更新，确保转换为整数
            self.uncertainties[idx][int(frame_idx)] = float(threshold)

    def reset_uncertainty_ema(self, video_indices, clip_indices_list, new_uncertainties, min_weight=0.2):
        """
        使用 EMA 更新 clip 中所有帧的不确定性
        重复帧按最大权重更新一次
        
        Args:
            video_indices: batch 中每个样本的视频索引 list/tensor
            clip_indices_list: batch 中每个样本的 clip 帧索引列表
                              例如：[[18,24,30,36,42], [2,8,14,20,26], ...]
            new_uncertainties: batch 中每个样本的新不确定性值（中心帧计算得到）
                              例如：[1.095, 0.823, ...]
            min_weight: 边缘帧最小权重（默认 0.2）
        """
        clip_length = len(clip_indices_list[0])
        center = clip_length // 2
        max_dist = center
        
        # 计算权重：中心=1.0，边缘=min_weight，线性衰减
        weights = []
        for pos in range(clip_length):
            dist = abs(pos - center)
            if max_dist == 0:
                w = 1.0
            else:
                w = 1.0 - (dist / max_dist) * (1.0 - min_weight)
            weights.append(w)
        
        for vid, clip, new_val in zip(video_indices, clip_indices_list, new_uncertainties):
            vid = int(vid)
            new_val = float(new_val)
            n_frames = len(self.uncertainties[vid])
            
            # 收集每个原始帧的最大权重（处理重复帧）
            frame_updates = {}
            for pos, frame_idx in enumerate(clip):
                frame_idx = int(frame_idx)
                # 边界检查：越界则跳过
                if frame_idx < 0 or frame_idx >= n_frames:
                    continue
                w = weights[pos]
                if frame_idx not in frame_updates or w > frame_updates[frame_idx]:
                    frame_updates[frame_idx] = w
            
            # 应用 EMA 更新
            for frame_idx, w in frame_updates.items():
                old_val = self.uncertainties[vid][frame_idx]
                updated = new_val * w + old_val * (1 - w)
                self.uncertainties[vid][frame_idx] = updated

    def get_item(self, index):
        assert self.selective==True, "please setting selective first!"
        _label = self.labels[index]
        cur_files = self.data_files[index]
        cur_uncertainty = self.uncertainties[index]
        cur_images = []
        for file_ in cur_files:
            pil_img = Image.open(file_).convert("RGB")
            if self.transforms is not None:
                pil_img = self.transforms(pil_img)
            cur_images.append(pil_img)
        cur_imgs = torch.stack(cur_images, 0)
        return cur_imgs, _label, index
        
    
    def _sample_subset(self, index):
        """从视频中采样一个子集，返回子集的原始索引列表"""
        n_frames = len(self.data_files[index])
        if self.subset_size is None or self.subset_size >= n_frames:
            # 使用所有帧
            return list(range(n_frames))
        else:
            # 随机采样子集
            return np.random.choice(n_frames, size=self.subset_size, replace=False).tolist()
    
    def _get_clip_indices(self, center_idx, n_frames):
        """
        获取以center_idx为中心，长度为clip_length的片段索引
        支持间隔采样，超出范围用首帧/尾帧填充
        
        Args:
            center_idx: 中心帧索引
            n_frames: 视频总帧数
        Returns:
            长度为clip_length的索引列表
        """
        half_len = (self.clip_length // 2) * self.clip_interval
        start_idx = center_idx - half_len
        
        clip_indices = []
        for i in range(self.clip_length):
            # 计算当前帧索引，考虑间隔
            cur_idx = start_idx + i * self.clip_interval
            # 边界填充：小于0用首帧，大于等于n_frames用尾帧
            if cur_idx < 0:
                clip_indices.append(0)
            elif cur_idx >= n_frames:
                clip_indices.append(n_frames - 1)
            else:
                clip_indices.append(cur_idx)
        return clip_indices
    
    def __getitem__(self, index):
        if self.init:
            _label = self.labels[index]
            cur_files = self.data_files[index]
            cur_index = 0

            if self.selective:
                subset_indices = self._sample_subset(index)
                self.subset_indices[index] = subset_indices
                cur_select = np.array(self.uncertainties[index])
                subset_uncertainties = cur_select[subset_indices]
                subset_min_idx = subset_uncertainties.argmin()
                cur_index = subset_indices[subset_min_idx]
            else:
                cur_index = np.random.randint(len(cur_files))

            if self.clip_length is not None:
                n_frames = len(cur_files)
                clip_indices = self._get_clip_indices(cur_index, n_frames)
                clip_imgs = []
                for idx in clip_indices:
                    pil_img = Image.open(cur_files[idx]).convert("RGB")
                    if self.transforms is not None:
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
            assert self.selective==True, "please setting selective first!"
            _label = self.labels[index]
            cur_files = self.data_files[index]
            cur_uncertainty = self.uncertainties[index]
            cur_images = []
            for file_ in cur_files:
                pil_img = Image.open(file_).convert("RGB")
                if self.transforms is not None:
                    pil_img = self.transforms(pil_img)
                cur_images.append(pil_img)
            cur_imgs = torch.stack(cur_images, 0)
            return cur_imgs, _label, index
        
    def __str__(self):
        pass

    def set_init(self, IFINIT):
        self.init = IFINIT


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
        self.epsilon = 0.8
        self.init = True
        if over_sample is not None:
            for idx in range(len(data_files)):
                for i in range(over_sample[labels[idx]]):
                    self.data_files.append(data_files[idx])
                    self.uncertainties.append(uncertainties[idx])
                    self.labels.append(labels[idx])
                    label_dist[labels[idx]] += 1   

        # shared-state fix (2026-10-02): epsilon and uncertainty bank must be
        # visible to persistent DataLoader workers. Both live in shared memory.
        n_unique = len(uncertainties)
        offsets_unique = []
        _total = 0
        for i in range(n_unique):
            offsets_unique.append((_total, len(uncertainties[i])))
            _total += len(uncertainties[i])
        self._unc_offsets = list(offsets_unique)
        if over_sample is not None:
            for idx in range(n_unique):
                for _ in range(over_sample[labels[idx]]):
                    self._unc_offsets.append(offsets_unique[idx])
        self._unc_flat = torch.full((_total,), -0.1,
                                    dtype=torch.float64).share_memory_()
        for i in range(n_unique):
            off, n = offsets_unique[i]
            self._unc_flat[off:off + n] = torch.tensor(uncertainties[i],
                                                       dtype=torch.float64)
        self._eps_shared = torch.tensor([0.2], dtype=torch.float64).share_memory_()

        print('the data length is %d, for %s' % (len(self.data_files), phase))
        print("class PLHLA->%d" % (label_dist[0]))
        print("class PMASA->%d" % (label_dist[1]))
        print("class PMVLSA->%d" % (label_dist[2]))
        print("class PASA->%d" % (label_dist[3]))
        print("class A4C->%d" % (label_dist[4]))
        print("class A5C->%d" % (label_dist[5]))
        print("class PMPALA->%d" % (label_dist[6]))
        print("class PPMLSA->%d" % (label_dist[7]))
        print("class SC4C->%d" % (label_dist[8]))
        print("uncertainty length is %d" % (len(uncertainties)))

    def __len__(self):
        L = len(self.data_files)
        return L

    def init_uncertainty(self, index, uncertainty):
        index = int(index)
        for idx, threshold in enumerate(uncertainty):
            seg_idx = min(idx // self.segment_size, len(self.uncertainties[index]) - 1)
            self.uncertainties[index][seg_idx] = float(threshold)
            off, _ = self._unc_offsets[index]
            self._unc_flat[off + seg_idx] = float(threshold)

    def reset_uncertainty(self, index, frame_indices, uncertainty):
        for idx, frame_idx, threshold in zip(index, frame_indices, uncertainty):
            idx = int(idx)
            seg_idx = int(frame_idx) // self.segment_size
            seg_idx = min(seg_idx, len(self.uncertainties[idx]) - 1)
            self.uncertainties[idx][seg_idx] = float(threshold)
            off, _ = self._unc_offsets[idx]
            self._unc_flat[off + seg_idx] = float(threshold)

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
                updated = new_val
            else:
                updated = new_val * ema_weight + old_val * (1 - ema_weight)
            self.uncertainties[vid][seg_idx] = updated
            off, _ = self._unc_offsets[vid]
            self._unc_flat[off + seg_idx] = updated

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
        self._eps_shared[0] = float(epsilon)

    def _load_video_to_cache(self, video_idx):
        """加载一个视频的所有帧到缓存，避免重复磁盘 I/O"""
        files = self.data_files[video_idx]
        frames = [Image.open(f).convert("RGB") for f in files]
        self._frame_cache[video_idx] = frames
        if len(self._frame_cache) > self._cache_max_videos:
            oldest = next(iter(self._frame_cache))
            del self._frame_cache[oldest]
        return frames

    def _get_video_frame(self, video_idx, frame_idx):
        """从缓存获取帧，缓存未命中则加载整个视频"""
        if video_idx not in self._frame_cache:
            self._load_video_to_cache(video_idx)
        return self._frame_cache[video_idx][frame_idx]

    def get_item(self, index):
        assert self.selective==True, "please setting selective first!"
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

    def _get_clip_indices(self, center_idx, n_frames):
        half_len = (self.clip_length // 2) * self.clip_interval
        start_idx = center_idx - half_len

        clip_indices = []
        for i in range(self.clip_length):
            cur_idx = start_idx + i * self.clip_interval
            if cur_idx < 0:
                clip_indices.append(0)
            elif cur_idx >= n_frames:
                clip_indices.append(n_frames - 1)
            else:
                clip_indices.append(cur_idx)
        return clip_indices

    def _random_frame_in_segment(self, seg_idx, n_frames):
        if self.fixed_center:
            start = seg_idx * self.segment_size
            return start + self.segment_size // 2
        start = seg_idx * self.segment_size
        end = min(start + self.segment_size, n_frames)
        if end <= start:
            return start
        return np.random.randint(start, end)

    def __getitem__(self, index):
        if self.init:
            _label = self.labels[index]
            cur_files = self.data_files[index]
            cur_index = 0

            if self.selective:
                off, n_seg = self._unc_offsets[index]
                uncertainties = self._unc_flat[off:off + n_seg]
                if np.random.random() < float(self._eps_shared[0]):
                    cur_seg = np.random.randint(n_seg)
                else:
                    cur_seg = int(uncertainties.argmax())
            else:
                n_segments = len(cur_files) // self.segment_size + 1
                cur_seg = np.random.randint(n_segments)

            n_frames = len(cur_files)
            cur_index = self._random_frame_in_segment(cur_seg, n_frames)

            if self.clip_length is not None:
                clip_indices = self._get_clip_indices(cur_index, n_frames)
                clip_imgs = []

                for idx in clip_indices:
                    pil_img = Image.open(cur_files[idx]).convert("RGB")
                    if self.transforms is not None:
                        pil_img = self.transforms(pil_img)
                    clip_imgs.append(pil_img)
                clip_tensor = torch.stack(clip_imgs, 0)
                center_img = clip_tensor[len(clip_tensor) // 2]
                return center_img, clip_tensor, _label, index, cur_index, clip_indices
            else:
                pil_img = Image.open(cur_files[cur_index]).convert("RGB")
                if self.transforms is not None:
                    pil_img = self.transforms(pil_img)
                return pil_img, _label, index, cur_index
        else:
            assert self.selective==True, "please setting selective first!"
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

    def __str__(self):
        pass

    def set_init(self, IFINIT):
        self.init = IFINIT


class EchoRandomFrameData(Dataset):
    # video-level training with random frame selection
    def __init__(self, transforms=None, phase='Train', parent_dir=None):
        self.phase = phase    
        self.transforms = transforms 
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
        subpar_dir = os.path.join(parent_dir, "Images") # convert videos into images first

        # processing the csv label file 
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
        print("class PLHLA->%d" % (label_dist[0]))
        print("class PMASA->%d" % (label_dist[1]))
        print("class PMVLSA->%d" % (label_dist[2]))
        print("class PASA->%d" % (label_dist[3]))
        print("class A4C->%d" % (label_dist[4]))
        print("class A5C->%d" % (label_dist[5]))
        print("class PMPALA->%d" % (label_dist[6]))
        print("class PPMLSA->%d" % (label_dist[7]))
        print("class SC4C->%d" % (label_dist[8]))

    def __len__(self):
        L = len(self.data_files)
        return L
    

    def __getitem__(self, index):        
        _label = self.labels[index]
        cur_files = self.data_files[index]
        # randomly choose one frame from the video
        rand_idx = np.random.randint(len(cur_files))
        pil_img = Image.open(cur_files[rand_idx]).convert("RGB")
        if self.transforms is not None:
            pil_img = self.transforms(pil_img)
        return pil_img, _label
    def __str__(self):
        pass


class EchoRNNData(Dataset):
    """Video-level clip sampling for RNN training.

    Training: random clip per video per iteration.
    Testing: 10 uniform clips per video, softmax averaged.
    """
    def __init__(self, transforms=None, phase='Train', parent_dir=None, over_sample=None,
                 seq_length=16, seq_interval=4, clip_size=10):
        self.phase = phase
        self.transforms = transforms
        self.seq_length = seq_length
        self.seq_interval = seq_interval
        self.clip_size = clip_size

        data_files, labels = [], []
        label_dict = {}
        label_dir = os.path.join(parent_dir, "labels.csv")
        subpar_dir = os.path.join(parent_dir, "Images")

        with open(label_dir, mode='r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                label_dict[row['names']] = row['labels']

        if phase == 'Train':
            cur_datadir = os.path.join(parent_dir, 'train.txt')
        elif phase == 'Val':
            cur_datadir = os.path.join(parent_dir, 'validation.txt')
        else:
            cur_datadir = os.path.join(parent_dir, 'test.txt')

        with open(cur_datadir, "r") as f:
            for line in f:
                video_name = line.strip()
                frame_paths = sorted(os.listdir(os.path.join(subpar_dir, video_name)))
                frame_paths = [os.path.join(subpar_dir, video_name, f) for f in frame_paths]
                data_files.append(frame_paths)
                labels.append(INDEXOFLABEL[label_dict[video_name]])

        self.data_files = data_files
        self.labels = labels

        if over_sample is not None and phase == 'Train':
            for idx in range(len(data_files)):
                for _ in range(over_sample[labels[idx]]):
                    self.data_files.append(data_files[idx])
                    self.labels.append(labels[idx])

        print(f"EchoRNNData: {len(self.data_files)} videos, phase={phase}, "
              f"seq_len={seq_length}, interval={seq_interval}")

    def __len__(self):
        return len(self.data_files)

    def _sample_clip(self, frames, n_frames):
        total_needed = (self.seq_length - 1) * self.seq_interval + 1
        if n_frames >= total_needed:
            start = np.random.randint(0, n_frames - total_needed + 1)
        else:
            start = 0
        indices = [min(start + i * self.seq_interval, n_frames - 1) for i in range(self.seq_length)]
        return [frames[i] for i in indices]

    def __getitem__(self, index):
        frames = self.data_files[index]
        label = self.labels[index]
        n_frames = len(frames)

        if self.phase == 'Train':
            clip_paths = self._sample_clip(frames, n_frames)
            clip_imgs = []
            for p in clip_paths:
                img = Image.open(p).convert("RGB")
                if self.transforms is not None:
                    img = self.transforms(img)
                clip_imgs.append(img)
            clip = torch.stack(clip_imgs, 0)
            return clip, label
        else:
            total_needed = (self.seq_length - 1) * self.seq_interval + 1
            if n_frames >= total_needed:
                max_start = n_frames - total_needed
                start_positions = np.linspace(0, max_start, self.clip_size, dtype=int)
            else:
                start_positions = np.zeros(self.clip_size, dtype=int)

            all_clips = []
            for start in start_positions:
                indices = [min(start + i * self.seq_interval, n_frames - 1) for i in range(self.seq_length)]
                clip_imgs = []
                for i in indices:
                    img = Image.open(frames[i]).convert("RGB")
                    if self.transforms is not None:
                        img = self.transforms(img)
                    clip_imgs.append(img)
                all_clips.append(torch.stack(clip_imgs, 0))
            clips = torch.stack(all_clips, 0)
            return clips, label
