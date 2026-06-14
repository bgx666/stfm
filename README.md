# STFM - Spatial-Temporal Fusion Model for Echocardiogram Classification

A deep learning framework for echocardiogram video view classification using Evidential Deep Learning with selective frame sampling.

## Overview

This model classifies 9 standard echocardiogram views from ultrasound video clips using a hybrid CNN-RNN architecture with uncertainty estimation. The training loop incorporates:

- **Enhanced STFM**: Shared backbone + spatial head + temporal head (conv + LSTM)
- **Selective frame sampling**: Uncertainty-guided frame selection during training
- **REEDL loss**: Regularized Evidential Deep Learning loss
- **GPU-accelerated augmentations**: Using torchvision v2 transforms with per-clip consistency

## Supported Views (9 classes)

| Class | View |
|:----:|------|
| PLHLA | Parasternal Long Axis |
| PMASA | Parasternal Short Axis (Mitral Valve) |
| PMVLSA | Parasternal Short Axis (Papillary Muscle) |
| PASA | Parasternal Short Axis (Aortic Valve) |
| A4C | Apical 4-Chamber |
| A5C | Apical 5-Chamber |
| PMPALA | Parasternal Long Axis (Apical) |
| PPMLSA | Parasternal Short Axis (LV) |
| SC4C | Subcostal 4-Chamber |

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy
scikit-learn
Pillow
matplotlib
pyyaml
```

## Data Preparation

Expected data structure:

```
data/EchoData/
├── labels.csv           # video_name, label (e.g. "2022-01-01_12-00-00_320x240_1,A4C")
├── train.txt            # video names for training (one per line)
├── validation.txt       # video names for validation
├── test.txt             # video names for testing
└── Images/
    └── {video_name}/
        ├── frame_000001.jpg
        ├── frame_000002.jpg
        └── ...
```

## Usage

### Training

```bash
python trainSelective.py --model_name resnet18 --gpu 0 --batch_size 64
```

### Testing / Evaluation

```bash
python trainSelective.py --model_name resnet18 --gpu 0 \
    --test_flag 1 --resume /path/to/model.ckpt
```

### Key Arguments

| Argument | Default | Description |
|----------|:-------:|-------------|
| `--model_name` | `resnet18` | Backbone: resnet18/50, convnext_*, densenet*, etc. |
| `--use_enhanced` | 1 | Use Enhanced STFM (shared backbone) |
| `--batch_size` | 64 | Training batch size |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 100 | Max epochs |
| `--clip_length` | 5 | Frames per clip |
| `--clip_interval` | 5 | Sampling interval within clip |
| `--segment_size` | 20 | Segment size for uncertainty bank |
| `--selective` | 1 | Enable selective (uncertainty-guided) sampling |
| `--uncertainty` | 1 | Enable uncertainty estimation |
| `--epsilon` | 0.2 | Random exploration rate (0=fully greedy, 1=fully random) |
| `--lamb2` | 0.8 | REEDL loss lambda parameter |
| `--temporal_hidden` | 512 | LSTM hidden size |
| `--temporal_layers` | 2 | LSTM layers |
| `--fixed_center` | 0 | Always pick segment center frame (no random offset) |
| `--over_sample` | 0_3_3_4_0_4_3_5_5 | Per-class oversampling multipliers |
| `--data_path` | ./data/EchoData/ | Data root directory |

## Architecture

```
Input clip (B, T, 3, 224, 224)
    │
    ├── SharedBackbone (frozen stem + conv layers)
    │   └── Feature maps (B*T, C', H', W')
    │
    ├── SpatialHead (center frame)
    │   └── Embedding (B, 128)
    │
    ├── TemporalHead (full clip)
    │   └── Conv blocks → LSTM → Embedding (B, 128)
    │
    └── Classifier
        └── Concat → FC → Logits → (B, 9)
```

## Data Augmentation (GPU)

Training transforms use torchvision v2 for GPU-accelerated augmentation with per-clip consistency:

1. **CPU**: Resize(256) → uint8 tensor
2. **GPU** (per-clip, shared random params):
   - ToDtype(float32, scale=True)
   - RandomResizedCrop(224, scale=(0.7, 1.0))
   - RandomRotation(±10°)
   - ColorJitter(brightness=0.2, contrast=0.2)
   - Normalize
