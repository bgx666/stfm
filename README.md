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
scipy
scikit-learn
Pillow
opencv-python-headless
matplotlib
PyYAML
```

## Data Preparation

The **EV9V** dataset used in this project is publicly available on Hugging Face:

**🔗 [bgx666/EV9V](https://huggingface.co/datasets/bgx666/EV9V)**

> Download and place the extracted data under `./data/EchoData/` following the structure below.

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
python trainSelective.py --gpu 0 --batch_size 64 --batched_aug 1 --fast_eval 1
```

### Testing / Evaluation

```bash
python trainSelective.py --gpu 0 \
    --test_flag 1 --batched_aug 1 --fast_eval 1 \
    --resume ./save/resnet18/reedl_loss/model.ckpt
```

### Default Configuration

All defaults below reproduce the protocol used in the paper:

| Item | Value |
|------|-------|
| Optimizer | AdamW, `lr=2e-5`, `weight_decay=0.05` |
| LR schedule | 5-epoch linear warmup (`start_factor=0.1`) → cosine annealing (`T_max = epochs - 5`) |
| Exploration rate | `epsilon=0.8`, with `eps_warmup_epochs=5` (forced `epsilon=1.0` for epochs 1–5) |
| Loss | `reedl_loss` (Re-EDL), `lambda1=1.0`, `lambda2=0.8` |
| Sampling | `segment_size=20`, `clip_length=5`, `clip_interval=5`, `over_sample=0_3_3_4_0_4_3_5_5` |
| Backbone / head | ResNet-18, LSTM hidden 512, 2 layers, `embed_dims=128` |
| Evaluation | `T=10` uniformly sampled key frames, best-val checkpoint, early stop `patient=10` |
| Seeds | 100 / 200 / 300 for reported results (default `--seed 666`) |

Use `--batched_aug 1` for the paper's augmentation: geometric transforms
(RandomResizedCrop + RandomRotation) are shared within a clip and independent
across clips, while brightness/contrast jitter is applied per frame.
`--fast_eval 1` lazily loads only the frames used at evaluation time.

### Key Arguments

| Argument | Default | Description |
|----------|:-------:|-------------|
| `--model_name` | `resnet18` | Backbone: resnet18/50, convnext_*, densenet*, etc. |
| `--use_enhanced` | 1 | Use Enhanced STFM (shared backbone) |
| `--batch_size` | 64 | Training batch size |
| `--lr` | 2e-5 | Learning rate |
| `--weight_decay` | 0.05 | AdamW weight decay |
| `--epochs` | 100 | Max epochs |
| `--patient` | 10 | Early-stopping patience (epochs) |
| `--clip_length` | 5 | Frames per clip |
| `--clip_interval` | 5 | Sampling interval within clip |
| `--segment_size` | 20 | Segment size for uncertainty bank |
| `--selective` | 1 | Enable selective (uncertainty-guided) sampling |
| `--uncertainty` | 1 | Enable uncertainty estimation |
| `--epsilon` | 0.8 | Random exploration rate (0=fully greedy, 1=fully random) |
| `--eps_warmup_epochs` | 5 | Force epsilon=1.0 for the first N epochs |
| `--lamb2` | 0.8 | REEDL loss lambda parameter |
| `--temporal_hidden` | 512 | LSTM hidden size |
| `--temporal_layers` | 2 | LSTM layers |
| `--test_num_frames` | 10 | Key frames sampled per video at evaluation |
| `--batched_aug` | 0 | 1 = paper's per-frame GPU augmentation |
| `--fast_eval` | 0 | 1 = lazy val/test loader (load only used frames) |
| `--num_workers` | 12 | DataLoader workers |
| `--seed` | 666 | Random seed |
| `--fixed_center` | 0 | Always pick segment center frame (no random offset) |
| `--over_sample` | 0_3_3_4_0_4_3_5_5 | Per-class oversampling multipliers |
| `--save_dir` | ./save | Output directory for checkpoints and logs |
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

