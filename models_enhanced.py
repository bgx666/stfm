
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BasicBlock(nn.Module):
    """ BasicBlock"""
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SharedBackbone(nn.Module):
    """
     Stem
    CNN :  conv1~maxpool / features[0]
    ViT / EfficientNet : Identity stem
    """
    def __init__(self, backbone='resnet50'):
        super().__init__()
        self.backbone = backbone.lower()

        if self.backbone == 'resnet18':
            resnet = models.resnet18(weights='DEFAULT')
            self.stem = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1,
            )

            self.output_channels = 64
            self.output_stride = 4
        elif self.backbone == 'resnet50':
            resnet = models.resnet50(weights='DEFAULT')
            self.stem = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1,
            )

            self.output_channels = 256     # layer1  256ch
            self.output_stride = 4
        elif self.backbone.startswith('convnext_'):
            convnext_cls = getattr(models, f'convnext_{self.backbone.split("_")[1]}')
            convnext = convnext_cls(weights='DEFAULT')
            self.stem = nn.Sequential(
                convnext.features[0],  # stem
                convnext.features[1],  # first stage CNBlocks
            )

            self.output_channels = {'tiny': 96, 'small': 96, 'base': 128, 'large': 192}.get(self.backbone.split('_')[1], 96)
            self.output_stride = 4
        elif self.backbone.startswith('densenet'):
            depth = int(self.backbone.replace('densenet', ''))
            densenet_cls = getattr(models, f'densenet{depth}')
            densenet = densenet_cls(weights='DEFAULT')
            self.stem = nn.Sequential(
                densenet.features.conv0, densenet.features.norm0,
                densenet.features.relu0, densenet.features.pool0,
                densenet.features.denseblock1,
            )

            self.output_channels = 256
            self.output_stride = 4
        elif self.backbone.startswith('mobilenet_v3'):
            mobilenet = getattr(models, self.backbone)(weights='DEFAULT')
            self.stem = nn.Sequential(
                mobilenet.features[0],  # stem conv
                mobilenet.features[1],  # first InvertedResidual
            )

            self.output_channels = 16
            self.output_stride = 4
        elif self.backbone in ('efficientnet_v2_s', 'efficientnet_v2_m'):
            enet = getattr(models, self.backbone)(weights='DEFAULT')
            self.stem = nn.Sequential(
                enet.features[0],  # stem conv
                enet.features[1],  # first stage MBConv
            )

            self.output_channels = 24
            self.output_stride = 4
        elif self.backbone.startswith('vit_') or self.backbone.startswith('efficientnet_'):
            self.stem = nn.Identity()
            self.output_channels = 3
            self.output_stride = 1
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        x = self.stem(x)
        return x

class SpatialHead(nn.Module):
    """
     Head ResNet / ConvNeXt / ViT / EfficientNet
    CNN :  stages + avgpool + fc
    ViT / EfficientNet :  head
    """
    def __init__(self, backbone='resnet50', embed_dims=128):
        super().__init__()
        self.backbone = backbone.lower()
        self.use_full_model = False

        if self.backbone == 'resnet18':
            resnet = models.resnet18(weights='DEFAULT')
            self.features = nn.Sequential(
                resnet.layer2,      # 64 -> 128 (layer1 )
                resnet.layer3,    # 128 -> 256
                resnet.layer4,    # 256 -> 512
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, embed_dims)
            self.input_channels = 64
        elif self.backbone == 'resnet50':
            resnet = models.resnet50(weights='DEFAULT')
            self.features = nn.Sequential(
                resnet.layer2,      # 256 -> 512 ( layer2 layer1 )
                resnet.layer3,    # 512 -> 1024
                resnet.layer4,    # 1024 -> 2048
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, embed_dims)
            self.input_channels = 256
        elif self.backbone == 'convnext_tiny':
            convnext = models.convnext_tiny(weights='DEFAULT')
            self.features = nn.Sequential(
                convnext.features[2],
                convnext.features[3],
                convnext.features[4],
                convnext.features[5],
                convnext.features[6],
                convnext.features[7],
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(768, embed_dims)
            self.input_channels = 96
        elif self.backbone == 'convnext_small':
            convnext = models.convnext_small(weights='DEFAULT')
            self.features = nn.Sequential(
                convnext.features[2],
                convnext.features[3],
                convnext.features[4],
                convnext.features[5],
                convnext.features[6],
                convnext.features[7],
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(768, embed_dims)
            self.input_channels = 96
        elif self.backbone == 'densenet121':
            densenet = models.densenet121(weights='DEFAULT')
            self.features = nn.Sequential(
                densenet.features.transition1,
                densenet.features.denseblock2,
                densenet.features.transition2,
                densenet.features.denseblock3,
                densenet.features.transition3,
                densenet.features.denseblock4,
                densenet.features.norm5,
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(1024, embed_dims)
            self.input_channels = 256
        elif self.backbone == 'densenet169':
            densenet = models.densenet169(weights='DEFAULT')
            self.features = nn.Sequential(
                densenet.features.transition1,
                densenet.features.denseblock2,
                densenet.features.transition2,
                densenet.features.denseblock3,
                densenet.features.transition3,
                densenet.features.denseblock4,
                densenet.features.norm5,
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(1664, embed_dims)
            self.input_channels = 256
        elif self.backbone == 'mobilenet_v3_small':
            mobilenet = models.mobilenet_v3_small(weights='DEFAULT')
            self.features = mobilenet.features[2:]
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(576, embed_dims)
            self.input_channels = 16
        elif self.backbone == 'mobilenet_v3_large':
            mobilenet = models.mobilenet_v3_large(weights='DEFAULT')
            self.features = mobilenet.features[2:]
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(960, embed_dims)
            self.input_channels = 16
        elif self.backbone == 'vit_tiny':
            import timm
            self.model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
            self.model.head = nn.Linear(self.model.num_features, embed_dims)
            self.use_full_model = True
        elif self.backbone == 'vit_b_16':
            self.model = models.vit_b_16(weights='DEFAULT')
            self.model.heads.head = nn.Linear(768, embed_dims)
            self.use_full_model = True
        elif self.backbone == 'vit_b_32':
            self.model = models.vit_b_32(weights='DEFAULT')
            self.model.heads.head = nn.Linear(768, embed_dims)
            self.use_full_model = True
        elif self.backbone == 'vit_l_16':
            self.model = models.vit_l_16(weights='DEFAULT')
            self.model.heads.head = nn.Linear(1024, embed_dims)
            self.use_full_model = True
        elif self.backbone == 'vit_l_32':
            self.model = models.vit_l_32(weights='DEFAULT')
            self.model.heads.head = nn.Linear(1024, embed_dims)
            self.use_full_model = True
        elif self.backbone == 'efficientnet_v2_s':
            enet = models.efficientnet_v2_s(weights='DEFAULT')
            self.features = enet.features[2:]
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(1280, embed_dims)
            self.input_channels = 24
        elif self.backbone == 'efficientnet_v2_m':
            enet = models.efficientnet_v2_m(weights='DEFAULT')
            self.features = enet.features[2:]
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(1280, embed_dims)
            self.input_channels = 24
        elif self.backbone == 'efficientnet_v2_l':
            self.model = models.efficientnet_v2_l(weights='DEFAULT')
            self.model.classifier[1] = nn.Linear(1280, embed_dims)
            self.use_full_model = True
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        if self.use_full_model:
            return self.model(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc(x)
        return x

class TemporalHead(nn.Module):
    """
     Head +
    proj: 5x5 stride=2, block1: 5x5 stride=2, block2: 3x3 stride=2
    """
    def __init__(self, in_channels=64, hidden_size=512, num_layers=2, embed_dims=128):
        super().__init__()

        # proj: 5x5, stride=2, 256 -> 32  (56x56 -> 28x28)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=False),
        )

        # Block 1: 5x5, stride=2, 32 -> 48  (28x28 -> 14x14)
        self.conv1_1 = nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(48)
        self.conv1_2 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(48)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(48),
        )

        # Block 2: 3x3, stride=2, 48 -> 64  (14x14 -> 7x7)
        self.conv2_1 = nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.down2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        lstm_dropout = 0.3 if num_layers > 1 else 0.0
        self.lstm_model = nn.LSTM(64, hidden_size, num_layers,
                                   batch_first=True, dropout=lstm_dropout)
        self.fc = nn.Linear(hidden_size, embed_dims)

    def _block1(self, x):
        identity = self.down1(x)
        x = F.relu(self.bn1_1(self.conv1_1(x)), inplace=False)
        x = self.bn1_2(self.conv1_2(x))
        return F.relu(x + identity, inplace=False)

    def _block2(self, x):
        identity = self.down2(x)
        x = F.relu(self.bn2_1(self.conv2_1(x)), inplace=False)
        x = self.bn2_2(self.conv2_2(x))
        return F.relu(x + identity, inplace=False)

    def forward(self, inputs):
        b, l, c, w, h = inputs.size()
        x = inputs.view(b * l, c, w, h)

        if self.training:
            x = torch.utils.checkpoint.checkpoint(self.proj, x, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._block1, x, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._block2, x, use_reentrant=False)
        else:
            x = self.proj(x)
            x = self._block1(x)
            x = self._block2(x)

        x = self.avgpool(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.view(b, l, 64)

        lstm_features, _ = self.lstm_model(x)
        lstm_features = F.dropout(lstm_features[:, -1], p=0.3, training=self.training)
        output = self.fc(lstm_features)
        return output

class EnhancedSTFM(nn.Module):
    """
     STFM ResNet / ConvNeXt / ViT / EfficientNet
    """
    def __init__(self, num_classes=9, embed_dims=128,
                 hidden_size=512, num_layers=2,
                 backbone='resnet50'):
        super().__init__()

        self.backbone = backbone.lower()
        self.embed_dims = embed_dims

        self.shared_backbone = SharedBackbone(backbone)

        self.space_model = SpatialHead(backbone, embed_dims=embed_dims)

        temporal_in_channels = self.shared_backbone.output_channels
        self.temporal_model = TemporalHead(
            in_channels=temporal_in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            embed_dims=embed_dims
        )

        # Concat →
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dims * 2, embed_dims),   # 256 -> 128
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embed_dims, num_classes)        # 128 -> 9
        )

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.LSTM):
                for name_param, param in m.named_parameters():
                    if 'weight_ih' in name_param:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name_param:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name_param:
                        nn.init.zeros_(param)
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.0)

    def forward(self, inputs):
        frame, clip = inputs
        B, T = clip.shape[0], clip.shape[1]

        #  clip
        clip_flat = clip.reshape(B * T, *clip.shape[2:])
        shared_feats = self.shared_backbone(clip_flat)
        _, C, H, W = shared_feats.shape
        shared_feats = shared_feats.reshape(B, T, C, H, W)

        center_idx = T // 2
        key_frame_feat = shared_feats[:, center_idx]
        space_embedding = self.space_model(key_frame_feat)

        temporal_embedding = self.temporal_model(shared_feats)

        # Concat →
        fused = torch.cat([space_embedding, temporal_embedding], dim=1)
        output = self.classifier(fused)
        return output

def create_enhanced_stfm(num_classes=9, embed_dims=128,
                         hidden_size=512, num_layers=2,
                         backbone='resnet50', **kwargs):
    """
     STFM

    Args:
        backbone: 'resnet18', 'resnet50', 'convnext_tiny', 'convnext_small',
                  'convnext_base', 'convnext_large',
                  'vit_tiny', 'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32',
                  'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'
    """
    model = EnhancedSTFM(
        num_classes=num_classes,
        embed_dims=embed_dims,
        hidden_size=hidden_size,
        num_layers=num_layers,
        backbone=backbone
    )
    return model
