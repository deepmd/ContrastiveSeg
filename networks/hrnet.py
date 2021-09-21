import torch
from torch import nn
import torch.nn.functional as F

from .hrnet_backbone import HighResolutionNet
from .hrnet_config import MODEL_CONFIGS
from .modules import BNReLU, ProjectionHead


class HRNet_W48(nn.Module):
    def __init__(self, num_classes, bn_type='torchbn'):
        super(HRNet_W48, self).__init__()
        self.num_classes = num_classes
        self.backbone = HighResolutionNet(MODEL_CONFIGS['hrnet48'],
                                          bn_type=bn_type,
                                          bn_momentum=0.1)
        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            BNReLU(in_channels, bn_type=bn_type),
            nn.Dropout2d(0.10),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out = self.cls_head(feats)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return out


class HRNet_W48_CONTRAST(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019
    """

    def __init__(self, num_classes, bn_type='torchbn', proj_dim=256, ):
        super(HRNet_W48_CONTRAST, self).__init__()
        self.num_classes = num_classes
        self.backbone = HighResolutionNet(MODEL_CONFIGS['hrnet48'],
                                          bn_type=bn_type,
                                          bn_momentum=0.1)

        self.proj_dim = proj_dim

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            BNReLU(in_channels, bn_type=bn_type),
            nn.Dropout2d(0.10),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.proj_head = ProjectionHead(dim_in=in_channels, proj_dim=self.proj_dim)

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out = self.cls_head(feats)

        emb = self.proj_head(feats)
        return {'seg': out, 'embed': emb}
