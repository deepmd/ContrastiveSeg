from torch import nn
import torch.nn.functional as F


def BNReLU(num_features, bn_type='torchbn', **kwargs):
    if bn_type == 'torchbn':
        return nn.Sequential(
            nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )


def BatchNorm2d(bn_type='torchbn'):
    if bn_type == 'torchbn':
        return nn.BatchNorm2d


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                BNReLU(dim_in, bn_type=bn_type),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)
