from main.models.drn import drn_d_22, drn_d_54
from main.models.modules import *

from torch import nn
import torch.nn.functional as F


class SegmentationDecoder(nn.Module):

    def __init__(self, feat_planes, output_planes):
        super().__init__()

        mid_planes = int((output_planes + feat_planes) / 2)

        self.segmentation_pred = nn.Sequential(
            nn.Conv2d(feat_planes, feat_planes, kernel_size=1),
            nn.ConvTranspose2d(feat_planes, mid_planes, kernel_size=2),
            nn.Conv2d(mid_planes, mid_planes, kernel_size=1),
            nn.ConvTranspose2d(mid_planes, output_planes, kernel_size=2),
            nn.Conv2d(output_planes, output_planes, kernel_size=1),
        )

        self.segmentation_pred = nn.Conv2d(feat_planes, output_planes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.segmentation_pred(x)
        x = self.softmax(x)
        return x


class PlaneNet(nn.Module):
    def __init__(self, options):
        super(PlaneNet, self).__init__()

        self.options = options
        drn_out_dim = options.drn_channels[-1]

        self.drn = drn_d_54(pretrained=True, out_map=options.drn_out_map, num_classes=-1,
                            out_middle=False, channels=options.drn_channels)
        # self.pool = torch.nn.AvgPool2d((int(32 * options.height / options.width), 32))
        self.plane_pred = nn.Linear(drn_out_dim, options.numOutputPlanes * 3)

        self.pyramid = PyramidModule(options, drn_out_dim, options.pyr_mid_planes)

        pyr_planes = drn_out_dim + options.pyr_mid_planes * 4

        self.feature_conv = ConvBlock(pyr_planes, options.feat_planes)

        self.segmentation_pred = SegmentationDecoder(options.feat_planes, options.numOutputPlanes + 1)
        self.depth_pred = nn.Conv2d(options.feat_planes, 1, kernel_size=1)
        self.upsample_size = (options.outputHeight, options.outputWidth)

    def forward(self, inp):
        batch_dim = inp.shape[0]
        features = self.drn(inp)
        # global average pooling
        features_pool = features.mean(3).mean(2)

        planes = self.plane_pred(features_pool).view((batch_dim, self.options.numOutputPlanes, 3))

        features = self.pyramid(features)
        features = self.feature_conv(features)

        seg_pred = self.segmentation_pred(features)
        segmentation = F.interpolate(seg_pred, size=self.upsample_size, mode='nearest')

        depth = self.depth_pred(features)
        depth = F.interpolate(depth, size=self.upsample_size, mode='bilinear')
        return planes, segmentation, depth
