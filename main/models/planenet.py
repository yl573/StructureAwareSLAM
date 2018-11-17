from main.models.drn import drn_d_22
from main.models.modules import *


class PlaneNet(nn.Module):
    def __init__(self, options):
        super(PlaneNet, self).__init__()

        self.options = options
        drn_channels = (4, 8, 16, 32, 64, 64, 64, 64)
        drn_out_dim = drn_channels[-1]
        drn_out_map = 32

        self.drn = drn_d_22(pretrained=False, out_map=drn_out_map, num_classes=-1,
                            out_middle=False, channels=drn_channels)
        # self.pool = torch.nn.AvgPool2d((int(32 * options.height / options.width), 32))
        self.plane_pred = nn.Linear(drn_out_dim, options.numOutputPlanes * 3)

        pyr_mid_planes = 32
        self.pyramid = PyramidModule(options, drn_out_dim, pyr_mid_planes)

        pyr_planes = drn_out_dim + pyr_mid_planes * 4
        feat_planes = 64
        self.feature_conv = ConvBlock(pyr_planes, feat_planes)

        self.segmentation_pred = nn.Conv2d(feat_planes, options.numOutputPlanes + 1, kernel_size=1)
        self.depth_pred = nn.Conv2d(feat_planes, 1, kernel_size=1)
        self.upsample = torch.nn.Upsample(size=(options.outputHeight, options.outputWidth), mode='bilinear')
        return

    def forward(self, inp):
        batch_dim = inp.shape[0]
        features = self.drn(inp)
        features_pool = features.mean(3).mean(2)
        planes = self.plane_pred(features_pool).view((batch_dim, self.options.numOutputPlanes, 3))
        features = self.pyramid(features)
        features = self.feature_conv(features)
        segmentation = self.upsample(self.segmentation_pred(features))
        depth = self.upsample(self.depth_pred(features))
        return planes, segmentation, depth
