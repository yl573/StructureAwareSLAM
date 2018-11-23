import torch
from torch import nn

## Conv + bn + relu
class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, mode='conv'):
        super(ConvBlock, self).__init__()
       
        if padding == None:
            padding = (kernel_size - 1) // 2
            pass
        if mode == 'conv':
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        elif mode == 'deconv':
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        elif mode == 'conv_3d':
            self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        elif mode == 'deconv_3d':
            self.conv = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        else:
            print('conv mode not supported', mode)
            exit(1)
            pass
        if '3d' not in mode:
            self.bn = nn.BatchNorm2d(out_planes)
        else:
            self.bn = nn.BatchNorm3d(out_planes)
            pass
        self.relu = nn.ReLU(inplace=True)
        return
   
    def forward(self, inp):
        #return self.relu(self.conv(inp))       
        return self.relu(self.bn(self.conv(inp)))

## The pyramid module from pyramid scene parsing
class PyramidModule(nn.Module):
    def __init__(self, options, in_planes, middle_planes, scales=[32, 16, 8, 4]):
        super(PyramidModule, self).__init__()
        
        self.pool_1 = torch.nn.AvgPool2d((int(scales[0] * options.height / options.width), scales[0]))
        self.pool_2 = torch.nn.AvgPool2d((int(scales[1] * options.height / options.width), scales[1]))
        self.pool_3 = torch.nn.AvgPool2d((int(scales[2] * options.height / options.width), scales[2]))
        self.pool_4 = torch.nn.AvgPool2d((int(scales[3] * options.height / options.width), scales[3]))
        self.conv_1 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.conv_2 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.conv_3 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.conv_4 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.upsample = torch.nn.Upsample(size=(int(scales[0] * options.height / options.width), scales[0]), mode='bilinear')
        return
    
    def forward(self, inp):
        x_1 = self.upsample(self.conv_1(self.pool_1(inp)))
        x_2 = self.upsample(self.conv_2(self.pool_2(inp)))
        x_3 = self.upsample(self.conv_3(self.pool_3(inp)))
        x_4 = self.upsample(self.conv_4(self.pool_4(inp)))
        out = torch.cat([inp, x_1, x_2, x_3, x_4], dim=1)
        return out



