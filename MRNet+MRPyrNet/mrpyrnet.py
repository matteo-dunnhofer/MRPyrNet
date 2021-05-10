import sys
sys.path.append('..')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from modules.alexnet import AlexNet
from modules.pdp import PyramidalDetailPooling


class MRPyrNet(nn.Module):

    def __init__(self, D, slice_size=256):
        super().__init__()
        
        self.D = D

        self.backbone = AlexNet()

        # compute PDP sub-region sizes

        # architectural hyper-params of AlexNet's layers
        # (kernel size, stride, padding)
        arch = [(11, 4, 2), # conv_1
            (3, 2, 0), # maxpool 1
            (5, 1, 2), # conv_2
            (3, 2, 0), # maxpool 2
            (3, 1, 1), # conv_3
            (3, 1, 1), # conv 4
            (3, 2, 0)] # maxpool 3

        factors = [1.0 - (ff * (1.0 / self.D)) for ff in range(self.D)]

        slice_level_sizes = [int(slice_size * factor) for factor in factors]
        slice_level_sizes = [size if (size % 2) == 0 else size+1 for size in slice_level_sizes]

        detail_sizes = [slice_level_sizes]

        n_layers = len(arch)
        for l in range(n_layers):
            kernel_size, stride, padding = arch[l]

            # base feature map size
            base_size = math.floor((float(detail_sizes[-1][0] - kernel_size + 2*padding) / stride) + 1)

            layer_detail_sizes = []
            for size in detail_sizes[-1]:

                new_size = (float(size - kernel_size + 2*padding) / stride) + 1

                new_size_f = math.floor(new_size)

                if base_size % 2 == 0:
                    if new_size_f % 2 == 0:
                        new_size = new_size_f
                    else:
                        new_size = new_size_f+1
                else:
                    if new_size_f % 2 == 1:
                        new_size = new_size_f
                    else:
                        new_size = new_size_f+1
               
                if (new_size < 1) or ((base_size % 2 == 0) and new_size == 1): 
                    break
                
                layer_detail_sizes.append(new_size)

            # remove duplicate regions
            final_layer_detail_sizes = [] 
            for ls in layer_detail_sizes: 
                if not ls in final_layer_detail_sizes:
                    final_layer_detail_sizes.append(ls)

            detail_sizes.append(final_layer_detail_sizes)

        # remove slice level sizes
        del detail_sizes[0]

        self.detail_sizes_per_layer = []
        for ps in detail_sizes:
            if all(ps != fps for fps in self.detail_sizes_per_layer): # check list duplicate
                self.detail_sizes_per_layer.append(ps)
        
        # remove sizes after first conv layer
        del self.detail_sizes_per_layer[0]

        # copy one to last layer
        self.detail_sizes_per_layer.insert(-2, self.detail_sizes_per_layer[-2])
        self.detail_sizes_per_layer.insert(-2, self.detail_sizes_per_layer[-2])

        print(self.detail_sizes_per_layer)

        # FPN modules
        self.conv_4_to_3 = nn.Conv2d(self.backbone.n_features4, self.backbone.n_features3, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3_1x1 = nn.Conv2d(self.backbone.n_features3, self.backbone.n_features3, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(self.backbone.n_features3, self.backbone.n_features3, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_3_to_2 = nn.Conv2d(self.backbone.n_features3, self.backbone.n_features2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2_1x1 = nn.Conv2d(self.backbone.n_features2, self.backbone.n_features2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.backbone.n_features2, self.backbone.n_features2, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_2_to_1 = nn.Conv2d(self.backbone.n_features2, self.backbone.n_features1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1_1x1 = nn.Conv2d(self.backbone.n_features1, self.backbone.n_features1, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(self.backbone.n_features1, self.backbone.n_features1, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_1_to_0 = nn.Conv2d(self.backbone.n_features1, self.backbone.n_features0, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv0_1x1 = nn.Conv2d(self.backbone.n_features0, self.backbone.n_features0, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv0 = nn.Conv2d(self.backbone.n_features0, self.backbone.n_features0, kernel_size=3, stride=1, padding=1, bias=False)


        # PDP modules 
        self.pdp4 = PyramidalDetailPooling(self.detail_sizes_per_layer[4])

        self.pdp3 = PyramidalDetailPooling(self.detail_sizes_per_layer[3])

        self.pdp2 = PyramidalDetailPooling(self.detail_sizes_per_layer[2])

        self.pdp1 = PyramidalDetailPooling(self.detail_sizes_per_layer[1])

        self.pdp0 = PyramidalDetailPooling(self.detail_sizes_per_layer[0])


        self.classifier0 = nn.Linear(self.backbone.n_features0 * len(self.detail_sizes_per_layer[0]), 1)
        self.classifier1 = nn.Linear(self.backbone.n_features1 * len(self.detail_sizes_per_layer[1]), 1)
        self.classifier2 = nn.Linear(self.backbone.n_features2 * len(self.detail_sizes_per_layer[2]), 1)
        self.classifier3 = nn.Linear(self.backbone.n_features3 * len(self.detail_sizes_per_layer[3]), 1)
        self.classifier4 = nn.Linear(self.backbone.n_features4 * len(self.detail_sizes_per_layer[4]), 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported

        xf0, xf1, xf2, xf3, xf4 = self.backbone(x)

        out_xf3 = F.interpolate(xf4, size=(xf3.size(2), xf3.size(3)), mode='bilinear')
        out_xf3 = F.relu(self.conv_4_to_3(out_xf3))
        out_xf3 = F.relu(self.conv3_1x1(xf3)) + F.relu(self.conv3(out_xf3))

        out_xf2 = F.interpolate(out_xf3, size=(xf2.size(2), xf2.size(3)), mode='bilinear')
        out_xf2 = F.relu(self.conv_3_to_2(out_xf2))
        out_xf2 = F.relu(self.conv2_1x1(xf2)) + F.relu(self.conv2(out_xf2))

        out_xf1 = F.interpolate(out_xf2, size=(xf1.size(2), xf1.size(3)), mode='bilinear')
        out_xf1 = F.relu(self.conv_2_to_1(out_xf1))
        out_xf1 = F.relu(self.conv1_1x1(xf1)) + F.relu(self.conv1(out_xf1))

        out_xf0 = F.interpolate(out_xf1, size=(xf0.size(2), xf0.size(3)), mode='bilinear')
        out_xf0 = F.relu(self.conv_1_to_0(out_xf0))
        out_xf0 = F.relu(self.conv0_1x1(xf0)) + F.relu(self.conv0(out_xf0))
        

        x0 = self.pdp0(out_xf0)
        x1 = self.pdp1(out_xf1)
        x2 = self.pdp2(out_xf2)
        x3 = self.pdp3(out_xf3)
        x4 = self.pdp4(xf4)

        x0 = x0.squeeze()
        x1 = x1.squeeze()
        x2 = x2.squeeze()
        x3 = x3.squeeze()
        x4 = x4.squeeze()

        x0 = torch.max(x0, 0, keepdim=True)[0]
        x1 = torch.max(x1, 0, keepdim=True)[0]
        x2 = torch.max(x2, 0, keepdim=True)[0]
        x3 = torch.max(x3, 0, keepdim=True)[0]
        x4 = torch.max(x4, 0, keepdim=True)[0]

        x0 = self.classifier0(x0)
        x1 = self.classifier1(x1)
        x2 = self.classifier2(x2)
        x3 = self.classifier3(x3)
        x4 = self.classifier4(x4)

        return x0, x1, x2, x3, x4
