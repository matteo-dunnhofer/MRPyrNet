import torch
import torch.nn as nn
from torchvision import models

class AlexNet(nn.Module):
    
    def __init__(self): 
        super().__init__()

        self.model = models.alexnet(pretrained=True)
        self.features0 =  nn.Sequential(*list(self.model.features.children())[:3])
        self.features1 =  nn.Sequential(*list(self.model.features.children())[3:6])
        self.features2 =  nn.Sequential(*list(self.model.features.children())[6:8])
        self.features3 =  nn.Sequential(*list(self.model.features.children())[8:10])
        self.features4 =  nn.Sequential(*list(self.model.features.children())[10:])

        self.n_features0 = 64
        self.n_features1 = 192
        self.n_features2 = 384
        self.n_features3 = 256
        self.n_features4 = 256

    def forward(self, x):
        xf0 = self.features0(x)
        xf1 = self.features1(xf0)
        xf2 = self.features2(xf1)
        xf3 = self.features3(xf2)
        xf4 = self.features4(xf3)

        return xf0, xf1, xf2, xf3, xf4
