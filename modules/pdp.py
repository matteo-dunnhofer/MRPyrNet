import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PyramidalDetailPooling(nn.Module):

    def __init__(self, detail_sizes):
        super().__init__()
        
        self.detail_sizes = detail_sizes
        self.D = len(self.detail_sizes)

    def forward(self, x):
        feats = []
        _, d, h, w = x.size()

        if (w % 2) == 1:
            cx = round((w-1) / 2)
        else:
            cx = round(w / 2)

        if (h % 2) == 1:
            cy = round((h-1) / 2)
        else:
            cy = round(h / 2)

        for d in range(self.D):
            
            ##pw, ph = self.sizes[d], self.sizes[d]
            #pw, ph = self.sizes[d]

            x1, x2 = cx - math.floor(self.detail_sizes[d] / 2), cx + math.ceil(self.detail_sizes[d] / 2)
            y1, y2 = cy - math.floor(self.detail_sizes[d] / 2), cy + math.ceil(self.detail_sizes[d] / 2)

            pooled_feats = F.adaptive_avg_pool2d(x[:,:,y1:y2,x1:x2], 1)

            feats.append(pooled_feats)

            """
            if (((py2 - py1) ** 2) > 0) and (((px2 - px1) ** 2) > 0):
                pf = F.adaptive_avg_pool2d(x[:,:,y1:y2,x1:x2], 1)
                

                #pf = F.relu(self.convs[l](pf))
                
            else:
                pf = torch.zeros((x.size(0), self.out_feat, 1, 1)).to(x.get_device())
            """

            
            #print(pf.shape)
            #pf = pf.view(x.size(0), -1)
            #print(pf.shape)

            

        x = torch.cat(feats, 1)

        return x

