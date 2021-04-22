import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from torch.nn.init import kaiming_uniform_, kaiming_normal_
from ..modules.pdp import PyramidalDetailPooling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_deterministic(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weight_init(m, seed=2, init_type='uniform'):

    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):

        if init_type == 'normal':
            kaiming_normal_(m.weight)
        else:
            kaiming_uniform_(m.weight, a=math.sqrt(5))
    else:
        raise TypeError("cannnot initialize such weights")


def get_norm_layer(channels, norm_type='layer'):
    if norm_type == 'instance':     # contrast normalization
        layer = nn.GroupNorm(channels, channels)
    elif norm_type == 'batch':
        layer = nn.BatchNorm2d(channels)
    else:
        layer = nn.GroupNorm(1, channels)  # layer norm by default

    return layer


def conv_block(channels, kernel_size, dilation=1, repeats=2, normalization='layer', seed=2, init_type='uniform'):
    """
    :param channels: the input channel amount (same for output)
    :param kernel_size: 2D convolution kernel
    :param dilation: the dialation for the kernels of a conv block
    :param padding: amount of padding
    :param repeats: amount of repeats before added with identity
    :param normalization: the type of multi-slice normalization used
    :param seed: which seed of initial weights to use
    :param init_type: which type of Kaiming Init to use
    :return: nn.Sequential(for the given block)
    """

    conv_list = nn.ModuleList([])

    for i in range(repeats):
        conv2d = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                           dilation=dilation, stride=1, bias=False,
                           padding=(kernel_size + ((dilation - 1) * (kernel_size - 1))) // 2)
        weight_init(conv2d, seed=seed, init_type=init_type)
        conv_list.append(conv2d)

        #   Instance Normalization and Layer Normalization are just variations of Group Normalization
        #   https://pytorch.org/docs/stable/nn.html#groupnorm

        conv_list.append(get_norm_layer(channels, normalization))
        conv_list.append(nn.ReLU())

    return nn.Sequential(*conv_list)


def get_antialiasing_filter(kernel_size):
    """Get an integer specifying the 2D kernel size >>> returns a (1 x 1 x kernel_size x kernel_size)"""

    kernel_dict = {
        1: [[[[1.]]]],

        2: [[[[0.2500, 0.2500],
              [0.2500, 0.2500]]]],

        3: [[[[0.0625, 0.1250, 0.0625],
              [0.1250, 0.2500, 0.1250],
              [0.0625, 0.1250, 0.0625]]]],

        4: [[[[0.0156, 0.0469, 0.0469, 0.0156],
              [0.0469, 0.1406, 0.1406, 0.0469],
              [0.0469, 0.1406, 0.1406, 0.0469],
              [0.0156, 0.0469, 0.0469, 0.0156]]]],

        5: [[[[0.0039, 0.0156, 0.0234, 0.0156, 0.0039],
              [0.0156, 0.0625, 0.0938, 0.0625, 0.0156],
              [0.0234, 0.0938, 0.1406, 0.0938, 0.0234],
              [0.0156, 0.0625, 0.0938, 0.0625, 0.0156],
              [0.0039, 0.0156, 0.0234, 0.0156, 0.0039]]]],

        6: [[[[0.0010, 0.0049, 0.0098, 0.0098, 0.0049, 0.0010],
              [0.0049, 0.0244, 0.0488, 0.0488, 0.0244, 0.0049],
              [0.0098, 0.0488, 0.0977, 0.0977, 0.0488, 0.0098],
              [0.0098, 0.0488, 0.0977, 0.0977, 0.0488, 0.0098],
              [0.0049, 0.0244, 0.0488, 0.0488, 0.0244, 0.0049],
              [0.0010, 0.0049, 0.0098, 0.0098, 0.0049, 0.0010]]]],

        7: [[[[0.0002, 0.0015, 0.0037, 0.0049, 0.0037, 0.0015, 0.0002],
              [0.0015, 0.0088, 0.0220, 0.0293, 0.0220, 0.0088, 0.0015],
              [0.0037, 0.0220, 0.0549, 0.0732, 0.0549, 0.0220, 0.0037],
              [0.0049, 0.0293, 0.0732, 0.0977, 0.0732, 0.0293, 0.0049],
              [0.0037, 0.0220, 0.0549, 0.0732, 0.0549, 0.0220, 0.0037],
              [0.0015, 0.0088, 0.0220, 0.0293, 0.0220, 0.0088, 0.0015],
              [0.0002, 0.0015, 0.0037, 0.0049, 0.0037, 0.0015, 0.0002]]]]

    }

    if kernel_size in kernel_dict:
        return torch.Tensor(kernel_dict[kernel_size])
    else:
        raise ValueError('Unrecognized kernel size')


class BlurPool(nn.Module):

    def __init__(self, channels, stride, filter_size=5):
        super(BlurPool, self).__init__()
        self.channels = channels  # same input and output channels
        self.filter_size = filter_size
        self.stride = stride
        '''Kernel is a 1x5x5 kernel'''

        # repeat tensor from (1 x 1 x fs x fs) >>> (channels x 1 x fs x fs)
        self.kernel = nn.Parameter(get_antialiasing_filter(filter_size).repeat(self.channels, 1, 1, 1),
                                   requires_grad=False)

    def forward(self, x):
        """
        x is a tensor of dimension (batch, in_channels, height, width)
        - assume same input and output channels, and groups = 1
        - CURRENTLY DON'T SUPPORT PADDING
        """

        y = F.conv2d(input=x, weight=self.kernel, stride=self.stride, groups=self.channels)
        return y

    def to(self, dtype):
        self.kernel = self.kernel.to(dtype)
        return self


class ELNet(nn.Module):

    def __init__(self, **kwargs):

        super(ELNet, self).__init__()

        self.K = kwargs.get('K', 4)  # default K for ELNet
        self.norm_type = kwargs.get('norm_type', 'instance')  # default multi-slice normalization
        self.aa_filter = kwargs.get('aa_filter_size', 5)  # default aa-filter configuration
        self.weight_init_type = kwargs.get('weight_init_type', 'normal')  # type of weight initialization
        self.seed = kwargs.get('seed', 2)  # default seed for initialization
        self.num_classes = kwargs.get('num_classes', 2)  # number of classes for ELNet
        self.feature_dropout = kwargs.get('dropout', 0.0) # 0.0 for MRNet
        self.D = kwargs.get('D', 5)
        #self.detail_sizes_per_layer = kwargs.get('detail_sizes_per_layer', [[62], [29], [13], [5], [5]])

        make_deterministic(self.seed)

        if isinstance(self.aa_filter, int):
            aa_filter_size = [self.aa_filter] * 5

        self.channel_config = [4 * self.K, 8 * self.K, 16 * self.K, 16 * self.K, 16 * self.K]

        self.conv_1 = nn.Conv2d(1, self.channel_config[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.downsample_1 = BlurPool(channels=self.channel_config[0], stride=2, filter_size=aa_filter_size[0])
        self.norm_1 = get_norm_layer(self.channel_config[0], norm_type=self.norm_type)

        self.conv_2 = conv_block(self.channel_config[0], kernel_size=5, repeats=2, normalization=self.norm_type)
        self.conv_2_to_3 = nn.Conv2d(self.channel_config[0], self.channel_config[1], kernel_size=5, stride=1, padding=2,
                                     bias=False)
        self.downsample_2 = BlurPool(channels=self.channel_config[1], stride=2, filter_size=aa_filter_size[1])

        self.conv_3 = conv_block(self.channel_config[1], kernel_size=3, repeats=2, normalization=self.norm_type)
        self.conv_3_to_4 = nn.Conv2d(self.channel_config[1], self.channel_config[2], kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.downsample_3 = BlurPool(channels=self.channel_config[2], stride=2, filter_size=aa_filter_size[2])

        self.conv_4 = conv_block(self.channel_config[2], kernel_size=3, repeats=1, normalization=self.norm_type)
        self.conv_4_to_5 = nn.Conv2d(self.channel_config[2], self.channel_config[3], kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.downsample_4 = BlurPool(channels=self.channel_config[3], stride=2, filter_size=aa_filter_size[3])

        self.conv_5 = conv_block(self.channel_config[3], kernel_size=3, repeats=1, normalization=self.norm_type)
        self.conv_5_to_6 = nn.Conv2d(self.channel_config[3], self.channel_config[4], kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.downsample_5 = BlurPool(channels=self.channel_config[4], stride=2, filter_size=aa_filter_size[4])


        # compute PDP sub-region sizes

        # architectural hyper-params of ELNet's layers
        # (kernel size, stride, padding)
        arch = [(7, 2, 3), # conv_1
            (5, 2, 0), # downsample_1
            (5, 1, 2), # conv_2
            (5, 1, 2), 
            (5, 1, 2), # conv_2_to_3
            (5, 2, 0), # downsample_2
            (3, 1, 1), # conv_3
            (3, 1, 1), 
            (3, 1, 1), # conv_3_to_4
            (5, 2, 0), # dowsample_3
            (3, 1, 1), # conv4
            (3, 1, 1), # conv_4_to_5
            (5, 2, 0), # dowsample_4
            (3, 1, 1), # conv5
            (3, 1, 1)] # conv_5_to_6 

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
        detail_sizes_per_layer.append(detail_sizes_per_layer[-1])


        # FPN
        #self.conv_5_1x1 = conv_block(self.channel_config[4], kernel_size=1, repeats=1, normalization=self.norm_type)
        
        self.conv_6_to_7 = nn.Conv2d(self.channel_config[4], self.channel_config[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_7 = conv_block(self.channel_config[3], kernel_size=3, repeats=1, normalization=self.norm_type)
        self.conv_4_1x1 = conv_block(self.channel_config[3], kernel_size=1, repeats=1, normalization=self.norm_type)

        self.conv_7_to_8 = nn.Conv2d(self.channel_config[3], self.channel_config[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_8 = conv_block(self.channel_config[2], kernel_size=3, repeats=1, normalization=self.norm_type)
        self.conv_3_1x1 = conv_block(self.channel_config[2], kernel_size=1, repeats=1, normalization=self.norm_type)

        self.conv_8_to_9 = nn.Conv2d(self.channel_config[2], self.channel_config[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_9 = conv_block(self.channel_config[1], kernel_size=3, repeats=1, normalization=self.norm_type)
        self.conv_2_1x1 = conv_block(self.channel_config[1], kernel_size=1, repeats=1, normalization=self.norm_type)

        self.conv_9_to_10 = nn.Conv2d(self.channel_config[1], self.channel_config[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_10 = conv_block(self.channel_config[0], kernel_size=3, repeats=1, normalization=self.norm_type)
        self.conv_1_1x1 = conv_block(self.channel_config[0], kernel_size=1, repeats=1, normalization=self.norm_type)

        stride = 2
        blur_kernel_size = 5

        self.pdp5 = PyramidalDetailPooling(self.detail_sizes_per_layer[4])

        self.pdp4 = PyramidalDetailPooling(self.detail_sizes_per_layer[3])

        self.pdp3 = PyramidalDetailPooling(self.detail_sizes_per_layer[2])

        self.pdp2 = PyramidalDetailPooling(self.detail_sizes_per_layer[1])

        self.pdp1 = PyramidalDetailPooling(self.detail_sizes_per_layer[0])
   

        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.feature_dp = nn.Dropout(p=self.feature_dropout)

        self.fc1 = nn.Linear(self.channel_config[0] * len(self.detail_sizes_per_layer[0]), self.num_classes)
        self.fc2 = nn.Linear(self.channel_config[1] * len(self.detail_sizes_per_layer[1]), self.num_classes)
        self.fc3 = nn.Linear(self.channel_config[2] * len(self.detail_sizes_per_layer[2]), self.num_classes)
        self.fc4 = nn.Linear(self.channel_config[3] * len(self.detail_sizes_per_layer[3]), self.num_classes)
        self.fc5 = nn.Linear(self.channel_config[4] * len(self.detail_sizes_per_layer[4]), self.num_classes)

        weight_init(self.conv_1, self.seed, self.weight_init_type)
        weight_init(self.conv_2_to_3, self.seed, self.weight_init_type)
        weight_init(self.conv_3_to_4, self.seed, self.weight_init_type)
        weight_init(self.conv_4_to_5, self.seed, self.weight_init_type)
        weight_init(self.conv_5_to_6, self.seed, self.weight_init_type)
        weight_init(self.conv_6_to_7, self.seed, self.weight_init_type)
        weight_init(self.conv_7_to_8, self.seed, self.weight_init_type)
        weight_init(self.conv_8_to_9, self.seed, self.weight_init_type)
        weight_init(self.conv_9_to_10, self.seed, self.weight_init_type)
        weight_init(self.fc1, self.seed, self.weight_init_type)
        weight_init(self.fc2, self.seed, self.weight_init_type)
        weight_init(self.fc3, self.seed, self.weight_init_type)
        weight_init(self.fc4, self.seed, self.weight_init_type)
        weight_init(self.fc5, self.seed, self.weight_init_type)

    def feature_extraction(self, x):
        x = x.permute(1, 0, 2, 3)

        x1 = self.downsample_1(F.relu(self.norm_1(self.conv_1(x))))

        x = x1 + self.conv_2(x1)  # skip connection (survival rate 1 for first skip)
        x2 = self.downsample_2(F.relu(self.conv_2_to_3(x)))

        x = x2 + self.conv_3(x2)
        x3 = self.downsample_3(F.relu(self.conv_3_to_4(x)))

        x = x3 + self.conv_4(x3)
        x4 = self.downsample_4(F.relu(self.conv_4_to_5(x)))

        x = x4 + self.conv_5(x4)
        x5 = F.relu(self.conv_5_to_6(x))

        #print(x5.size(), x4.size(), x3.size(), x2.size(), x1.size())

        out_x5 = self.pdp5(x5)  # get [sx16Kx1x1]
        out_x5 = self.feature_dp(out_x5) 

        x = F.interpolate(x5, size=(x4.size(2), x4.size(3)), mode='bilinear')
        x = F.relu(self.conv_6_to_7(x))
        x = F.relu(self.conv_4_1x1(x4)) + F.relu(self.conv_7(x))

        out_x4 = self.pdp4(x)
        out_x4 = self.feature_dp(out_x4) 

        x = F.interpolate(x, size=(x3.size(2), x3.size(3)), mode='bilinear')
        x = F.relu(self.conv_7_to_8(x))
        x = F.relu(self.conv_3_1x1(x3)) + F.relu(self.conv_8(x))

        out_x3 = self.pdp3(x)
        out_x3 = self.feature_dp(out_x3) 

        x = F.interpolate(x, size=(x2.size(2), x2.size(3)), mode='bilinear')
        x = F.relu(self.conv_8_to_9(x))
        x = F.relu(self.conv_2_1x1(x2)) + F.relu(self.conv_9(x))

        out_x2 = self.pdp2(x)
        out_x2 = self.feature_dp(out_x2) 

        x = F.interpolate(x, size=(x1.size(2), x1.size(3)), mode='bilinear')
        x = F.relu(self.conv_9_to_10(x))
        x = F.relu(self.conv_1_1x1(x1)) + F.relu(self.conv_10(x))

        out_x1 = self.pdp1(x)
        out_x1 = self.feature_dp(out_x1) 

        return out_x1, out_x2, out_x3, out_x4, out_x5

    def forward(self, x):
        feats1, feats2, feats3, feats4, feats5 = self.feature_extraction(x)  # get [sx16Kx1x1]
        #print(feats1.size())
        #feats1, feats2, feats3, feats4, feats5 = feats1.unsqueeze(-1), feats2.unsqueeze(-1), feats3.unsqueeze(-1), feats4.unsqueeze(-1), feats5.squeeze(3)
        feats1, feats2, feats3, feats4, feats5 = feats1.squeeze(3), feats2.squeeze(3), feats3.squeeze(3), feats4.squeeze(3), feats5.squeeze(3)
        feats1, feats2, feats3, feats4, feats5 = feats1.permute(2, 1, 0), feats2.permute(2, 1, 0), feats3.permute(2, 1, 0), feats4.permute(2, 1, 0), feats5.permute(2, 1, 0)  # [1x16Kxs]
        #feats1, feats2, feats3, feats4, feats5 = feats1.permute(2, 1, 0), feats2.permute(2, 1, 1), feats3.permute(2, 0, 1), feats4.permute(2, 0, 1), feats5.permute(2, 0, 1)  # [1x16Kxs]
        #print(feats1.size())
        # classifier
        feats1, feats2, feats3, feats4, feats5 = self.max_pool(feats1).squeeze(2), self.max_pool(feats2).squeeze(2), self.max_pool(feats3).squeeze(2) , self.max_pool(feats4).squeeze(2), self.max_pool(feats5).squeeze(2) #1x16K]
        #print(feats1.size())
        scores1, scores2, scores3, scores4, scores5 = self.fc1(feats1), self.fc2(feats2), self.fc3(feats3), self.fc4(feats4), self.fc5(feats5)
        #print(scores1)
        return scores1, scores2, scores3, scores4, scores5
