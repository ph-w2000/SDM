import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import random
import numpy as np
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import einsum
        
## Adapted from https://github.com/joaomonteirof/e2e_antispoofing


class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = 6  # Number of attention heads
        
        # Define linear layers for query, key, and value projections
        self.query_projection = nn.Linear(in_channels, in_channels)
        self.key_projection = nn.Linear(in_channels, in_channels)
        self.value_projection = nn.Linear(in_channels, in_channels)
        
    def forward(self, input1, input2):
        batch_size, keypoints, height, width = input1.size()
        
        # Flatten the spatial dimensions
        input1_flat = input1.view(batch_size, keypoints, -1)  # Shape: [B, C, H*W]
        input2_flat = input2.view(batch_size, keypoints, -1)  # Shape: [B, C, H*W]
        
        # Project the input tensors
        query = self.query_projection(input1_flat)  # Shape: [B, C, H*W]
        key = self.key_projection(input2_flat)  # Shape: [B, C, H*W]
        value = self.value_projection(input2_flat)  # Shape: [B, C, H*W]
        
        # Reshape the query, key, and value tensors for multi-head attention
        query = query.view(batch_size, self.num_heads, self.in_channels // self.num_heads, -1)  # Shape: [B, num_heads, C//num_heads, H*W]
        key = key.view(batch_size, self.num_heads, self.in_channels // self.num_heads, -1)  # Shape: [B, num_heads, C//num_heads, H*W]
        value = value.view(batch_size, self.num_heads, self.in_channels // self.num_heads, -1)  # Shape: [B, num_heads, C//num_heads, H*W]
        
        # Perform multi-head cross-attention
        attention_scores = torch.matmul(query.permute(0, 1, 3, 2), key)  # Shape: [B, num_heads, H*W, H*W]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # Shape: [B, num_heads, H*W, H*W]
        context = torch.matmul(value, attention_weights.permute(0, 1, 3, 2))  # Shape: [B, num_heads, C//num_heads, H*W]
        
        # Reshape the context tensor
        context = context.view(batch_size, keypoints, height, width)  # Shape: [B, C, H, W]
        
        return context
    
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, mean_only=False):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        self.mean_only = mean_only

        init.kaiming_uniform_(self.att_weights)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        height = inputs.size(1)
        width = inputs.size(2)
        num_channels = inputs.size(3)

        reshaped_inputs = inputs.permute(0, 3, 1, 2).reshape(batch_size, num_channels, -1)  # Reshape to [B, C, N]

        weights = torch.bmm(reshaped_inputs.permute(0, 2, 1), self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

        if inputs.size(0) == 1:
            attentions = F.softmax(torch.tanh(weights), dim=1)
            weighted = torch.mul(reshaped_inputs, attentions.expand_as(reshaped_inputs))
        else:
            attentions = F.softmax(torch.tanh(weights.squeeze()), dim=1)
            weighted = torch.mul(reshaped_inputs, attentions.unsqueeze(1).expand_as(reshaped_inputs))

        weighted = weighted.reshape(batch_size, num_channels, height, width).permute(0, 2, 3, 1)  # Reshape back to [B, H, W, C]

        if self.mean_only:
            return weighted.sum(3)
        else:
            noise = 1e-5 * torch.randn(weighted.size())

            if inputs.is_cuda:
                noise = noise.to(inputs.device)

            # avg_repr, std_repr = weighted.sum(3), (weighted + noise).std(3)

            # representations = torch.cat((avg_repr, std_repr), 1)

            return weighted


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

RESNET_CONFIGS = {'18': [[2, 2, 2, 2], PreActBlock],
                  '28': [[3, 4, 6, 3], PreActBlock],
                  '34': [[3, 4, 6, 3], PreActBlock],
                  '50': [[3, 4, 6, 3], PreActBottleneck],
                  '101': [[3, 4, 23, 3], PreActBottleneck]
                  }

def setup_seed(random_seed, cudnn_deterministic=True):
    # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False

class ResNet(nn.Module):
    def __init__(self, num_nodes, resnet_type='18'):
        self.in_planes = 16
        super(ResNet, self).__init__()

        layers, block = RESNET_CONFIGS[resnet_type]

        self._norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(2, 16, kernel_size=(9, 9), stride=(3, 3), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv12 = nn.Conv2d(2, 16, kernel_size=(9, 9), stride=(3, 3), padding=(1, 1), bias=False)
        self.bn12 = nn.BatchNorm2d(16)

        self.activation = nn.ReLU()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.in_planes = 16

        self.layer12 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer22 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer32 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer42 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.conv52 = nn.Conv2d(512 * block.expansion, 512, kernel_size=(num_nodes, 6), stride=(1, 1), padding=(0, 1),
                               bias=False)
        
        self.bn52 = nn.BatchNorm2d(512)

        self.initialize_params()
        self.attention = SelfAttention(512)

        self.cs_attention = CrossAttention(26*33)


    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):

        x = self.conv1(x1)
        x = self.activation(self.bn1(x))
        x = self.layer1(x)
        hor_map = self.layer2(x)

        x = self.conv12(x2)
        x = self.activation(self.bn12(x))
        x = self.layer12(x)
        ver_map = self.layer22(x)
        h_v_attention = self.cs_attention(ver_map,hor_map)
        x = self.layer32(h_v_attention)
        x = self.layer42(x)
        x = self.conv52(x)
        x = self.activation(self.bn52(x))
        
        # stats = self.attention(x.permute(0, 2, 3, 1).contiguous())
        # x = stats.permute(0, 3, 1, 2).contiguous()

        return x
    
if __name__ == '__main__':
    mspn = ResNet(3, resnet_type='18')
    imgs = torch.randn(3, 2, 160, 200)
    imgs2 = torch.randn(3, 2, 160, 200)
    x = mspn(imgs, imgs2)
    print(x.shape)