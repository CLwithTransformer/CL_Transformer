from math import ceil
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.functional import relu
# helpers
from torch.nn.utils import spectral_norm
from torch.nn import init
import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module


class CosineClassifier(Module):
    def __init__(self, in_features, n_classes, sigma=True):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = n_classes
        self.weight = Parameter(torch.Tensor(n_classes, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  #for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, l = 3):
    val = val if isinstance(val, tuple) else (val,)
    return (*val, *((val[-1],) * max(l - len(val), 0)))


def always(val):
    return lambda *args, **kwargs: val


# ResNet18
class Bottleneck(nn.Module):
    expansion = 4  # # output cahnnels / # input channels

    def __init__(self, inplanes, outplanes, stride=1):
        assert outplanes % self.expansion == 0
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.bottleneck_planes = int(outplanes / self.expansion)
        self.stride = stride

        self._make_layer()

    def _make_layer(self):
        # conv 1x1
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.conv1 = nn.Conv2d(self.inplanes, self.bottleneck_planes, kernel_size=1, stride=self.stride, bias=False)
        # conv 3x3
        self.bn2 = nn.BatchNorm2d(self.bottleneck_planes)
        self.conv2 = nn.Conv2d(self.bottleneck_planes, self.bottleneck_planes, kernel_size=3, stride=1, padding=1, bias=False)
        # conv 1x1
        self.bn3 = nn.BatchNorm2d(self.bottleneck_planes)
        self.conv3 = nn.Conv2d(self.bottleneck_planes, self.outplanes, kernel_size=1,
                               stride=1)
        if self.inplanes != self.outplanes:
            self.shortcut = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1,
                                      stride=self.stride, bias=False)
        else:
            self.shortcut = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        # we do pre-activation
        out = self.relu(self.bn1(x))
        out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.conv2(out)

        out = self.relu(self.bn3(out))
        out = self.conv3(out)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out += residual
        return out


class ResNet164(nn.Module):
    def __init__(self):
        super(ResNet164, self).__init__()
        nstages = [16, 64, 128, 256]
        # one conv at the beginning (spatial size: 32x32)
        self.conv1 = nn.Conv2d(3, nstages[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        depth = 164
        block = Bottleneck
        n = int((depth - 2) / 9)
        # use `block` as unit to construct res-net
        # Stage 0 (spatial size: 32x32)
        self.layer1 = self._make_layer(block, nstages[0], nstages[1], n)
        # Stage 1 (spatial size: 32x32)
        self.layer2 = self._make_layer(block, nstages[1], nstages[2], n, stride=2)
        # Stage 2 (spatial size: 16x16)
        self.layer3 = self._make_layer(block, nstages[2], nstages[3], n, stride=2)
        # Stage 3 (spatial size: 8x8)
        self.bn = nn.BatchNorm2d(nstages[3])
        self.relu = nn.ReLU(inplace=True)

        # weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, inplanes, outplanes, nstage, stride=1):
        layers = []
        layers.append(block(inplanes, outplanes, stride))
        for i in range(1, nstage):
            layers.append(block(outplanes, outplanes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.bn(x))

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x
####################################################################################################


# ResNet18
def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
            nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out, inplace=True)
        return out


class ResNet18Pre(nn.Module):
    def __init__(self, nf=32, stages=3):
        super(ResNet18Pre, self).__init__()
        self.stages = stages
        self.in_planes = nf
        self.block = BasicBlock
        num_blocks = [2, 2, 2, 2]
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, nf * 8, num_blocks[3], stride=2)
        self._resnet_high = nn.Sequential(
                                          self.layer4,
                                          nn.Identity()
                                          )
        if nf == 64:
            if self.stages == 3:
                self.resnet_low = nn.Sequential(self.conv1,
                                                self.bn1,
                                                self.relu,
                                                self.layer1,  # 64, 32, 32
                                                self.layer2,  # 128, 16, 16
                                                # self.layer3,  # 256, 8, 8
                                                )
            if self.stages == 2:
                self.resnet_low = nn.Sequential(self.conv1,
                                                self.bn1,
                                                self.relu,
                                                self.layer1,  # 64, 32, 32
                                                self.layer2,  # 128, 16, 16
                                                self.layer3,  # 256, 8, 8
                                                )

        else:
            self.resnet_low = nn.Sequential(self.conv1,
                                            self.bn1,
                                            self.relu,
                                            self.layer1,  # nf, h, w
                                            self.layer2,  # 2*nf, h/2, w/2
                                            self.layer3,  # 4*nf, h/4, w/4
                                            self.layer4  # 8*nf, h/8, w/8
                                            )

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet_low(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class ResNet18Pre224(nn.Module):
    def __init__(self, stages):
        super(ResNet18Pre224, self).__init__()

        nf = 64
        self.stages = stages
        self.in_planes = nf
        self.block = BasicBlock
        num_blocks = [2, 2, 2, 2]
        self.nf = nf
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, nf * 4, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(self.block, nf * 8, num_blocks[3], stride=2)
        # self._resnet_high = nn.Sequential(
        #                                   self.layer4,
        #                                   nn.Identity()
        #                                   )
        if self.stages == 2:
            self.resnet_low = nn.Sequential(self.conv1,
                                            self.maxpool,
                                            self.bn1,
                                            self.relu,
                                            self.layer1,  # 64, 32, 32
                                            self.layer2,  # 128, 16, 16
                                            self.layer3,  # 256, 8, 8
                                            # self.layer4
                                            )
        if self.stages == 3:
            self.resnet_low = nn.Sequential(self.conv1,
                                            self.maxpool,
                                            self.bn1,
                                            self.relu,
                                            self.layer1,  # 64, 32, 32
                                            self.layer2,  # 128, 16, 16
                                            # self.layer3,  # 256, 8, 8
                                            # self.layer4
                                            )

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet_low(x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class FeedForward(nn.Module):
    def __init__(self, dim, mult, dropout = 0., SN=False):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(dim, dim * mult, 1)) if SN else nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Conv2d(dim * mult, dim, 1)) if SN else nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))
    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()  # dim=1 channel-wise
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class ExternalAttention_module(nn.Module):

    def __init__(self, d_model,S=64):
        super().__init__()
        self.mk=nn.Linear(d_model,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model

        return out


class Attention(nn.Module):
    def __init__(self, dim, fmap_size, heads = 8, dim_key = 32, dim_value = 64, dropout = 0., dim_out = None, downsample = False, BN=True, SN=False):
        super().__init__()
        inner_dim_key = dim_key *  heads
        inner_dim_value = dim_value *  heads
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_key ** -0.5

        if SN:
            self.to_q = nn.Sequential(spectral_norm(nn.Conv2d(dim, inner_dim_key, 1, stride = (2 if downsample else 1), bias = False)), nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity())
            self.to_k = nn.Sequential(spectral_norm(nn.Conv2d(dim, inner_dim_key, 1, bias = False)), nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity())
            self.to_v = nn.Sequential(spectral_norm(nn.Conv2d(dim, inner_dim_value, 1, bias = False)), nn.BatchNorm2d(inner_dim_value) if BN else nn.Identity())
        else:
            self.to_q = nn.Sequential(nn.Conv2d(dim, inner_dim_key, 1, stride = (2 if downsample else 1), bias = False), nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity())
            self.to_k = nn.Sequential(nn.Conv2d(dim, inner_dim_key, 1, bias = False), nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity())
            self.to_v = nn.Sequential(nn.Conv2d(dim, inner_dim_value, 1, bias = False), nn.BatchNorm2d(inner_dim_value) if BN else nn.Identity())

        self.attend = nn.Softmax(dim = -1)

        out_batch_norm = nn.BatchNorm2d(dim_out)
        nn.init.zeros_(out_batch_norm.weight)

        self.to_out = nn.Sequential(
            nn.GELU(),
            spectral_norm(nn.Conv2d(inner_dim_value, dim_out, 1)) if SN else nn.Conv2d(inner_dim_value, dim_out, 1),
            out_batch_norm if BN else nn.Identity(),
            nn.Dropout(dropout)
        )

        # positional bias

        self.pos_bias = nn.Embedding(fmap_size * fmap_size, heads)

        q_range = torch.arange(0, fmap_size, step = (2 if downsample else 1))
        k_range = torch.arange(fmap_size)

        q_pos = torch.stack(torch.meshgrid(q_range, q_range), dim=-1)
        k_pos = torch.stack(torch.meshgrid(k_range, k_range), dim=-1)

        q_pos, k_pos = map(lambda t: rearrange(t, 'i j c -> (i j) c'), (q_pos, k_pos))
        rel_pos = (q_pos[:, None, ...] - k_pos[None, :, ...]).abs()

        x_rel, y_rel = rel_pos.unbind(dim = -1)
        pos_indices = (x_rel * fmap_size) + y_rel

        self.register_buffer('pos_indices', pos_indices)

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, 'i j h -> () h i j')
        return fmap + (bias / self.scale)

    def forward(self, x):
        b, n, *_, h = *x.shape, self.heads

        q = self.to_q(x)
        y = q.shape[2]

        qkv = (q, self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = self.apply_pos_bias(dots)

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)


class AttentionDIY(nn.Module):
    def __init__(self, dim, fmap_size, heads = 8, dim_key = 32, dim_value = 64, dropout = 0., dim_out = None, downsample = False, BN=True, SN=False):
        super().__init__()
        inner_dim_key = dim_key *  heads
        inner_dim_value = dim_value *  heads
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_key ** -0.5

        if SN:
            self.to_q = nn.Sequential(spectral_norm(nn.Conv2d(dim, inner_dim_key, 1, stride = (2 if downsample else 1), bias = False)), nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity())
            self.to_v = nn.Sequential(spectral_norm(nn.Conv2d(dim, inner_dim_value, 1, bias = False)), nn.BatchNorm2d(inner_dim_value) if BN else nn.Identity())
        else:
            self.to_q = nn.Sequential(nn.Conv2d(dim, inner_dim_key, 1, stride = (2 if downsample else 1), bias = False), nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity())
            self.to_v = nn.Sequential(nn.Conv2d(dim, inner_dim_value, 1, bias = False), nn.BatchNorm2d(inner_dim_value) if BN else nn.Identity())

        self.attend = nn.Softmax(dim = -1)

        out_batch_norm = nn.BatchNorm2d(dim_out)
        nn.init.zeros_(out_batch_norm.weight)

        # self.external_k = nn.Sequential(
        #     nn.GELU(),
        #     spectral_norm(nn.Conv2d(inner_dim_value, dim_out, 1)) if SN else nn.Conv2d(inner_dim_value, dim_out, 1),
        #     out_batch_norm if BN else nn.Identity(),
        #     nn.Dropout(dropout)
        # )

        self.mk_batch_norm = nn.BatchNorm2d(fmap_size * fmap_size)
        self.mk = nn.Sequential(
            nn.Linear(dim_key, fmap_size * fmap_size, bias=False),
            # mk_batch_norm
        )

        self.to_out = nn.Sequential(
            nn.GELU(),
            spectral_norm(nn.Conv2d(inner_dim_value, dim_out, 1)) if SN else nn.Conv2d(inner_dim_value, dim_out, 1),
            out_batch_norm if BN else nn.Identity(),
            nn.Dropout(dropout)
        )

        # positional bias

        self.pos_bias = nn.Embedding(fmap_size * fmap_size, heads)

        q_range = torch.arange(0, fmap_size, step = (2 if downsample else 1))
        k_range = torch.arange(fmap_size)

        q_pos = torch.stack(torch.meshgrid(q_range, q_range), dim=-1)
        k_pos = torch.stack(torch.meshgrid(k_range, k_range), dim=-1)

        q_pos, k_pos = map(lambda t: rearrange(t, 'i j c -> (i j) c'), (q_pos, k_pos))
        rel_pos = (q_pos[:, None, ...] - k_pos[None, :, ...]).abs()

        x_rel, y_rel = rel_pos.unbind(dim = -1)
        pos_indices = (x_rel * fmap_size) + y_rel

        self.register_buffer('pos_indices', pos_indices)

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, 'i j h -> () h i j')
        return fmap + (bias / self.scale)

    def forward(self, x):
        b, n, *_, h = *x.shape, self.heads

        q = self.to_q(x)
        y = q.shape[2]

        qv = (q, self.to_v(x))

        q, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = h), qv)
        # q = map(lambda t: rearrange(t, 'b (h d) ... -> b h d (...)', h=h), q)

        dots = self.mk(q)

        dots = rearrange(dots, 'b h hw d -> b d hw h')

        dots = self.mk_batch_norm(dots)

        dots = rearrange(dots, 'b d hw h -> b h hw d')

        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = self.apply_pos_bias(dots)

        attn = self.attend(dots)

        # attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S  效果不好，而且波动加大

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)


class AttentionDIYbn(nn.Module):
    def __init__(self, dim, fmap_size, heads = 8, dim_key = 32, dim_value = 64, dropout = 0., dim_out = None, downsample = False, BN=True, SN=False):
        super().__init__()
        inner_dim_key = dim_key * heads
        inner_dim_value = dim_value * heads
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_key ** -0.5

        if SN:
            self.to_q = nn.Sequential(spectral_norm(nn.Conv2d(dim, inner_dim_key, 1, stride = (2 if downsample else 1), bias = False)), nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity())
            self.to_v = nn.Sequential(spectral_norm(nn.Conv2d(dim, inner_dim_value, 1, bias = False)), nn.BatchNorm2d(inner_dim_value) if BN else nn.Identity())
        else:
            self.to_q = nn.Sequential(nn.Conv2d(dim, inner_dim_key, 1, stride = (2 if downsample else 1), bias = False), nn.BatchNorm2d(inner_dim_key) if BN else nn.Identity())
            self.to_v = nn.Sequential(nn.Conv2d(dim, inner_dim_value, 1, bias = False), nn.BatchNorm2d(inner_dim_value) if BN else nn.Identity())

        self.mk = nn.Sequential(
            nn.Sequential(nn.Conv2d(inner_dim_key, self.heads*fmap_size*fmap_size, 1, bias=False), nn.BatchNorm2d(self.heads*fmap_size*fmap_size)),
        )

        self.attend = nn.Softmax(dim = -1)

        out_batch_norm = nn.BatchNorm2d(dim_out)
        nn.init.zeros_(out_batch_norm.weight)

        # self.external_k = nn.Sequential(
        #     nn.GELU(),
        #     spectral_norm(nn.Conv2d(inner_dim_value, dim_out, 1)) if SN else nn.Conv2d(inner_dim_value, dim_out, 1),
        #     out_batch_norm if BN else nn.Identity(),
        #     nn.Dropout(dropout)
        # )

        self.to_out = nn.Sequential(
            nn.GELU(),
            spectral_norm(nn.Conv2d(inner_dim_value, dim_out, 1)) if SN else nn.Conv2d(inner_dim_value, dim_out, 1),
            out_batch_norm if BN else nn.Identity(),
            nn.Dropout(dropout)
        )

        # positional bias

        self.pos_bias = nn.Embedding(fmap_size * fmap_size, heads)

        q_range = torch.arange(0, fmap_size, step = (2 if downsample else 1))
        k_range = torch.arange(fmap_size)

        q_pos = torch.stack(torch.meshgrid(q_range, q_range), dim=-1)
        k_pos = torch.stack(torch.meshgrid(k_range, k_range), dim=-1)

        q_pos, k_pos = map(lambda t: rearrange(t, 'i j c -> (i j) c'), (q_pos, k_pos))
        rel_pos = (q_pos[:, None, ...] - k_pos[None, :, ...]).abs()

        x_rel, y_rel = rel_pos.unbind(dim = -1)
        pos_indices = (x_rel * fmap_size) + y_rel

        self.register_buffer('pos_indices', pos_indices)

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, 'i j h -> () h i j')
        return fmap + (bias / self.scale)

    def forward(self, x):
        b, n, *_, h = *x.shape, self.heads

        q = self.to_q(x)
        y = q.shape[2]

        v = rearrange(self.to_v(x), 'b (h d) ... -> b h (...) d', h=h)

        dots = self.mk(q)

        dots = rearrange(dots, 'b (h d) ... -> b h (...) d', h=h)

        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = self.apply_pos_bias(dots)

        attn = self.attend(dots)

        # attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S  效果不好，而且波动加大

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult=2, dropout=0., dim_out=None, downsample=False, BN=True, SN=False, LN=False):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.layers = nn.ModuleList([])
        self.attn_residual = (not downsample) and dim == dim_out

        if LN:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, AttentionDIYbn(dim, fmap_size=fmap_size, heads=heads, dim_key=dim_key, dim_value=dim_value,
                                           dropout=dropout, downsample=downsample, dim_out=dim_out, BN=BN, SN=SN)),
                    PreNorm(dim_out, FeedForward(dim_out, mlp_mult, dropout=dropout, SN=SN))
                ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    AttentionDIYbn(dim, fmap_size = fmap_size, heads = heads, dim_key = dim_key, dim_value = dim_value, dropout = dropout, downsample = downsample, dim_out = dim_out, BN = BN, SN=SN),
                    FeedForward(dim_out, mlp_mult, dropout = dropout, SN=SN)
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            attn_res = (x if self.attn_residual else 0)
            x = attn(x) + attn_res
            x = ff(x) + x
        return x


class CVT_online(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_mult,
        stages = 3,
        dim_key = 32,
        dim_value = 64,
        dropout = 0.,
        cnnbackbone = 'ResNet18Pre',
        independent_classifier=False,
        frozen_head = False,
        BN = True,  # Batchnorm
        LN=False,  # LayerNorm
        SN = False,  # SpectralNorm
        grow=False,  # Expand the network
        mean_cob=False,
        sum_cob=True,
        max_cob=False,
        distill_classifier=True,
        cosine_classifier=False,
        use_WA = False,
        init="kaiming",
        device='cuda',
        use_bias=False,
    ):
        super().__init__()

        dims = cast_tuple(dim, stages)
        depths = cast_tuple(depth, stages)
        layer_heads = cast_tuple(heads, stages)

        self.dims = dims
        self.depths = depths
        self.layer_heads = layer_heads
        self.image_size = image_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_mult = mlp_mult
        self.stages = stages
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dropout = dropout
        self.distill_classifier = distill_classifier
        self.cnnbackbone = cnnbackbone
        if image_size == 224 and num_classes == 100:
            self.cnnbackbone = 'ResNet18Pre224'     # ResNet18Pre224   PreActResNet
        self.nf = 64 if image_size < 100 else 32
        self.independent_classifier = independent_classifier
        self.frozen_head = frozen_head
        self.BN = BN
        self.SN = SN
        self.LN = LN
        self.grow = grow
        self.init = init
        self.use_WA = use_WA
        self.device = device
        self.weight_normalization = cosine_classifier
        self.use_bias = use_bias
        self.mean_cob=mean_cob
        self.sum_cob=sum_cob
        self.max_cob=max_cob
        self.gamma = None

        print('-----------------------------------------', depths)
        assert all(map(lambda t: len(t) == stages, (dims, depths, layer_heads))), 'dimensions, depths, and heads must be a tuple that is less than the designated number of stages'

        if self.cnnbackbone == 'ResNet18Pre':
            self.conv = ResNet18Pre(self.nf, self.stages)
        elif self.cnnbackbone == 'ResNet164':
            self.conv = ResNet164()
        elif self.cnnbackbone == 'ResNet18Pre224':
            self.conv = ResNet18Pre224(self.stages)
        elif self.cnnbackbone == 'PreActResNet':
            self.conv = PreActResNet()
        else:
            assert()

        if grow:
            print("Enable dynamical Transformer expansion!")
            self.transformers = nn.ModuleList()
            self.transformers.append(self.add_transformer())
            # self.transformers.append(self.add_transformer())
            # self.transformers.append(self.add_transformer())
            # self.transformers.append(self.add_transformer())
            # self.transformers.append(self.add_transformer())
        else:
            self.transformer = self.add_transformer()  # self.add_transformer()  # self.conv._resnet_high

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...')
        )

        self.distill_head = self._gen_classifier(dims[-1], num_classes) if self.distill_classifier else always(None)

        if self.independent_classifier:
            task_class = 2 if num_classes < 20 else 20
            self.fix = nn.ModuleList([self._gen_classifier(dims[-1], task_class) for i in range(num_classes//task_class)])
        else:
            self.mlp_head = spectral_norm(self._gen_classifier(dims[-1], num_classes)) if SN else self._gen_classifier(dims[-1], num_classes)

        self.feature_head = self._gen_classifier(dims[-1], num_classes)

        # self.focuses = nn.Parameter(torch.FloatTensor(self.num_classes, self.dims[-1]), requires_grad=True).to(self.device)
        self.focuses = nn.Parameter(torch.FloatTensor(self.num_classes, 512).fill_(1), requires_grad=True).to(self.device)
        # self.focuses = F.normalize(focuses_org, dim=1)
        self.focus_labels = torch.tensor([i for i in range(self.num_classes)]).to(self.device)

    def focuses_head(self):
        return F.normalize(self.feature_head(self.focuses), dim=1)

    def add_transformer(self):
        if self.nf == 64:
            fmap_size = self.image_size // ((2 ** 2) if self.stages < 3 else (2 ** 1))
        else:
            fmap_size = self.image_size // (2 ** 3)
        if self.cnnbackbone == 'ResNet18Pre224' or self.cnnbackbone == 'PreActResNet':
            if self.stages == 3:
                fmap_size = self.image_size // (2 ** 3)
            if self.stages == 2:
                fmap_size = self.image_size // (2 ** 4)
        layers = []

        for ind, dim, depth, heads in zip(range(self.stages), self.dims, self.depths, self.layer_heads):
            is_last = ind == (self.stages - 1)
            layers.append(Transformer(dim, fmap_size, depth, heads, self.dim_key, self.dim_value, self.mlp_mult, self.dropout, BN=self.BN, SN=self.SN, LN=self.LN))

            if not is_last:   # downsample
                next_dim = self.dims[ind + 1]
                layers.append(Transformer(dim, fmap_size, 1, heads * 2, self.dim_key, self.dim_value, dim_out=next_dim, downsample=True, BN=self.BN, SN=self.SN, LN=self.LN))
                fmap_size = ceil(fmap_size / 2)
        return nn.Sequential(*layers)

    def fix_and_grow(self):
        print('fix and grow !!!')
        # for param in self.conv.parameters():
        #     param.requires_grad = False
        # # self.conv.eval()
        # for param in self.transformers.parameters():
        #     param.requires_grad = False
        # # self.transformers.eval()
        self.transformers.append(self.add_transformer())
        # return self

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)
        return classifier

    @torch.no_grad()
    def update_gamma(self, task_num, class_per_task):
        if task_num == 0:
            return 1
        if self.distill_classifier:
            classifier = self.distill_head
        else:
            classifier = self.mlp_head
        old_weight_norm = torch.norm(classifier.weight[:task_num*class_per_task], p=2, dim=1)
        new_weight_norm = torch.norm(classifier.weight[task_num*class_per_task:task_num*class_per_task+class_per_task], p=2, dim=1)
        self.gamma = old_weight_norm.mean() / new_weight_norm.mean()
        print('gamma: ', self.gamma.cpu().item(), '  use_WA:', self.use_WA)
        if not self.use_WA:
            return 1
        return self.gamma

    def forward(self, img):
        x = self.conv(img)
        if self.grow:
            x = [transformer(x) for transformer in self.transformers]
            if self.sum_cob:
                x = torch.stack(x).sum(dim=0)  # add growing transformers' output
            elif self.mean_cob:
                x = torch.stack(x).mean(dim=0)
            elif self.max_cob:
                for i in range(len(x)-1):
                    x[i+1] = x[i].max(x[i+1])
                x = x[-1]
            else:
                ValueError
        else:
            x = self.transformer(x)

        x = self.pool(x)

        if self.independent_classifier:
            y = torch.tensor([])
            for fix in self.fix:
                y = torch.cat((fix(x), y), 1)
            out = y
        else:
            out = self.mlp_head(x)

        # print('Out size:', out.size())

        return out

    def distill_classification(self, img):
        # with torch.cuda.amp.autocast():
        x = self.conv(img)
        if self.grow:
            x = [transformer(x) for transformer in self.transformers]
            if self.sum_cob:
                x = torch.stack(x).sum(dim=0)  # add growing transformers' output
            elif self.mean_cob:
                x = torch.stack(x).mean(dim=0)
            elif self.max_cob:
                for i in range(len(x)-1):
                    x[i+1] = x[i].max(x[i+1])
                x = x[-1]
        else:
            x = self.transformer(x)

        x = self.pool(x)
        distill = self.distill_head(x)

        if self.independent_classifier:
            y = torch.tensor([]).to('cuda')
            for fix in self.fix:
                y = torch.cat((fix(x).to('cuda'), y), 1)
            out = y
        else:
            out = self.mlp_head(x)

        if exists(distill):
            return distill
        # print('distill_classification Out size:', out.size())

        return out

    def contrasive_f(self, img):
        x = self.conv(img)
        if self.grow:
            x = [transformer(x) for transformer in self.transformers]
            if self.sum_cob:
                x = torch.stack(x).sum(dim=0)  # add growing transformers' output
            elif self.mean_cob:
                x = torch.stack(x).mean(dim=0)
            elif self.max_cob:
                for i in range(len(x)-1):
                    x[i+1] = x[i].max(x[i+1])
                x = x[-1]
            else:
                ValueError
        else:
            x = self.transformer(x)

        x = self.pool(x)
        x = self.feature_head(x)  # 去掉效果好像略好，区别不太大！！
        x = F.normalize(x, dim=1)

        return x

    def frozen(self, t):
        if self.independent_classifier and self.frozen_head:
            print('----------frozen-----------')
            for i in range(t+1):
                self.fix[i].weight.requires_grad = False
                print('frozen ', i)
        if t > -1 and self.grow:
            self.fix_and_grow()
            pass
