# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, upscale_factor,
                 upsample_mode='up', BN_enable=True, norm_layer=None):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.BN_enable = BN_enable
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = nn.Sequential(nn.ReplicationPad2d(1),
                                  nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=3, stride=1, padding=0, bias=False))

        if self.BN_enable:
            self.norm1 = norm_layer(out_channels)
        self.relu1 = nn.ReLU(inplace=False)

        if self.upsample_mode == 'deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels,
                                               kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        elif self.upsample_mode == 'pixelshuffle':
            self.upsample = nn.PixelShuffle(upscale_factor=upscale_factor)
        elif self.upsample_mode == 'up':
            self.upsample = nn.Upsample(scale_factor=upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        if self.BN_enable:
            x = self.norm1(x)
        x = self.relu1(x)
        x = self.upsample(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=4, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_dense(self, x):
        outputs = {}
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outputs['d{}'.format(i+1)] = x
        return outputs

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXtU(nn.Module):
    """CMC model with a single linear/mlp projection head"""
    def __init__(self, in_chans=4, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., BN_enable=True):
        super(ConvNeXtU, self).__init__()

        self.embed_dim = dims[0]
        self.BN_enable = BN_enable
        # use split batchnorm
        norm_layer = nn.BatchNorm2d

        self.encoder = ConvNeXt(in_chans=in_chans, num_classes=num_classes,
                 depths=depths, dims=dims, drop_path_rate=drop_path_rate,
                 layer_scale_init_value=layer_scale_init_value, head_init_scale=head_init_scale)

        self.center = DecoderBlock(in_channels=dims[3], mid_channels=dims[3] * 4, out_channels=dims[3],
                                   upscale_factor=2, BN_enable=self.BN_enable, norm_layer=norm_layer)
        self.decoder1 = DecoderBlock(in_channels=dims[3] + dims[2], mid_channels=dims[2] * 4, out_channels=dims[2],
                                     upscale_factor=2, BN_enable=self.BN_enable, norm_layer=norm_layer)
        self.decoder2 = DecoderBlock(in_channels=dims[2] + dims[1], mid_channels=dims[1] * 4, out_channels=dims[1],
                                     upscale_factor=2, BN_enable=self.BN_enable, norm_layer=norm_layer)
        self.decoder3 = DecoderBlock(in_channels=dims[1] + dims[0], mid_channels=dims[0] * 4, out_channels=dims[0],
                                     upscale_factor=4, BN_enable=self.BN_enable, norm_layer=norm_layer)
        self.last = nn.Conv2d(dims[0], dims[0], kernel_size=1, stride=1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.encoder.forward_dense(x)
        e1 = feat['d1']
        e2 = feat['d2']
        e3 = feat['d3']
        e4 = feat['d4']
        center = self.center(e4)
        d2 = self.decoder1(torch.cat([center, e3], dim=1))
        d3 = self.decoder2(torch.cat([d2, e2], dim=1))
        d4 = self.decoder3(torch.cat([d3, e1], dim=1))
        return d4

    def get_dense_feature(self, x):
        x = self.forward(x)
        return self.last(x)


def convnextu_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXtU(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextu_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXtU(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextu_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXtU(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def convnextu_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXtU(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


def convnextu_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXtU(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    return model


class twinshift(nn.Module):
    """CMC model with a single linear/mlp projection head"""

    def __init__(self, in_chans=4, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., BN_enable=True):
        super(twinshift, self).__init__()

        self.embed_dim = dims[0]
        self.BN_enable = BN_enable
        # use split batchnorm
        norm_layer = nn.BatchNorm2d

        self.encoder = ConvNeXt(in_chans=in_chans, num_classes=num_classes,
                                depths=depths, dims=dims, drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value, head_init_scale=head_init_scale)

        self.center = DecoderBlock(in_channels=dims[3], mid_channels=dims[3] * 4, out_channels=dims[3],
                                   upscale_factor=2, BN_enable=self.BN_enable, norm_layer=norm_layer)
        self.decoder1 = DecoderBlock(in_channels=dims[3] + dims[2], mid_channels=dims[2] * 4, out_channels=dims[2],
                                     upscale_factor=2, BN_enable=self.BN_enable, norm_layer=norm_layer)
        self.decoder2 = DecoderBlock(in_channels=dims[2] + dims[1], mid_channels=dims[1] * 4, out_channels=dims[1],
                                     upscale_factor=2, BN_enable=self.BN_enable, norm_layer=norm_layer)
        self.decoder3 = DecoderBlock(in_channels=dims[1] + dims[0], mid_channels=dims[0] * 4, out_channels=dims[0],
                                     upscale_factor=4, BN_enable=self.BN_enable, norm_layer=norm_layer)

    def forward_one(self, x):
        feat = self.encoder.forward_dense(x)
        e1 = feat['d1']
        e2 = feat['d2']
        e3 = feat['d3']
        e4 = feat['d4']
        center = self.center(e4)
        d2 = self.decoder1(torch.cat([center, e3], dim=1))
        d3 = self.decoder2(torch.cat([d2, e2], dim=1))
        d4 = self.decoder3(torch.cat([d3, e1], dim=1))
        return d4

    def forward(self, x1, x2, mode=0):
        # mode --
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)
        if mode == 1:
            return feat1, feat2



class ResUnet182(nn.Module):
    def __init__(self, in_chans=4, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., BN_enable=True):
        super(ResUnet182, self).__init__()

        self.embed_dim = dims[0]
        self.BN_enable = BN_enable
        # use split batchnorm
        norm_layer = nn.BatchNorm2d

        self.encoder = ConvNeXt(in_chans=in_chans, num_classes=num_classes,
                                depths=depths, dims=dims, drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value, head_init_scale=head_init_scale)

        self.center = DecoderBlock(in_channels=dims[3], mid_channels=dims[3] * 4, out_channels=dims[3],
                                   upscale_factor=2, BN_enable=self.BN_enable, norm_layer=norm_layer)
        self.decoder1 = DecoderBlock(in_channels=dims[3] + dims[2], mid_channels=dims[2] * 4, out_channels=dims[2],
                                     upscale_factor=2, BN_enable=self.BN_enable, norm_layer=norm_layer)
        self.decoder2 = DecoderBlock(in_channels=dims[2] + dims[1], mid_channels=dims[1] * 4, out_channels=dims[1],
                                     upscale_factor=2, BN_enable=self.BN_enable, norm_layer=norm_layer)
        self.decoder3 = DecoderBlock(in_channels=dims[1] + dims[0], mid_channels=dims[0] * 4, out_channels=dims[0],
                                     upscale_factor=4, BN_enable=self.BN_enable, norm_layer=norm_layer)
        self.last_conv = nn.Conv2d(dims[0], dims[0], kernel_size=1, stride=1)
        self.unt = nn.Sequential(
            nn.Conv2d(dims[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, mode=0):
        feat = self.encoder.forward_dense(x)
        e1 = feat['d1']
        e2 = feat['d2']
        e3 = feat['d3']
        e4 = feat['d4']
        center = self.center(e4)
        d2 = self.decoder1(torch.cat([center, e3], dim=1))
        d3 = self.decoder2(torch.cat([d2, e2], dim=1))
        d4 = self.decoder3(torch.cat([d3, e1], dim=1))
        val = self.last_conv(d4)
        std = self.unt(d4)
        return val, std

    def get_dense_feature(self, x):
        x = self.forward(x)
        return self.last(x)


if __name__ == '__main__':
    model = convnextu_tiny()
    x = torch.zeros(1, 4, 96, 96)
    y = model(x)
    print(y.shape)
