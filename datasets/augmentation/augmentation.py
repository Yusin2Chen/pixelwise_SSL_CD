
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from .geometric import (
    hflip,
    vflip,
    rotate,
    warp_perspective,
    get_perspective_transform,
    warp_affine,
    get_affine_matrix2d)




class AugmentationBase(nn.Module):

    def __init__(self, p=0.2):
        super(AugmentationBase, self).__init__()
        self.p = p

    def apply(self):
        raise NotImplemented


    def forward(self, img, params):

        # apply transform
        img = self.apply(img, params)

        return img

# --------------------------------------
#             Geometric
# --------------------------------------


class RandomHorizontalFlip(AugmentationBase):
    r"""Applies a random horizontal flip to a tensor image or a batch of tensor images with a given probability.

    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(B, C, H, W)`.
    If Input is a tuple it is assumed that the first element contains the aforementioned tensors and the second,
    the corresponding transformation matrix that has been applied to them. In this case the module
    will Horizontally flip the tensors and concatenate the corresponding transformation matrix to the
    previous one. This is especially useful when using this functionality as part of an ``nn.Sequential`` module.

    """

    def __init__(self, p=0.2):
        super(RandomHorizontalFlip, self).__init__(p=p)

    def apply(self, inp, params):
        out = inp.clone()
        target = params["target"]
        out[target] = hflip(inp)[target]
        return out


class RandomVerticalFlip(AugmentationBase):

    r"""Applies a random vertical flip to a tensor image or a batch of tensor images with a given probability.
    """
    def __init__(self, p=0.2):
        super(RandomVerticalFlip, self).__init__(p=p)

    def apply(self, inp, params):
        out = inp.clone()
        target = params["target"]
        out[target] = vflip(inp)[target]
        return out


class RandomPerspective(AugmentationBase):
    r"""Applies a random perspective transformation to an image tensor with a given probability.

    """

    def __init__(self, p, distortion_scale, interpolation='bilinear', border_mode='zeros', align_corners=False):
        super(RandomPerspective, self).__init__(p=p)
        self.distortion_scale = distortion_scale
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.align_corners = align_corners


    def apply(self, inp, params):

        out = inp.clone()
        target = params["target"]

        transform = get_perspective_transform(params['start_points'], params['end_points']).type_as(inp)

        size = inp.shape[-2:]

        out[target] = warp_perspective(inp, transform, size,
                                       interpolation=self.interpolation,
                                       border_mode=self.border_mode,
                                       align_corners=self.align_corners)[target]

        return out


class RandomAffine(AugmentationBase):
    r"""Applies a random 2D affine transformation to a tensor image.

    The transformation is computed so that the image center is kept invariant.
    """

    def __init__(self, p, theta, h_trans, v_trans, scale, shear,
                 interpolation='nearest', padding_mode='zeros', align_corners=False):
        super(RandomAffine, self).__init__(p=p)
        if theta is not None:
            self.theta = [-theta, theta]
        self.translate = [h_trans, v_trans]
        self.scale = scale
        self.shear = shear

        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.align_corners = align_corners


    def apply(self, inp, params):

        out = inp.clone()
        target = params["target"]

        # concatenate transforms
        transform = get_affine_matrix2d(translations=params["translations"],
                                        center=params["center"],
                                        scale=params["scale"],
                                        angle=params["angle"],
                                        sx=params["sx"],
                                        sy=params["sy"]).type_as(inp)
        size = inp.shape[-2:]

        out[target] = warp_affine(inp, transform, size,
                                  interpolation=self.interpolation,
                                  padding_mode=self.padding_mode,
                                  align_corners=self.align_corners)[target]

        return out


class RandomRotation(AugmentationBase):
    r"""Applies a random rotation to a tensor image or a batch of tensor images given an amount of degrees.
    """

    def __init__(self, p, theta, interpolation='bilinear', padding_mode='zeros', align_corners=False):
        super(RandomRotation, self).__init__(p=p)

        self.theta = [-theta, theta]

        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.align_corners = align_corners


    def apply(self, inp, params):

        out = inp.clone()
        target = params["target"]

        out[target] = rotate(inp,
                             angle=params["angle"],
                             align_corners=self.align_corners,
                             interpolation=self.interpolation)[target]

        return out

