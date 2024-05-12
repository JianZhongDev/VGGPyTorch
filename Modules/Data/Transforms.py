"""
FILENAME: Transforms.py
DESCRIPTION: VGG data transforms
@author: Jian Zhong
"""

import torch
from torchvision.transforms import v2


# subtract mean value of each channel
def subtract_channel_mean(src_image):
    ch_mean = torch.mean(src_image, dim = (-1,-2), keepdim = True)
    return src_image - ch_mean


# subtract constant value from the image
def subtract_const(src_image, const_val):
    return src_image - const_val


# image channel radom PCA eigenvec addition agumentation
def random_ch_shift_pca(src_image, pca_eigenvecs, pca_eigenvals, random_paras = None):
    norm_meam = 0
    norm_std = 0.1
    if isinstance(random_paras, dict):
        norm_meam = random_paras.get("mean", norm_meam)
        norm_std = random_paras.get("std", norm_std)
    
    nof_dims = len(src_image.size())
    nof_channels = src_image.size(0)

    assert(pca_eigenvecs.size(0) == nof_channels)
    assert(len(pca_eigenvals.size()) == 1)
    assert(pca_eigenvals.size(0) == pca_eigenvecs.size(1))

    norm_means = 0 * torch.ones(pca_eigenvals.size())
    norm_stds = 0.1 * torch.ones(pca_eigenvals.size())

    alphas = torch.normal(norm_means, norm_stds)
    scale_factors = (alphas * pca_eigenvals).view((-1,1))

    ch_offset = torch.matmul(pca_eigenvecs, scale_factors)

    ch_offset = ch_offset.view((nof_channels,) + (1,) * (nof_dims - 1))

    dst_image = src_image + ch_offset

    return dst_image

    


