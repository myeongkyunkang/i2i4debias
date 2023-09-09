"""This module contains simple helper functions """
from __future__ import print_function

import argparse
import importlib
import os
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA as PCA


def normalize(v):
    if type(v) == list:
        return [normalize(vv) for vv in v]

    return v * torch.rsqrt((torch.sum(v ** 2, dim=1, keepdim=True) + 1e-8))


def lerp(a, b, r):
    if type(a) == list or type(a) == tuple:
        return [lerp(aa, bb, r) for aa, bb in zip(a, b)]
    return a * (1 - r) + b * r


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls


def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=2):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if len(image_tensor.shape) == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.shape[0]):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile is not False:
            tile = max(min(images_np.shape[0] // 2, 4), 1) if tile is True else tile
            images_tiled = tile_images(images_np, picturesPerRow=tile)
            return images_tiled
        else:
            return images_np

    if len(image_tensor.shape) == 2:
        assert False
    image_numpy = image_tensor.detach().cpu().numpy() if type(image_tensor) is not np.ndarray else image_tensor
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, axis=2)
    return image_numpy.astype(imtype)


def toPILImage(images, tile=None):
    if isinstance(images, list):
        if all(['tensor' in str(type(image)).lower() for image in images]):
            return toPILImage(torch.cat([im.cpu() for im in images], dim=0), tile)
        return [toPILImage(image, tile=tile) for image in images]

    if 'ndarray' in str(type(images)).lower():
        return toPILImage(torch.from_numpy(images))

    assert 'tensor' in str(type(images)).lower(), "input of type %s cannot be handled." % str(type(images))

    if tile is None:
        max_width = 2560
        tile = min(images.size(0), int(max_width / images.size(3)))

    return Image.fromarray(tensor2im(images, tile=tile))


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def visualize_spatial_code(sp):
    device = sp.device
    if sp.size(1) <= 2:
        sp = sp.repeat([1, 3, 1, 1])[:, :3, :, :]
    if sp.size(1) == 3:
        pass
    else:
        sp = sp.detach().cpu().numpy()
        X = np.transpose(sp, (0, 2, 3, 1))
        B, H, W = X.shape[0], X.shape[1], X.shape[2]
        X = np.reshape(X, (-1, X.shape[3]))
        X = X - X.mean(axis=0, keepdims=True)
        try:
            Z = PCA(3).fit_transform(X)
        except ValueError:
            print("Running PCA on the structure code has failed.")
            print("This is likely a bug of scikit-learn in version 0.18.1.")
            print("https://stackoverflow.com/a/42764378")
            print("The visualization of the structure code on visdom won't work.")
            return torch.zeros(B, 3, H, W, device=device)
        sp = np.transpose(np.reshape(Z, (B, H, W, -1)), (0, 3, 1, 2))
        sp = (sp - sp.min()) / (sp.max() - sp.min()) * 2 - 1
        sp = torch.from_numpy(sp).to(device)
    return sp


def apply_random_crop(x, target_size, scale_range, num_crops=1, return_rect=False):
    # build grid
    B = x.size(0) * num_crops
    flip = torch.round(torch.rand(B, 1, 1, 1, device=x.device)) * 2 - 1.0
    unit_grid_x = torch.linspace(-1.0, 1.0, target_size, device=x.device)[np.newaxis, np.newaxis, :, np.newaxis].repeat(B, target_size, 1, 1)
    unit_grid_y = unit_grid_x.transpose(1, 2)
    unit_grid = torch.cat([unit_grid_x * flip, unit_grid_y], dim=3)

    x = x.unsqueeze(1).expand(-1, num_crops, -1, -1, -1).flatten(0, 1)
    scale = torch.rand(B, 1, 1, 2, device=x.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    offset = (torch.rand(B, 1, 1, 2, device=x.device) * 2 - 1) * (1 - scale)
    sampling_grid = unit_grid * scale + offset
    crop = F.grid_sample(x, sampling_grid, align_corners=False)

    crop = crop.view(B // num_crops, num_crops, crop.size(1), crop.size(2), crop.size(3))

    return crop


def to_numpy(metric_dict):
    new_dict = {}
    for k, v in metric_dict.items():
        if "numpy" not in str(type(v)):
            v = v.detach().cpu().mean().numpy()
        new_dict[k] = v
    return new_dict


def is_custom_kernel_supported():
    version_str = str(torch.version.cuda).split(".")
    major = version_str[0]
    minor = version_str[1]
    return int(major) >= 10 and int(minor) >= 1


def resize2d_tensor(x, size_or_tensor_of_size):
    if torch.is_tensor(size_or_tensor_of_size):
        size = size_or_tensor_of_size.size()
    elif isinstance(size_or_tensor_of_size, np.ndarray):
        size = size_or_tensor_of_size.shape
    else:
        size = size_or_tensor_of_size

    if isinstance(size, tuple) or isinstance(size, list):
        return F.interpolate(x, size[-2:], mode='bilinear', align_corners=False)
    else:
        raise ValueError("%s is unrecognized" % str(type(size)))
