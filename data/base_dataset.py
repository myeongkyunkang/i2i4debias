"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
from abc import ABC, abstractmethod

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """

        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

    def set_phase(self, phase):
        assert phase in ["train", "test", "val"]
        self.current_phase = phase
        pass


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if not opt.isTrain:
        transform_list.append(transforms.Resize((opt.crop_size, opt.crop_size), method))
    elif opt.preprocess == 'custom':
        transform_list.append(transforms.Resize((opt.crop_size, opt.crop_size), method))
        transform_list.append(transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2)))
    elif opt.preprocess == 'custom_noaffine':
        transform_list.append(transforms.Resize((opt.crop_size, opt.crop_size), method))
    else:
        if 'fixsize' in opt.preprocess:
            transform_list.append(transforms.Resize((opt.crop_size, opt.load_size), method))
        if 'resize' in opt.preprocess:
            osize = [opt.load_size, opt.load_size]
            if "gta2cityscapes" in opt.dataroot:
                osize[0] = opt.load_size // 2
            transform_list.append(transforms.Resize(osize, method))
        elif 'scale_width' in opt.preprocess:
            transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
        elif 'scale_shortside' in opt.preprocess:
            transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, opt.crop_size, method)))
        elif 'scale_longside' in opt.preprocess:
            transform_list.append(transforms.Lambda(lambda img: __scale_longside(img, opt.load_size, opt.crop_size, method)))

        if 'zoom' in opt.preprocess:
            if params is None:
                transform_list.append(transforms.Lambda(lambda img: __random_zoom(img, opt.load_size, opt.crop_size, method)))
            else:
                transform_list.append(transforms.Lambda(lambda img: __random_zoom(img, opt.load_size, opt.crop_size, method, factor=params["scale_factor"])))

        if 'centercrop' in opt.preprocess:
            transform_list.append(transforms.Lambda(lambda img: __centercrop(img)))
        elif 'crop' in opt.preprocess:
            if params is None or 'crop_pos' not in params:
                transform_list.append(transforms.RandomCrop(opt.crop_size, padding=opt.preprocess_crop_padding))
            else:
                transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

        if 'patch' in opt.preprocess:
            transform_list.append(transforms.Lambda(lambda img: __patch(img, params['patch_index'], opt.crop_size)))

        if 'trim' in opt.preprocess:
            transform_list.append(transforms.Lambda(lambda img: __trim(img, opt.crop_size)))

        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=16, method=method)))

    random_flip = opt.isTrain and (not opt.no_flip)
    if random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    return img.resize((w, h), method)


def __random_zoom(img, target_width, crop_width, method=Image.BICUBIC, factor=None):
    iw, ih = img.size
    if factor is None:
        zoom_level = np.random.uniform(crop_width / iw, 1.0, size=[2])
    else:
        zoom_level = (factor[0], factor[1])
    zoomw = max(crop_width, iw * zoom_level[0])
    zoomh = max(crop_width, ih * zoom_level[1])
    img = img.resize((int(round(zoomw)), int(round(zoomh))), method)
    return img


def __scale_shortside(img, target_width, crop_width, method=Image.BICUBIC):
    ow, oh = img.size
    shortside = min(ow, oh)
    scale = target_width / shortside
    return img.resize((round(ow * scale), round(oh * scale)), method)


def __centercrop(img):
    ow, oh = img.size
    s = min(ow, oh)
    return img.crop(((ow - s) // 2, (oh - s) // 2, (ow + s) // 2, (oh + s) // 2))


def __scale_longside(img, target_width, crop_width, method=Image.BICUBIC):
    ow, oh = img.size
    longside = max(ow, oh)
    scale = target_width / longside
    return img.resize((round(ow * scale), round(oh * scale)), method)


def __trim(img, trim_width):
    ow, oh = img.size
    if ow > trim_width:
        xstart = np.random.randint(ow - trim_width)
        xend = xstart + trim_width
    else:
        xstart = 0
        xend = ow
    if oh > trim_width:
        ystart = np.random.randint(oh - trim_width)
        yend = ystart + trim_width
    else:
        ystart = 0
        yend = oh
    return img.crop((xstart, ystart, xend, yend))


def __scale_width(img, target_width, crop_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width and oh >= crop_width:
        return img
    w = target_width
    # h = int(max(target_width * oh / ow, crop_width))
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __patch(img, index, size):
    ow, oh = img.size
    nw, nh = ow // size, oh // size
    roomx = ow - nw * size
    roomy = oh - nh * size
    startx = np.random.randint(int(roomx) + 1)
    starty = np.random.randint(int(roomy) + 1)

    index = index % (nw * nh)
    ix = index // nh
    iy = index % nh
    gridx = startx + ix * size
    gridy = starty + iy * size
    return img.crop((gridx, gridy, gridx + size, gridy + size))
