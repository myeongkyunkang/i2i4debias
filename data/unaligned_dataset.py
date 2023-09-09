import os
import random

import torchvision.transforms as transforms
from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, 'trainA')  # trainA only
        self.dir_B = os.path.join(opt.dataroot, 'trainB')  # trainB only
        self.A_paths = sorted(make_dataset(self.dir_A))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.transform_A = get_transform(self.opt, convert=False)
        self.transform_B = get_transform(self.opt, convert=False)

        self.transform_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range

        index_B = random.randint(0, self.B_size - 1)  # randomize the index for domain B to avoid fixed pairs.
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        A_tensor = self.transform_tensor(A)
        B_tensor = self.transform_tensor(B)

        return {'real_A': A_tensor, 'real_B': B_tensor, 'path_A': A_path, 'path_B': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
