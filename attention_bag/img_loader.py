from torch.utils.data import Dataset
import torch
import random
from PIL import Image
import numpy as np


def read_img(img_path):
    """
    keep reading until succeed
    :param img_path:
    :return:
    """
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):

    def __init__(self, dataset, sample='evenly', transform=None):
        self.dataset = dataset
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, pid, bag = self.dataset[index]
        return img,pid,bag
        #
        # if self.sample == 'random':
        #     """
        #     read Image for train dataset
        #     """
        #     img = read_img(img_path)
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     # img = tuple(img)
        #     return img, pid, bag
        # elif self.sample == 'dense':
        #     img = read_img(img_path)
        #     if self.transform is not None:
        #         img = self.transform(img)
        #     return img, pid, bag
        # else:
        #     raise KeyError("Unknown sample method: {}".format(self.sample))




