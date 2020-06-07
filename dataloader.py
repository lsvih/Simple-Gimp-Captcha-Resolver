import cv2
import numpy as np
import torch
import os
import torch.utils.data as data
from torchvision.transforms.functional import to_tensor
from utils import characters


class Dataset(data.Dataset):
    """ Digits dataset."""

    def __init__(self, mode):
        assert (mode == 'train' or mode == 'test')

        print('Loading %s data...' % mode)
        self.mode = mode
        self.img_root = 'dataset_' + mode
        self.img_names = list(filter(lambda x: x.endswith('.jpg'), os.listdir(self.img_root)))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        label = img_name[:-4]
        img = cv2.imread(os.path.join(self.img_root, img_name))
        img = cv2.resize(img, (200, 64))
        lower = np.array([0, 0, 0])
        upper = np.array([100, 100, 100])
        img = cv2.inRange(img, lower, upper)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.dilate(img, element, iterations=2)
        img = 255 - img
        return to_tensor(img), [characters.index(i) for i in label]


class TrainBatch:
    def __init__(self, batch):
        transposed_data = list(zip(*batch))
        self.images = torch.stack(transposed_data[0], 0)
        self.labels = torch.tensor(transposed_data[1])
        self.label_lengths = torch.full(size=(1, self.labels.shape[0]), fill_value=7, dtype=torch.long)


class TestBatch:
    def __init__(self, batch):
        transposed_data = list(zip(*batch))
        self.images = torch.stack(transposed_data[0], 0)
        self.labels = torch.tensor(transposed_data[1])


def train_fn(batch):
    return TrainBatch(batch)


def test_fn(batch):
    return TestBatch(batch)
