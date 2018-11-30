import _infimnist as infimnist
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class InfiMNIST(data.Dataset):
    def __init__(self, train=True, num_transformations=1, transform=None, target_transform=None):
        self.mnist = infimnist.InfimnistGenerator()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.num_transformations = num_transformations

    def __getitem__(self, index):
        idxs = np.array([10000 + index if self.train else index], dtype=np.int64)
        digits, labels = self.mnist.gen(idxs)
        img = Image.fromarray(digits.reshape(28, 28), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        target = torch.tensor(labels[0], dtype=torch.int64)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return 60000 * self.num_transformations if self.train else 10000
