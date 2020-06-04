from PIL import Image
import torch.utils.data as data
from abc import ABC, abstractmethod


class ImageLoader(object):
    def __init__(self, size):
        if isinstance(size, tuple) or isinstance(size, list):
            self.size = tuple(size)
        else:
            self.size = (size, size)

    def __call__(self, path):
        img = Image.open(path).convert('RGB')
        return img.resize(self.size, resample=Image.BICUBIC)


class BaseDataset(data.Dataset, ABC):
    def __init__(self, root, source, img_loader):
        self.root = root
        self.data = [line.rstrip('\n') for line in open(source)]
        self.loader = img_loader

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.data)