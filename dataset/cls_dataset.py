import os.path as osp
from base import BaseDataset


class ClsDataset(BaseDataset):
    def __init__(self, root, source, transformer, loader):
        super(ClsDataset, self).__init__(root, source, loader)
        self.transformer = transformer

    def __getitem__(self, index):
        path, label = self.data[index].split(' ')
        img = self.loader(osp.join(self.root, path))
        img = self.transformer(img)
        return img, int(label)