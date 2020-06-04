import os.path as osp
import random
from PIL import Image
import numbers
import warnings
import math
import torch
from torchvision import transforms
import torchvision.transforms.functional as F


def erase(img, i, j, h, w, v, inplace=False):
    if not isinstance(img, torch.Tensor):
        raise TypeError('img should be Tensor Image. Got {}'.format(type(img)))
    if not inplace:
        img = img.clone()
    img[:, i:i+h, j:j+w] = v
    return img


class RandomErasing(object):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn('range should be of kind (min, max)')
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError('range of scale should be between 0 and 1')
        if p < 0 or p > 1:
            raise ValueError('range of random erasing probability should be between 0 and 1')
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for attempt in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(0, img_h - h)
                j = random.randint(0, img_w - w)
                if isinstance(value, numbers.Number):
                    v = value
                elif isinstance(value, torch._six.string_classes):
                    v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                elif isinstance(value, (list, tuple)):
                    v = torch.tensor(value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
                return i, j, h, w, v
        return 0, 0, img_h, img_w, img

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value)
            return erase(img, x, y, h, w, v, self.inplace)
        return img


class BatchTrainingAug(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=Image.BILINEAR):
        super(BatchTrainingAug, self).__init__(size, scale, ratio, interpolation)
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        out = [F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
               for img in imgs]
        if random.random() < 0.5:
            out = [F.hflip(img) for img in out]
        return [self.transformer(img) for img in out]

    def process_one(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        out = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        if random.random() < 0.5:
            out = F.hflip(out)
        return self.transformer(out)


class BatchTrainingAug2(transforms.RandomCrop):
    def __init__(self, size):
        super(BatchTrainingAug2, self).__init__(size)
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.size)
        out = [F.crop(img, i, j, h, w) for img in imgs]
        if random.random() < 0.5:
            out = [F.hflip(img) for img in out]
        return [self.transformer(img) for img in out]

    def process_one(self, img):
        i, j, h, w = self.get_params(img, self.size)
        out = F.crop(img, i, j, h, w)
        if random.random() < 0.5:
            out = F.hflip(out)
        return self.transformer(out)


class BatchTrainingAug3(object):
    def __init__(self, crop_size):
        self.transformer = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def __call__(self, imgs):
        if random.random() < 0.5:
            imgs = [F.hflip(img) for img in imgs]
        return [self.transformer(img) for img in imgs]

    def process_one(self, img):
        if random.random() < 0.5:
            img = F.hflip(img)
        return self.transformer(img)


class BatchTestingAug(object):
    def __init__(self, size):
        self.size = size
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def __call__(self, imgs):
        out = [F.center_crop(img, self.size) for img in imgs]
        return [self.transformer(img) for img in out]

    def process_one(self, img):
        out = F.center_crop(img, self.size)
        return self.transformer(out)

class TrainingTransformer(object):
    def __init__(self, crop_size):
        self.transformer = transforms.Compose([
                # transforms.RandomResizedCrop(crop_size, scale=resize_scale),
                transforms.RandomCrop(crop_size),
                # transforms.Resize(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

    def __call__(self, img):
        return self.transformer(img)


class TestingTransformer(object):
    def __init__(self, crop_size):
        self.transformer = transforms.Compose([
            # transforms.CenterCrop(crop_size),
            transforms.Resize(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def __call__(self, img):
        return self.transformer(img)


class TestingTransformerTTA(object):
    def __init__(self, crop_size):
        self.transformer = transforms.Compose([
            transforms.FiveCrop(crop_size),
            transforms.Lambda(lambda crops: torch.stack(
                [self.to_tensor(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack(
                [self.normalize(crop) for crop in crops]))
        ])

    def __call__(self, img):
        return self.transformer(img)