import os
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms

class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image):
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        return image

class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image):
        if not isinstance(image, np.ndarray) :
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image

class test_Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        self.size = size

    def __call__(self, image):

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]

        def find_new_hw(ori_h, ori_w, test_size):
            if max(ori_h, ori_w) > test_size:
                if ori_h >= ori_w:
                    ratio = test_size * 1.0 / ori_h
                    new_h = test_size
                    new_w = int(ori_w * ratio)
                elif ori_w > ori_h:
                    ratio = test_size * 1.0 / ori_w
                    new_h = int(ori_h * ratio)
                    new_w = test_size

                if new_h % 8 != 0:
                    new_h = (int(new_h / 8)) * 8
                else:
                    new_h = new_h
                if new_w % 8 != 0:
                    new_w = (int(new_w / 8)) * 8
                else:
                    new_w = new_w
                return new_h, new_w
            else:
                return ori_h, ori_w

        test_size = self.size
        new_h, new_w = find_new_hw(image.shape[0], image.shape[1], test_size)
        if new_w != image.shape[0] or new_h != image.shape[1]:
            image_crop = cv2.resize(image, dsize=(int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)
        else:
            image_crop = image.copy()
        back_crop = np.zeros((test_size, test_size, 3))
        back_crop[:new_h, :new_w, :] = image_crop
        image = back_crop

        return image


def visualize(image_path, label_path):
    # test_Resize_function = test_Resize(400)
    # ToTensor_function = ToTensor()
    # Normalize_function = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_Resize_function = transforms.Resize((400,400))
    ToTensor_function = transforms.ToTensor()
    Normalize_function = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


    #image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = np.float32(image) / 255.
    image = Image.open(image_path).convert('RGB')
    image = test_Resize_function(image)
    image = ToTensor_function(image)
    img = Normalize_function(image)

    img[0] = img[0] * 0.229 + 0.485
    img[1] = img[1] * 0.224 + 0.456
    img[2] = img[2] * 0.225 + 0.406

    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    mask = np.array(label)
    # mask = torch.from_numpy(mask)

    img = img.numpy()
    # colorize
    img[0, ...] = np.where(mask != 0, 255, img[0, ...])
    img[1, ...] = np.where(mask != 0, img[1, ...] * 0.7, img[1, ...])
    img[2, ...] = np.where(mask != 0, img[2, ...] * 0.7, img[2, ...])

    img = torch.tensor(img)

    return img