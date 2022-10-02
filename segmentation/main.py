import json

import cv2
import numpy

from segmentation.augmentation import (get_train_augmentation,
                                       get_val_augmentation)
from segmentation.config import BaseConfig
from segmentation.data import Dataset
from segmentation.preprocessing import get_preprocessing

with open('../data/train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


# min_width = math.inf
# min_height = math.inf
#
# max_width = 0
# max_height = 0
#
# for idx in range(len(dataset)):
#     image, mask = dataset[idx]
#     width = image.shape[0]
#     height = image.shape[1]
#     min_width = min(width, min_width)
#     min_height = min(height, min_height)
#     max_width = max(width, max_width)
#     max_height = max(height, max_height)
#
# print(min_width, min_height)
# print(max_width, max_height)

config = BaseConfig()

preprocessing = get_preprocessing(config)

valid_transforms = get_val_augmentation(config)
train_transforms = get_train_augmentation(config)

dataset = Dataset(
    image_dir='../data/images',
    mask_dir='../data/masks',
    metadata=data[:10],
    augmentation=train_transforms,
    preprocessing=preprocessing,
)

image, mask = dataset[1]

print(image.float())
print(mask)
#
# print(image)

# cv2.imshow('before', cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(numpy.uint8))
#
# sample = train_transforms(image=image, mask=mask)
#
# print(sample['image'].shape)
# print(sample['mask'].shape)
#
# cv2.imshow('after', cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR).astype(numpy.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


