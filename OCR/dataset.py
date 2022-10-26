# -*- coding: utf-8 -*-
"""dataset.py

Create letters symbols dataset from dataset
https://www.kaggle.com/datasets/evgrafovmaxim/nomeroff-russian-license-plates
"""
import json
import os

import cv2
import numpy as np


def find_contours(
    dimensions: list,
    img: numpy.ndarray,
) -> np.array:
    """
 Find letters symbols contours
 :param dimensions: allowed character size
 :param img: image after preprocessing
 :returns: array with letters symbols contours
 """
    cntrs, _ = cv2.findContours(
        img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
    )

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    max_count_cntrs = 15
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:max_count_cntrs]

    ii = cv2.imread('contour.jpg')

    # Finding suitable contours
    x_cntr_list = []
    img_res = []
    cntrs_int = []
    for cntr1 in cntrs:
        int_x, int_y, int_width, int_height = cv2.boundingRect(cntr1)
        if (lower_width < int_width < upper_width):
            if (lower_height < int_height < upper_height):
                cntrs_int.append(cntr1)

    # Deleting nested contours
    cntrs_last = []
    for cntr2 in cntrs_int:
        int_x, int_y, int_width, int_height = cv2.boundingRect(cntr2)
        point = (int_x, int_y)
        cntrs_last.append(cntr2)
        for cntrout in cntrs_int:
            if (cv2.boundingRect(cntrout) != cv2.boundingRect(cntr2)):
                if (cv2.pointPolygonTest(cntrout, point) >= 0):
                    cntrs_last.pop()
                    break

    for cntr3 in cntrs_last:
        int_x, int_y, int_width, int_height = cv2.boundingRect(cntr3)

        x_cntr_list.append(int_x)

        char_copy = np.zeros((44, 24))
        char = img[int_y:int_y + int_height, int_x:int_x + int_width]
        char = cv2.resize(char, (20, 40))
        cv2.rectangle(
            ii, (int_x, int_y),
            (int_width + int_x, int_y + int_height),
            (50, 21, 200), 2,
        )
        char = cv2.subtract(255, char)

        char_copy[2:42, 2:22] = char
        char_copy[0:2, :] = 0
        char_copy[:, 0:2] = 0
        char_copy[42:44, :] = 0
        char_copy[:, 22:24] = 0

        img_res.append(char_copy)

    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])

    return np.array(img_res_copy)


def segment_characters(
    image: numpy.ndarray,
) -> np.array:
    """
 Segment characters and find letters symbols contours
 :param image: original image
 :returns: array with letters symbols contours
 """
    img_lp = cv2.resize(image, (333, 75))
    img_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_lp = cv2.threshold(
        img_lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    img_lp = cv2.erode(img_lp, (3, 3))
    img_lp = cv2.dilate(img_lp, (3, 3))
    img_lp = cv2.dilate(img_lp, (3, 3))

    lp_width = img_lp.shape[0]
    lp_height = img_lp.shape[1]
    # cutting edges
    img_lp[0:10, :] = 255
    img_lp[:, 0:20] = 255
    img_lp[72:75, :] = 255
    img_lp[:, 330:333] = 255
    # allowed character size
    dimensions = [
        lp_width / 8,
        lp_width / 2,
        lp_height / 15,
        5 * lp_height / 6,
        ]

    return find_contours(dimensions, img_lp)


train_symbols = []
path = './autoriaNumberplateOcrRu-2021-09-01/train/'
path_json = '{0}ann/'.format(path)
for dirname, _, filenames in os.walk('{0}img'.format(path)):
    for filename in filenames:
        name = json.load(open(path_json + filename.replace('.png', '.json')))
        train_symbols.append(
            os.path.join(dirname, filename),
            name['description'],
        )
# create train dataset
for index, _ in enumerate(train_symbols):
    char = segment_characters(cv2.imread(train_symbols[index][0]))
    if len(char) == (len(train_symbols[index][1])):
        for in_index, ch in enumerate(train_symbols[index][1]):
            if (os.path.exists('./train/class_{0}'.format(ch)) is False):
                os.mkdir('./train/class_{0}'.format(ch))
            cv2.imwrite(
                './train/class_' +
                train_symbols[index][1][in_index] +
                '/' +
                train_symbols[index][1] +
                '_' +
                str(in_index) +
                '.jpg',
                char[in_index],
            )

val_symbols = []
path = './autoriaNumberplateOcrRu-2021-09-01/val/'
path_json = '{0}ann/'.format(path)
for dirname, _, filenames in os.walk('{0}img'.format(path)):
    for filename in filenames:
        name = json.load(open(path_json + filename.replace('.png', '.json')))
        val_symbols.append(
            os.path.join(dirname, filename),
            name['description'],
        )
# create train dataset
for index, _ in enumerate(val_symbols):
    char = segment_characters(cv2.imread(val_symbols[index][0]))
    if len(char) == (len(val_symbols[index][1])):
        for in_index, ch in enumerate(val_symbols[index][1]):
            if (os.path.exists('./val/class_{0}'.format(ch)) is False):
                os.mkdir('./val/class_{0}'.format(ch))
            cv2.imwrite(
                './val/class_' +
                val_symbols[index][1][in_index] +
                '/' +
                val_symbols[index][1] +
                '_' +
                str(in_index) +
                '.jpg',
                char[in_index],
                )

test_symbols = []
path = './autoriaNumberplateOcrRu-2021-09-01/test/'
path_json = '{0}ann/'.format(path)
for dirname, _, filenames in os.walk('{0}img'.format(path)):
    for filename in filenames:
        name = json.load(open(path_json + filename.replace('.png', '.json')))
        test_symbols.append(
            os.path.join(dirname, filename),
            name['description'],
        )
# create train dataset
for index, _ in enumerate(test_symbols):
    char = segment_characters(cv2.imread(test_symbols[index][0]))
    if len(char) == (len(test_symbols[index][1])):
        for in_index, ch in enumerate(test_symbols[index][1]):
            if (os.path.exists('./test/class_{0}'.format(ch)) is False):
                os.mkdir('./test/class_{0}'.format(ch))
            cv2.imwrite(
                './test/class_' +
                test_symbols[index][1][in_index] +
                '/' +
                test_symbols[index][1] +
                '_' +
                str(in_index) +
                '.jpg',
                char[in_index],
                )
