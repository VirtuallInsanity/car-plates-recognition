# -*- coding: utf-8 -*-
"""use_model.py

Show text's result OCR for one picture with car plate.
Use keras pretrained model.
"""
import sys

import cv2
import numpy as np
from keras.models import model_from_json


def find_contours(dimensions, img):
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


def segment_characters(image):
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


def load_keras_model(model_name):
    """
 Loading pretrained model
 :param model_name: Name .h5, .json formats files pretrained model
 :returns: pretrained model
 """
    json_file = open(
        'OCR/CNN/{}.json'.format(model_name),
        'r',
    )  # path to model-json
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('OCR/CNN/{}.h5'.format(model_name)
                       )  # path to model weight
    return model


def fix_dimension(img):
    """
 Fix dimension every character's image
 :param img: character's image
 :returns: character's image with fixed dimension
 """
    new_img = np.zeros((28, 28, 3))
    for i in range(3):
        new_img[:, :, i] = img
    return new_img


def show_results(char):
    """
 OCR and show text's result
 :param char: array with letters symbols contours
 :returns: text's result OCR
 """
    dic = {}
    characters = '0123456789ABCEHKMOPTXY'
    for i, c in enumerate(characters):
        dic[i] = c

    output = []
    for i, ch in enumerate(char):
        img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)
        y_prob = model.predict(img)
        y_ = y_prob.argmax(axis=-1)
        character = dic[y_[0]]
        if (y_[0] in (0, 17)):  # '0' or 'O'
            if (i in (0, 4, 5)):  # letters
                character = dic[17]  # 'O'
            else:
                character = dic[0]  # '0'
        output.append(character)

    plate_number = ''.join(output)

    return plate_number


# load pretrained model
pre_trained_model = load_keras_model('model_LicensePlate_2')
model = pre_trained_model

if __name__ == '__main__':
    if len(sys.argv) == 2:
        raw_path = sys.argv[1]  # path
        # \ replace /, delete ""
        path = raw_path.replace('\\', '/').replace('"', '')

        img = cv2.imread(path)
        img = cv2.resize(img, (400, 90))
        char = segment_characters(img)
        print(show_results(char))
    else:
        print('Не указан путь!')
