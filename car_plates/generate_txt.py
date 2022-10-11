import tqdm
import numpy as np
import cv2
from argparse import ArgumentParser
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "data_dir",
        help="Path to dir containing 'train/', 'test/', 'train.json'.")
    return parser.parse_args()


def main(args):
    """
    Create yolo annotation in .txt format for each image

    :param args: Path to dir
    """
    config_filename = os.path.join(args.data_dir, "train.json")
    with open(config_filename, "rt") as fp:
        config = json.load(fp)

    for item in tqdm.tqdm(config):
        new_item = {}
        new_item["file"] = item["file"]

        image_filename = item["file"]
        image_base, ext = os.path.splitext(image_filename)
        txt_filename = image_base + '.txt'

        nums = item["nums"]

        image = cv2.imread(os.path.join(args.data_dir, image_filename))

        if image is None:
            continue

        last_txt_filename = None

        for num in nums:
            bbox = np.asarray(num["box"])
            if bbox[0][1] <= bbox[1][1]:
                x_center = round(
                    ((bbox[1][0] - bbox[0][0]) / 2 + bbox[0][0]) / image.shape[1], 6)
                y_center = round(
                    ((bbox[2][1] - bbox[0][1]) / 2 + bbox[0][1]) / image.shape[0], 6)

                w = round((bbox[1][0] - bbox[0][0]) / image.shape[1], 6)
                h = round((bbox[2][1] - bbox[0][1]) / image.shape[0], 6)
            else:
                x_center = round(
                    ((bbox[1][0] - bbox[0][0]) / 2 + bbox[0][0]) / image.shape[1], 6)
                y_center = round(
                    ((bbox[3][1] - bbox[1][1]) / 2 + bbox[1][1]) / image.shape[0], 6)

                w = round((bbox[1][0] - bbox[0][0]) / image.shape[1], 6)
                h = round((bbox[3][1] - bbox[1][1]) / image.shape[0], 6)
            print(
                "x_center: {}, y_center: {}, width: {}, height: {}".format(
                    x_center, y_center, w, h))

            if last_txt_filename == image_base:
                with open(os.path.join(args.data_dir, txt_filename), "a+") as file:
                    file.write('0')
                    file.write(' ' + str(x_center))
                    file.write(' ' + str(y_center))
                    file.write(' ' + str(w))
                    file.write(' ' + str(h))
                    file.write('\n')
            else:
                with open(os.path.join(args.data_dir, txt_filename), "w") as file:
                    file.write('0')
                    file.write(' ' + str(x_center))
                    file.write(' ' + str(y_center))
                    file.write(' ' + str(w))
                    file.write(' ' + str(h))
                    file.write('\n')

            last_txt_filename = image_base


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
