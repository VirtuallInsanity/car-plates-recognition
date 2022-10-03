import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

import config as configs
from augmentation import get_val_augmentation
from preprocessing import get_preprocessing


def main():
    config = configs.BaseConfig()
    args = parse_arguments()

    device = torch.device(args.device_name)
    model = load_model(args.checkpoint_filepath, device)
    model.eval()

    image = cv2.imread(args.image_filepath)
    preprocessed_image = preprocess_image(config, image)

    with torch.no_grad():
        outputs = model.predict(preprocessed_image).squeeze().cpu().numpy()
        mask = (outputs > config.metric_threshold).astype(np.uint8)
    for idx, box in enumerate(get_boxes_form_mask(mask)):
        output_size = (config.output_image_width, config.output_image_height)
        output_image = crop_image(image, order_points(box), output_size)
        _, extension = os.path.basename(args.image_filepath).split('.')
        save_image(
            output_image,
            '{0}.{1}'.format(
                idx,
                extension,
            ),
            args.output_dir,
        )


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        '-d',
        help='device name',
        type=str,
        default='cpu',
        dest='device_name',
    )
    parser.add_argument(
        '-o',
        help='output dir',
        type=str,
        required=True,
        dest='output_dir',
    )
    parser.add_argument(
        '-i',
        help='input image filepath',
        type=str,
        required=True,
        dest='image_filepath',
    )
    parser.add_argument(
        '-c',
        help='model checkpoint path',
        type=str,
        required=True,
        dest='checkpoint_filepath',
    )
    return parser.parse_args()


def load_model(checkpoint_filepath, device):
    return torch.load(
        checkpoint_filepath,
        map_location=device,
    )


def preprocess_image(config, image):
    augmentation = get_val_augmentation(config)
    preprocessing = get_preprocessing(config)
    sample = preprocessing(
        image=augmentation(image=image)['image'],
    )['image']
    return sample.unsqueeze(0).float()


def get_boxes_form_mask(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    boxes = []
    for contour in contours:
        boxes.append(
            cv2.boxPoints(
                cv2.minAreaRect(contour),
            ),
        )
    return boxes


def order_points(points):
    rect = np.zeros((4, 2))
    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def prepare_dirs(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def crop_image(image, box, output_size):
    dst = np.asarray(
        [
            [0, 0],
            [output_size[0], 0],
            [output_size[0], output_size[1]],
            [0, output_size[1]],
        ],
    )
    homography, _ = cv2.findHomography(order_points(box), dst)
    return cv2.warpPerspective(
        image,
        homography,
        dsize=output_size,
    )


def save_image(
    image,
    filename,
    output_dir,
):
    filepath = os.path.join(
        output_dir,
        filename,
    )
    cv2.imwrite(filepath, image)


if __name__ == '__main__':
    main()
