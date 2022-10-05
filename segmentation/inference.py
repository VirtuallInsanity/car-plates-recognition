import os
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from segmentation import config as configs
from segmentation.augmentation import get_val_augmentation
from segmentation.config import BaseConfig
from segmentation.preprocessing import get_preprocessing


def predict(
    image_filepath: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_filepath: Optional[str] = None,
    device_name: str = 'cpu',
) -> List[str]:
    config = configs.BaseConfig()
    console_run = (
        image_filepath is None and
        output_dir is None and
        checkpoint_filepath is None
    )
    if console_run:
        args = parse_arguments()
        image_filepath = args.image_filepath
        output_dir = args.output_dir
        checkpoint_filepath = args.checkpoint_filepath
        device_name = args.device_name

    device = torch.device(device_name)
    model = load_model(checkpoint_filepath, device)
    model.eval()

    image = cv2.imread(image_filepath)
    preprocessed_image = preprocess_image(config, image).to(device)

    with torch.no_grad():
        outputs = model.predict(preprocessed_image).squeeze().cpu().numpy()
        mask = (outputs > config.metric_threshold).astype(np.uint8)
    boxes = get_boxes_form_mask(
        mask,
        config.min_output_image_width,
        config.min_output_image_height,
    )
    output_filepaths = []
    for idx, box in enumerate(boxes):
        output_size = (config.output_image_width, config.output_image_height)
        output_image = crop_image(image, order_points(box), output_size)
        _, extension = os.path.basename(image_filepath).split('.')
        output_filepath = save_image(
            output_image,
            '{0}.{1}'.format(
                idx,
                extension,
            ),
            output_dir,
        )
        output_filepaths.append(output_filepath)
    return output_filepaths


def parse_arguments() -> Namespace:
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


def load_model(
    checkpoint_filepath: str,
    device: torch.device,
) -> torch.nn.Module:
    return torch.load(
        checkpoint_filepath,
        map_location=device,
    )


def preprocess_image(
    config: BaseConfig,
    image: np.ndarray,
) -> torch.FloatTensor:
    augmentation = get_val_augmentation(config)
    preprocessing = get_preprocessing(config)
    aug_image = augmentation(image=image)['image']
    pre_image = preprocessing(image=aug_image)['image']
    return pre_image.unsqueeze(0).float()


def get_boxes_form_mask(
    mask: np.ndarray,
    min_width: int,
    min_height: int,
) -> List[np.ndarray]:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    boxes = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width > min_width and height > min_height:
            boxes.append(
                cv2.boxPoints(rect),
            )
    return boxes


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2))
    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def prepare_dirs(output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def crop_image(
    image: np.ndarray,
    box: np.ndarray,
    output_size: Tuple[int, int],
) -> np.ndarray:
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
        flags=cv2.INTER_LANCZOS4,
    )


def save_image(
    image: np.ndarray,
    filename: str,
    output_dir: str,
) -> str:
    filepath = os.path.join(
        output_dir,
        filename,
    )
    cv2.imwrite(filepath, image)
    return filepath


if __name__ == '__main__':
    predict()
