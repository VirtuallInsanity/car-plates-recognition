from typing import Any, Dict, List

import torch


class BaseConfig(object):
    # core
    classes: List[str] = ['car']
    seed: int = 42

    # training
    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu',
    )
    epochs: int = 10
    val_size: float = 0.05
    learning_rate: float = 1e-4
    verbose: bool = True

    # images
    in_channels: int = 3
    image_height: int = 320
    image_width: int = 320
    height_divisor: int = 32
    width_divisor: int = 32
    output_image_height: int = 112
    output_image_width: int = 520
    min_output_image_height: int = 16
    min_output_image_width: int = 16

    # model
    segmentation_model: str = 'unet'
    encoder_name: str = 'resnet18'
    encoder_weights: str = 'imagenet'
    activation: str = 'sigmoid'

    # data loading
    train_batch_size: int = 4
    val_batch_size: int = 1
    shuffle_train: bool = True
    shuffle_val: bool = False
    drop_last_train: bool = True
    drop_last_val: bool = False
    num_workers: int = 0

    # filepaths
    checkpoint_dir: str = 'checkpoints'
    checkpoint_filename: str = 'best_model.pth'
    metadata_filepath: str = 'data/train.json'
    image_dir: str = 'data/images'
    mask_dir: str = 'data/masks'

    # metrics
    metric_threshold: float = 0.5
    metric: str = 'iou_score'

    def get_model_config(self) -> Dict[str, Any]:
        return {
            'encoder_name': self.encoder_name,
            'encoder_weights': self.encoder_weights,
            'activation': self.activation,
            'classes': len(self.classes),
            'in_channels': self.in_channels,
        }
