from typing import Iterable

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.base.model import SegmentationModel
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW, Optimizer

from segmentation.config import BaseConfig


def get_model(config: BaseConfig) -> SegmentationModel:
    if config.segmentation_model == 'unet':
        return smp.Unet(
            **config.get_model_config(),
        )
    elif config.segmentation_model == 'deeplabv3+':
        return smp.DeepLabV3Plus(
            **config.get_model_config(),
        )
    else:
        NotImplementedError('Unexpected segmentation model class.')


def get_optimizer(
    config: BaseConfig,
    model_parameters: Iterable[torch.Tensor],
) -> Optimizer:
    return AdamW(
        params=model_parameters,
        lr=config.learning_rate,
    )


def get_loss() -> torch.nn.Module:
    return BCEWithLogitsLoss()
