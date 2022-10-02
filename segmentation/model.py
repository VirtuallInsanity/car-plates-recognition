import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import Fscore, IoU
from torch.optim import AdamW


def get_model(config):
    if config.segmentation_model == 'unet':
        return smp.Unet(
            **config.get_model_config(),
        )
    else:
        NotImplementedError('Unexpected segmentation model class.')


def get_optimizer(config, model_parameters):
    return AdamW(
        params=model_parameters,
        lr=config.learning_rate,
    )


def get_loss():
    return DiceLoss()


def get_metrics(config):
    return [
        IoU(threshold=config.metric_threshold),
        Fscore(threshold=config.metric_threshold),
    ]
