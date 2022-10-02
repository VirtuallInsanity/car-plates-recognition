import albumentations as al
import segmentation_models_pytorch as smp
from albumentations.pytorch.transforms import ToTensorV2


def get_preprocessing(config):
    encoder_preprocessing = smp.encoders.get_preprocessing_fn(
        config.encoder_name,
        config.encoder_weights,
    )
    return al.Compose(
        [
            al.Lambda(image=encoder_preprocessing),
            ToTensorV2(),
        ],
    )
