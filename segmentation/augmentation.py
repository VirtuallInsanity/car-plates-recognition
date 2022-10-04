import albumentations as al

from config import BaseConfig


def get_train_augmentation(
    config: BaseConfig,
) -> al.Compose:
    return al.Compose(
        [
            al.PadIfNeeded(
                min_height=config.image_height,
                min_width=config.image_width,
                border_mode=0,
                value=0,
                mask_value=0,
                always_apply=True,
            ),
            al.CropNonEmptyMaskIfExists(
                height=config.image_height,
                width=config.image_width,
            ),
            al.Perspective(
                pad_mode=0,
            ),
            al.OneOf(
                [
                    al.Blur(blur_limit=7, p=1.0),
                    al.MotionBlur(blur_limit=7, p=1.0),
                ],
                p=0.8,
            ),
        ],
    )


def get_val_augmentation(
    config: BaseConfig,
) -> al.Compose:
    return al.Compose(
        [
            al.PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=config.height_divisor,
                pad_width_divisor=config.width_divisor,
                border_mode=0,
                value=0,
                mask_value=0,
                always_apply=True,
            ),
        ],
    )
