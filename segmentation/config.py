import torch


class BaseConfig(object):
    # core
    classes = ['car']
    seed = 42

    # training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epochs = 10
    val_size = 0.05
    learning_rate = 1e-4
    verbose = True

    # images
    in_channels = 3
    image_height = 320
    image_width = 320
    height_divisor = 32
    width_divisor = 32
    output_image_height = 112
    output_image_width = 520
    min_output_image_height = 16
    min_output_image_width = 16

    # model
    segmentation_model = 'unet'
    encoder_name = 'resnet18'
    encoder_weights = 'imagenet'
    activation = 'sigmoid'

    # data loading
    train_batch_size = 4
    val_batch_size = 1
    shuffle_train = True
    shuffle_val = False
    drop_last_train = True
    drop_last_val = False
    num_workers = 0

    # filepaths
    checkpoint_dir = 'checkpoints'
    checkpoint_filename = 'best_model.pth'
    metadata_filepath = 'data/train.json'
    image_dir = 'data/images'
    mask_dir = 'data/masks'

    # metrics
    metric_threshold = 0.5
    metric = 'iou_score'

    def get_model_config(self):
        return {
            'encoder_name': self.encoder_name,
            'encoder_weights': self.encoder_weights,
            'activation': self.activation,
            'classes': len(self.classes),
            'in_channels': self.in_channels,
        }
