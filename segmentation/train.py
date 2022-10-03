import json
import os

import torch
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch

import config as configs
from data import get_data_loaders, get_datasets
from augmentation import get_train_augmentation, get_val_augmentation
from model import get_loss, get_metrics, get_model, get_optimizer
from preprocessing import get_preprocessing


def main():
    config = configs.BaseConfig()
    metadata = load_metadata(config)
    augmentation = {
        'train': get_train_augmentation(config),
        'val': get_val_augmentation(config),
    }
    preprocessing = {
        'train': get_preprocessing(config),
        'val': get_preprocessing(config),
    }
    train_dataset, val_dataset = get_datasets(
        config,
        metadata,
        augmentation,
        preprocessing,
    )
    train_loader, val_loader = get_data_loaders(
        config,
        train_dataset,
        val_dataset,
    )
    model = get_model(config)
    optimizer = get_optimizer(
        config,
        model.parameters(),
    )
    loss = get_loss()
    metrics = get_metrics(config)
    train_epoch = TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=config.device,
        verbose=config.verbose,
    )
    val_epoch = ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=config.device,
        verbose=config.verbose,
    )

    prepare_dirs(config)

    best_score = 0

    for _ in range(config.epochs):
        train_logs, val_logs = step_epoch(
            train_epoch,
            val_epoch,
            train_loader,
            val_loader,
        )
        if val_logs[config.metric] > best_score:
            best_score = val_logs[config.metric]
            save_model(
                config,
                model,
            )


def load_metadata(config):
    with open(
        config.metadata_filepath,
        'r',
        encoding='utf-8',
    ) as metadata_file:
        return json.load(metadata_file)


def save_model(config, model):
    torch.save(
        model,
        os.path.join(
            config.checkpoint_dir,
            config.checkpoint_filename,
        ),
    )


def prepare_dirs(config):
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)


def step_epoch(
    train_epoch,
    val_epoch,
    train_loader,
    val_loader,
):
    train_logs = train_epoch.run(train_loader)
    valid_logs = val_epoch.run(val_loader)
    return train_logs, valid_logs


if __name__ == '__main__':
    main()
