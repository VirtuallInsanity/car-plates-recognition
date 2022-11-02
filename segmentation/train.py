"""Train logic."""
import json
import logging
import os
from typing import Dict, List, Tuple

import segmentation_models_pytorch as smp
import torch
import tqdm
from augmentation import get_train_augmentation, get_val_augmentation
from model import get_loss, get_model, get_optimizer
from preprocessing import get_preprocessing
from torch.utils.data import DataLoader

from segmentation import config as configs
from segmentation.config import BaseConfig
from segmentation.data import get_data_loaders, get_datasets


def main() -> None:
    """Train segmentation model."""
    logger = get_logger()
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
    criterion = get_loss()

    prepare_dirs(config)

    model.to(config.device)

    best_val_score: float = 0

    for epoch in range(config.epochs):
        logger.info(
            'training, epoch {0}'.format(
                epoch + 1,
            ),
        )
        train_stats, train_loss = train_epoch(
            config,
            model,
            optimizer,
            criterion,
            train_loader,
        )
        f1_score_train = calculate_metrics(train_stats)
        logger.info(
            'train loss: {0}, train F1 score: {1}'.format(
                round(train_loss, 5),
                round(f1_score_train, 5),
            ),
        )
        logger.info(
            'validation, epoch {0}'.format(
                epoch + 1,
            ),
        )
        val_stats, val_loss = val_epoch(
            config,
            model,
            criterion,
            val_loader,
        )
        f1_score_val = calculate_metrics(val_stats)
        logger.info(
            'val loss: {0}, val F1 score: {1}'.format(
                round(val_loss, 5),
                round(f1_score_val, 5),
            ),
        )
        if f1_score_val > best_val_score:
            logger.info('saving model...')
            save_model(config, model)
            best_val_score = f1_score_val


def load_metadata(config: BaseConfig) -> List[Dict]:
    """Load metadata.

    :param config: experiment config
    :return: data about samples
    """
    with open(
        config.metadata_filepath,
        'r',
        encoding='utf-8',
    ) as metadata_file:
        return json.load(metadata_file)


def save_model(
    config: BaseConfig,
    model: torch.nn.Module,
) -> None:
    """Save given model.

    :param config: experiment config
    :param model: model to save
    """
    torch.save(
        model,
        os.path.join(
            config.checkpoint_dir,
            config.checkpoint_filename,
        ),
    )


def prepare_dirs(config: BaseConfig) -> None:
    """Prepare dirs for model training.

    :param config: experiment config
    """
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)


def train_epoch(
    config: BaseConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    data_loader: DataLoader,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Train one epoch.

    :param config: experiment config
    :param model: segmentation model
    :param optimizer: optimizer
    :param criterion: loss function
    :param data_loader: data loader
    :return: training metrics
    """
    model.train()
    total_loss = 0
    prediction_types = ['tp', 'fp', 'fn', 'tn']
    stats = {
        prediction_type: torch.Tensor([0])
        for prediction_type in prediction_types
    }
    for batch in tqdm.tqdm(data_loader):
        images, masks = batch
        images = images.to(config.device)
        masks = masks.to(config.device).unsqueeze(1)
        outputs = model(images)

        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        stats = update_stats(config, outputs, masks, stats)
    return stats, total_loss / len(data_loader)


def val_epoch(
    config: BaseConfig,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: DataLoader,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Validate one epoch.

    :param config: experiment config
    :param model: segmentation model
    :param criterion: loss function
    :param data_loader: data loader
    :return: validation metrics
    """
    model.eval()
    total_loss = 0
    prediction_types = ['tp', 'fp', 'fn', 'tn']
    stats = {
        prediction_type: torch.Tensor([0])
        for prediction_type in prediction_types
    }
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader):
            images, masks = batch
            images = images.to(config.device)
            masks = masks.to(config.device).unsqueeze(1)
            outputs = model(images)

            loss = criterion(outputs, masks)
            total_loss += loss.item()

            stats = update_stats(config, outputs, masks, stats)
    return stats, total_loss / len(data_loader)


def update_stats(
    config: BaseConfig,
    outputs: torch.Tensor,
    masks: torch.Tensor,
    stats: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Update stats after another step.

    :param config: experiment config
    :param outputs: predicted masks
    :param masks: GT masks
    :param stats: stats
    :return: updated stats
    """
    masks = masks.int()
    batch_stats = smp.metrics.get_stats(
        outputs,
        masks,
        mode='binary',
        threshold=config.metric_threshold,
    )
    stats['tp'] += batch_stats[0].sum()
    stats['fp'] += batch_stats[1].sum()
    stats['fn'] += batch_stats[2].sum()
    stats['tn'] += batch_stats[3].sum()
    return stats


def calculate_metrics(
    stats: Dict[str, torch.Tensor],
) -> float:
    """Calculate metrics.

    :param stats: dict with model predictions
    :return: metrics
    """
    return smp.metrics.f1_score(
        stats['tp'],
        stats['fp'],
        stats['fn'],
        stats['tn'],
        reduction='micro',
    ).item()


def get_logger() -> logging.Logger:
    """Get logger.

    :return: logger for experiment run
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


if __name__ == '__main__':
    main()
