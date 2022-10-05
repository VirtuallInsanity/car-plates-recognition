import os
from typing import Dict, List, Optional, Tuple

import albumentations as al
import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm

from segmentation.config import BaseConfig


class Dataset(BaseDataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        metadata: List[Dict],
        augmentation: Optional[al.Compose] = None,
        preprocessing: Optional[al.Compose] = None,
     ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.metadata = metadata
        self.image_filepaths: List[str] = []
        self.mask_filepaths: List[str] = []
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self._prepare_dirs()
        self._process_masks()

    def __len__(self) -> int:
        return len(self.image_filepaths)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        image_filepath = self.image_filepaths[idx]
        mask_filepath = self.mask_filepaths[idx]
        image = cv2.imread(
            image_filepath,
        ).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(
            mask_filepath,
            cv2.IMREAD_GRAYSCALE,
        ).astype(np.float32)
        if self.augmentation is not None:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing is not None:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image.float(), mask.float()

    def _prepare_dirs(self) -> None:
        if not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir)

    def _process_masks(self) -> None:
        for sample in tqdm(self.metadata):
            filename = os.path.basename(sample['file'])
            image_filepath = os.path.join(
                self.image_dir,
                filename,
            )
            mask_filepath = os.path.join(
                self.mask_dir,
                filename,
            )
            image = cv2.imread(image_filepath)
            if image is not None:
                mask = np.zeros(
                    shape=image.shape[:2],
                    dtype=np.uint8,
                )
                for num in sample['nums']:
                    cv2.fillConvexPoly(
                        mask,
                        np.asarray(num['box']),
                        1.0,
                    )
                cv2.imwrite(mask_filepath, mask)
                self.image_filepaths.append(image_filepath)
                self.mask_filepaths.append(mask_filepath)


def get_data_loaders(
    config: BaseConfig,
    train_dataset: BaseDataset,
    val_dataset: BaseDataset,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=config.shuffle_train,
        num_workers=config.num_workers,
        drop_last=config.drop_last_train,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=config.shuffle_val,
        num_workers=config.num_workers,
        drop_last=config.drop_last_val,
    )
    return train_loader, val_loader


def get_datasets(
    config: BaseConfig,
    metadata: List[Dict],
    augmentation: Dict[str, al.Compose],
    preprocessing: Dict[str, al.Compose],
) -> Tuple[Dataset, Dataset]:
    train_metadata, val_metadata = train_test_split(
        metadata,
        test_size=config.val_size,
        random_state=config.seed,
    )
    train_dataset = Dataset(
        config.image_dir,
        config.mask_dir,
        train_metadata,
        augmentation['train'],
        preprocessing['train'],
    )
    val_dataset = Dataset(
        config.image_dir,
        config.mask_dir,
        val_metadata,
        augmentation['val'],
        preprocessing['val'],
    )
    return train_dataset, val_dataset
