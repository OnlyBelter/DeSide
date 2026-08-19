from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


def _to_float_tensor(df_or_array) -> torch.Tensor:
    if isinstance(df_or_array, pd.DataFrame):
        values = df_or_array.values
    else:
        values = np.asarray(df_or_array)
    return torch.as_tensor(values, dtype=torch.float32)


class DeSideDataset(Dataset):
    """Dataset for DeSide regression training and inference."""

    def __init__(
        self,
        gep,
        cell_fraction=None,
        pathway_profile=None,
        sample_ids: Optional[list[str]] = None,
    ):
        self.gep = _to_float_tensor(gep)
        self.cell_fraction = None if cell_fraction is None else _to_float_tensor(cell_fraction)
        self.pathway_profile = None if pathway_profile is None else _to_float_tensor(pathway_profile)
        if sample_ids is None:
            sample_ids = [str(i) for i in range(self.gep.shape[0])]
        self.sample_ids = [str(i) for i in sample_ids]

    def __len__(self) -> int:
        return int(self.gep.shape[0])

    def __getitem__(self, index: int) -> dict:
        sample = {
            "gep": self.gep[index],
            "sample_id": self.sample_ids[index],
        }
        if self.pathway_profile is not None:
            sample["pathway_profile"] = self.pathway_profile[index]
        if self.cell_fraction is not None:
            sample["cell_fraction"] = self.cell_fraction[index]
        return sample


@dataclass
class DatasetSplit:
    train: Dataset
    val: Optional[Dataset]


def split_deside_dataset(
    dataset: DeSideDataset,
    validation_split: float = 0.2,
    seed: int = 42,
) -> DatasetSplit:
    n_samples = len(dataset)
    if n_samples == 0:
        raise ValueError("Cannot split an empty dataset.")
    if validation_split <= 0 or n_samples == 1:
        return DatasetSplit(train=dataset, val=None)

    n_val = int(round(n_samples * validation_split))
    n_val = min(max(n_val, 1), n_samples - 1)

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    val_indices = indices[:n_val].tolist()
    train_indices = indices[n_val:].tolist()
    return DatasetSplit(train=Subset(dataset, train_indices), val=Subset(dataset, val_indices))
