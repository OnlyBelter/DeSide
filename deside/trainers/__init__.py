from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from torch import nn

if TYPE_CHECKING:
    from . import lightning_trainer as _lightning_trainer
    from .lightning_trainer import (
        DeSideLightningModule,
        train_deside_lightning,
    )

__all__ = [
    "DeSideLightningModule",
    "DeSideTrainingHistory",
    "train_deside_lightning",
    "load_model_state_from_checkpoint",
]


@dataclass
class DeSideTrainingHistory:
    history_df: pd.DataFrame
    best_model_path: str
    last_model_path: str


def load_model_state_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    map_location: Any = "cpu",
) -> nn.Module:
    checkpoint_path = str(Path(checkpoint_path).expanduser())
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned_state_dict[key[len("model."):]] = value
        else:
            cleaned_state_dict[key] = value
    model.load_state_dict(cleaned_state_dict)
    model.eval()
    return model


def _load_trainer_submodule():
    from . import lightning_trainer

    return lightning_trainer


def __getattr__(name: str):
    if name in {"DeSideLightningModule", "train_deside_lightning"}:
        trainer_module = _load_trainer_submodule()
        if name == "DeSideLightningModule":
            return trainer_module.DeSideLightningModule
        if name == "train_deside_lightning":
            return trainer_module.train_deside_lightning
    if name == "_lightning_trainer":
        return _load_trainer_submodule()
    raise AttributeError(f"module 'deside.trainers' has no attribute {name!r}")

