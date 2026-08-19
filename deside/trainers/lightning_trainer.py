from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

try:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ImportError:  # pragma: no cover - compatibility for older environments
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger


def loss_fn_mae_rmse_torch(y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    mae = torch.mean(torch.abs(y_true - y_pred))
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    return alpha * mae + (1 - alpha) * rmse


@dataclass
class DeSideTrainingHistory:
    history_df: pd.DataFrame
    best_model_path: str
    last_model_path: str


class DeSideLightningModule(L.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float, loss_alpha: float = 0.5):
        super().__init__()
        self.model = model
        self.learning_rate = float(learning_rate)
        self.loss_alpha = float(loss_alpha)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(
            gep=batch["gep"],
            pathway_profile=batch.get("pathway_profile"),
        )

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        y_true = batch["cell_fraction"]
        y_pred = self(batch)
        mae = torch.mean(torch.abs(y_true - y_pred))
        rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
        loss = self.loss_alpha * mae + (1 - self.loss_alpha) * rmse
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=(stage == "val"))
        self.log(f"{stage}_mae", mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_rmse", rmse, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate)


def _build_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _history_from_metrics(metrics_csv_path: Path) -> pd.DataFrame:
    metrics_df = pd.read_csv(metrics_csv_path)
    if metrics_df.empty:
        return pd.DataFrame(columns=["epoch", "loss", "val_loss", "mae", "val_mae", "rmse", "val_rmse"])

    history_df = (
        metrics_df.groupby("epoch", dropna=True)
        .agg(
            train_loss=("train_loss", "max"),
            val_loss=("val_loss", "max"),
            train_mae=("train_mae", "max"),
            val_mae=("val_mae", "max"),
            train_rmse=("train_rmse", "max"),
            val_rmse=("val_rmse", "max"),
        )
        .reset_index()
    )
    history_df.rename(
        columns={
            "train_loss": "loss",
            "train_mae": "mae",
            "train_rmse": "rmse",
        },
        inplace=True,
    )
    return history_df


def train_deside_lightning(
    *,
    model: nn.Module,
    train_dataset,
    val_dataset,
    learning_rate: float,
    loss_alpha: float,
    model_dir: str,
    max_epochs: int,
    batch_size: int,
    callback: bool,
    patience: int,
    verbose: int,
) -> DeSideTrainingHistory:
    pl_module = DeSideLightningModule(
        model=model,
        learning_rate=learning_rate,
        loss_alpha=loss_alpha,
    )
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    train_loader = _build_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = _build_loader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename="best_model",
        monitor="val_loss" if val_loader is not None else "train_loss",
        mode="min",
        save_top_k=1,
    )
    callbacks = [checkpoint_callback]
    if callback and val_loader is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=int(patience),
                mode="min",
            )
        )

    logger = CSVLogger(save_dir=model_dir, name="training_logs")

    def _probe_cuda_usable() -> bool:
        """Best-effort preflight: returns True only if a CUDA kernel actually runs.

        Lightning ``accelerator="auto"`` picks CUDA whenever ``torch.cuda.is_available()`` is True,
        even if the installed torch wheel was built against a CUDA compute capability that the
        current GPU does not implement (e.g. cpu-only wheel in a GPU env, or cu124 wheel on a
        Kepler-era card). That situation raises ``cudaErrorNoKernelImageForDevice`` the first time
        any kernel launches (e.g. the first DataLoader length probe in ``_run_sanity_check``),
        surfacing as a cryptic ``AcceleratorError`` at ``Sanity Checking: 0/?`` instead of a
        usable error message. This probe runs a single tiny matmul on device 0 before building
        the Trainer so we can fall back to CPU cleanly.
        """
        import os as _os

        if _os.environ.get("DESIDE_FORCE_CUDA", "0") not in ("0", "", "false", "False", "no", "No"):
            return True
        if not torch.cuda.is_available():
            return False
        try:
            dev = torch.device("cuda:0")
            a = torch.empty(2, 2, device=dev, dtype=torch.float32)
            b = torch.empty(2, 2, device=dev, dtype=torch.float32)
            _ = (a @ b).sum().item()
            del a, b
            return True
        except Exception as exc:  # pragma: no cover - HPC env specific
            msg = str(exc).lower()
            is_cuda_mismatch = any(
                tag in msg for tag in (
                    "no kernel image", "nokernelfordevice", "cuda error",
                    "invalid device function", "kernel image",
                )
            )
            print(
                f"[WARN] CUDA probe failed ({type(exc).__name__}: {exc}).\n"
                f"       This usually means the installed torch wheel was not built for this GPU's\n"
                f"       compute capability (cudaErrorNoKernelImageForDevice). Falling back to CPU\n"
                f"       training for this run. To force CUDA (hard-fail on probe), set DESIDE_FORCE_CUDA=1.\n"
                + (
                    ""
                    if is_cuda_mismatch
                    else "       (probe raised an unexpected exception class — check cuda-memcheck / drivers.)\n"
                ),
                flush=True,
            )
            try:
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
            except Exception:
                pass
            return False

    _cuda_ok = _probe_cuda_usable()
    _accelerator = "cuda" if _cuda_ok else "cpu"
    _devices = 1 if _cuda_ok else "auto"
    if not _cuda_ok:
        print(
            "[INFO] Using accelerator='cpu' for this run. Install a CUDA-enabled torch build matching\n"
            "       your GPU to regain GPU acceleration (see 'Install PyTorch' on pytorch.org).",
            flush=True,
        )
    trainer = L.Trainer(
        max_epochs=int(max_epochs),
        accelerator=_accelerator,
        devices=_devices,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=verbose > 0,
        enable_model_summary=verbose > 0,
        log_every_n_steps=1,
    )
    trainer.fit(
        pl_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    best_model_path = checkpoint_callback.best_model_path
    last_model_path = str(model_dir_path / "last_model.ckpt")
    trainer.save_checkpoint(last_model_path)

    metrics_csv_path = Path(logger.log_dir) / "metrics.csv"
    history_df = _history_from_metrics(metrics_csv_path)
    history_out = model_dir_path / "history_reg.csv"
    history_df.to_csv(history_out, index=False)

    metrics_out = model_dir_path / "metrics.csv"
    if metrics_csv_path.exists():
        shutil.copy2(metrics_csv_path, metrics_out)

    return DeSideTrainingHistory(
        history_df=history_df,
        best_model_path=best_model_path,
        last_model_path=last_model_path,
    )


def load_model_state_from_checkpoint(model: nn.Module, checkpoint_path: str, map_location: str = "cpu") -> nn.Module:
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
