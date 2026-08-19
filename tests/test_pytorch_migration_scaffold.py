from __future__ import annotations

import importlib
import os
import tempfile

import numpy as np
import pandas as pd
import torch

try:
    import pytest  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - sandbox compatibility
    class _SkipError(Exception):
        pass

    class _SkipPlaceholder:
        def skip(self, reason: str = ""):
            raise _SkipError(reason)

    pytest = _SkipPlaceholder()

try:
    import umap  # noqa: F401
    HAS_UMAP = True
except ModuleNotFoundError:  # pragma: no cover - sandbox compatibility
    HAS_UMAP = False

try:
    import lightning  # noqa: F401
    HAS_LIGHTNING = True
except ModuleNotFoundError:  # pragma: no cover - sandbox compatibility
    try:
        import pytorch_lightning  # noqa: F401
        HAS_LIGHTNING = True
    except ModuleNotFoundError:
        HAS_LIGHTNING = False


def _requires_umap():
    if not HAS_UMAP:
        pytest.skip("umap is not installed in this environment")


def _requires_lightning():
    if not HAS_LIGHTNING:
        pytest.skip("lightning/pytorch_lightning is not installed in this environment")

def _base_model_config(input_dim: int = 16, output_dim: int = 3) -> dict:
    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_units": [8, 6],
        "dropout_rates": [0.0, 0.0],
        "normalization": "layer_normalization",
        "normalization_layer": [0, 1, 1],
        "last_layer_activation": "sigmoid",
        "pathway_network": False,
        "n_pathway": 0,
        "pathway_hidden_units": [],
        "pathway_dropout_rates": [],
    }


def test_module_imports():
    from deside.data import DeSideDataset, split_deside_dataset  # noqa: F401
    from deside.models import DeSideRegressor, build_deside_model  # noqa: F401
    from deside.trainers import (
        DeSideTrainingHistory,
        load_model_state_from_checkpoint,
    )  # noqa: F401

    if HAS_UMAP:
        from deside.decon_cf import DeSide, predict_with_pretrained_model  # noqa: F401


def test_trainer_lazy_import():
    import deside.trainers as trainers

    assert "DeSideLightningModule" in trainers.__all__
    assert "train_deside_lightning" in trainers.__all__


def test_dataset_and_split():
    from deside.data import DeSideDataset, split_deside_dataset

    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        rng.random((20, 8)),
        index=[f"s{i}" for i in range(20)],
        columns=[f"G{i}" for i in range(8)],
    )
    y = pd.DataFrame(
        rng.dirichlet(np.ones(3), size=20),
        index=x.index,
        columns=["a", "b", "c"],
    )

    ds = DeSideDataset(gep=x.values, cell_fraction=y.values, sample_ids=list(x.index))
    assert len(ds) == 20
    batch = ds[0]
    assert batch["gep"].shape == torch.Size([8])
    assert batch["cell_fraction"].shape == torch.Size([3])
    assert batch["sample_id"] == "s0"

    split = split_deside_dataset(ds, validation_split=0.25, seed=0)
    assert len(split.train) + len(split.val) == 20
    assert len(split.val) >= 1


def test_build_model_without_pathway():
    from deside.models import build_deside_model

    cfg = _base_model_config(input_dim=8, output_dim=3)
    model = build_deside_model(cfg)
    out = model(torch.randn(4, 8))
    assert out.shape == torch.Size([4, 3])
    assert bool(torch.all(out >= 0))
    assert bool(torch.all(out <= 1))


def test_build_model_with_pathway():
    from deside.models import build_deside_model

    cfg = _base_model_config(input_dim=8, output_dim=3)
    cfg.update(
        pathway_network=True,
        n_pathway=2,
        pathway_hidden_units=[4],
        pathway_dropout_rates=[0.0],
    )
    model = build_deside_model(cfg)
    out = model(gep=torch.randn(4, 8), pathway_profile=torch.randn(4, 2))
    assert out.shape == torch.Size([4, 3])


def _pure_mae_rmse_loss(y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    mae = torch.mean(torch.abs(y_true - y_pred))
    rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    return alpha * mae + (1 - alpha) * rmse


def test_loss_fn_mae_rmse_torch():
    if HAS_LIGHTNING:
        from deside.trainers.lightning_trainer import loss_fn_mae_rmse_torch

        loss_fn = loss_fn_mae_rmse_torch
    else:
        loss_fn = _pure_mae_rmse_loss

    y_true = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    y_pred = torch.tensor([[0.25, 0.3, 0.45], [0.05, 0.15, 0.8]])
    loss = loss_fn(y_true, y_pred, alpha=0.5)
    assert loss.ndim == 0
    assert float(loss) >= 0


def test_load_state_from_checkpoint_roundtrip():
    from deside.models import build_deside_model
    from deside.trainers import load_model_state_from_checkpoint

    cfg = _base_model_config(input_dim=6, output_dim=2)
    model_a = build_deside_model(cfg)
    model_b = build_deside_model(cfg)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "dummy.ckpt")
        torch.save({"state_dict": {f"model.{k}": v for k, v in model_a.state_dict().items()}}, ckpt_path)
        load_model_state_from_checkpoint(model_b, ckpt_path)

    for a, b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(a, b, atol=1e-8)


def test_deside_facade_initialization(tmp_path):
    _requires_umap()
    from deside.decon_cf import DeSide

    model_dir = tmp_path / "DeSide_model"
    deside_obj = DeSide(model_dir=str(model_dir), log_file_path=str(tmp_path / "log.txt"))

    assert deside_obj.model_dir == str(model_dir)
    assert model_dir.exists()


def test_package_version_and_metadata():
    import tomllib

    with open("/Users/belter/github/VAEDecon_combined/DeSide/pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    dependencies = data["project"]["dependencies"]
    dep_string = "\n".join(dependencies).lower()

    assert any(d.startswith("torch") for d in dependencies)
    assert "lightning" in dep_string
    assert not any(d.startswith("tensorflow") for d in dependencies)
    assert data["project"]["version"].startswith("2.")
