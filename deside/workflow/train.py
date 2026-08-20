"""Convenience entry points for DeSide YAML-driven workflows."""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional, Union
from pathlib import Path

try:
    from ..configs import DeSideConfig
except Exception:  # pragma: no cover - import robustness for direct invocation.
    from deside.configs import DeSideConfig

try:
    from ..decon_cf import DeSide
except Exception:  # pragma: no cover - import robustness.
    from deside.decon_cf import DeSide

try:
    from ..utility import check_dir, print_msg
except Exception:  # pragma: no cover - import robustness.
    from deside.utility import check_dir, print_msg


__all__ = [
    "train_from_config",
    "train_from_config_file",
]


def _maybe_copy_source_config(config: DeSideConfig, config_file_path: Optional[Union[str, Path]]) -> None:
    if config_file_path is None:
        return
    src = Path(config_file_path).resolve()
    if not src.exists():
        return
    dst_dir = Path(config.model_dir)
    check_dir(str(dst_dir))
    for dst_name in ("example_config.yaml", src.name, f"config_used.yaml"):
        try:
            shutil.copy2(src, dst_dir / dst_name)
        except Exception:  # pragma: no cover - best-effort only.
            continue


def train_from_config(config: DeSideConfig, config_file_path: Optional[Union[str, Path]] = None) -> DeSide:
    """Train a DeSide model from a :class:`DeSideConfig` object.

    The YAML configuration mirrors the VAEDecon style and is mapped to the
    ``DeSide.train_model(...)`` kwargs and ``hyper_params`` dict the DeSide
    facade expects. If ``evaluation.test_sets`` is defined in the config, the
    trained model is applied to each test set immediately after training and
    predictions are saved under ``evaluation.test_predict_output_dir``.
    """

    check_dir(config.output_dir)
    deside = config.instantiate_deside()
    train_kwargs = config.build_train_model_kwargs()
    print_msg(
        f"Training {config.model_name} under model_dir={config.model_dir} "
        f"using training sets {config.training_set_file_paths}",
        log_file_path=deside.log_file_path,
    )
    _maybe_copy_source_config(config, config_file_path)
    deside.train_model(**train_kwargs)

    test_sets = config.test_sets
    if test_sets:
        try:
            from ..utility.read_file import read_gene_set  # noqa: F401
        except Exception:  # pragma: no cover - import path tolerance.
            from deside.utility.read_file import read_gene_set

        pathway_mask = train_kwargs.get("pathway_mask")
        if pathway_mask is None and config.build_hyper_params().get("pathway_network"):
            pw_files = config.pathway_gene_set_files
            if pw_files:
                try:
                    from ..utility.read_file import read_gene_set as _read_gene_set
                except Exception:
                    from deside.utility.read_file import read_gene_set as _read_gene_set
                pathway_mask = _read_gene_set(pw_files)

        predict_kwargs = config.build_predict_kwargs(pathway_mask=pathway_mask)
        out_dir = Path(config.test_predict_output_dir)
        check_dir(str(out_dir))
        for test_name, test_path in test_sets.items():
            output_file = str(out_dir / f"{test_name}_predicted_fractions.csv")
            print_msg(
                f"Predicting test set '{test_name}' ({test_path}) -> {output_file}",
                log_file_path=deside.log_file_path,
            )
            try:
                deside.predict(
                    input_file=test_path,
                    output_file_path=output_file,
                    **{k: v for k, v in predict_kwargs.items() if k != "input_file" and k != "output_file_path"},
                )
            except Exception as exc:  # pragma: no cover - best effort, never fail the whole pipeline on test-set predict error.
                print_msg(
                    f"[WARN] Failed to predict test set '{test_name}': {type(exc).__name__}: {exc}. "
                    "Training output is still valid.",
                    log_file_path=deside.log_file_path,
                )
                continue

    return deside


def train_from_config_file(config_file_path: Union[str, Path]) -> DeSide:
    """One-line helper to train DeSide from a YAML configuration file.

    Usage
    -----
    >>> from deside.workflow import train_from_config_file
    >>> model = train_from_config_file('deside/configs/example_config.yaml')
    """

    config = DeSideConfig.from_yaml(config_file_path)
    return train_from_config(config, config_file_path=config_file_path)
