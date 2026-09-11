"""Convenience entry points for DeSide YAML-driven workflows."""

from __future__ import annotations

import shutil
from typing import Dict, Optional, Union
from pathlib import Path

import pandas as pd

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

try:
    from ..plot import compare_y_y_pred_plot
except Exception:  # pragma: no cover - import robustness.
    from deside.plot import compare_y_y_pred_plot

try:
    from ..utility.read_file import ReadH5AD
except Exception:  # pragma: no cover - import robustness.
    from deside.utility.read_file import ReadH5AD


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
    for dst_name in ("example_model_training_config.yaml", src.name, f"config_used.yaml"):
        try:
            shutil.copy2(src, dst_dir / dst_name)
        except Exception:  # pragma: no cover - best-effort only.
            continue


def _aggregate_true_cell_fraction(
        true_cell_fraction: pd.DataFrame,
        group_cell_types: Optional[Dict[str, list[str]]],
) -> pd.DataFrame:
    if true_cell_fraction is None or true_cell_fraction.empty:
        return true_cell_fraction
    if not group_cell_types:
        return true_cell_fraction.copy()
    grouped = pd.DataFrame(index=true_cell_fraction.index)
    for group_name, subtypes in group_cell_types.items():
        available = [subtype for subtype in subtypes if subtype in true_cell_fraction.columns]
        if available:
            grouped[group_name] = true_cell_fraction.loc[:, available].sum(axis=1)
        elif group_name in true_cell_fraction.columns:
            grouped[group_name] = true_cell_fraction.loc[:, group_name]
    for col in true_cell_fraction.columns:
        if col not in grouped.columns and col not in group_cell_types:
            grouped[col] = true_cell_fraction.loc[:, col]
    return grouped


def _load_truth_cell_fraction(
        config: DeSideConfig,
        test_name: str,
        test_path: str,
        log_file_path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    truth_files = config.test_truth_files or {}
    truth_path = truth_files.get(test_name)
    if truth_path:
        truth_file = Path(truth_path)
        if truth_file.suffix.lower() == ".h5ad":
            return ReadH5AD(str(truth_file)).get_cell_fraction(round_decimals=None, copy=True)
        return pd.read_csv(truth_file, index_col=config.test_truth_index_col)

    test_file = Path(test_path)
    if test_file.suffix.lower() == ".h5ad":
        try:
            truth_df = ReadH5AD(str(test_file)).get_cell_fraction(round_decimals=None, copy=True)
            if truth_df is not None and not truth_df.empty:
                return truth_df
        except Exception as exc:  # pragma: no cover - optional best effort.
            print_msg(
                f"[WARN] Failed to read ground truth from test set '{test_name}' ({test_path}): "
                f"{type(exc).__name__}: {exc}",
                log_file_path=log_file_path,
            )
    return None


def _save_test_set_comparison(
        *,
        config: DeSideConfig,
        test_name: str,
        pred_df: pd.DataFrame,
        true_df: pd.DataFrame,
        test_output_dir: Path,
        log_file_path: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    true_df = _aggregate_true_cell_fraction(true_df, config.group_cell_types)
    pred_df = pred_df.copy()
    if "1-others" in pred_df.columns and "Cancer Cells" in pred_df.columns:
        pred_df = pred_df.drop(columns=["1-others"])

    compare_columns = [col for col in pred_df.columns if col in true_df.columns]
    shared_ids = [sample_id for sample_id in true_df.index if sample_id in pred_df.index]
    if not compare_columns or not shared_ids:
        print_msg(
            f"[WARN] Skip truth comparison for '{test_name}' because shared sample IDs or cell types were not found.",
            log_file_path=log_file_path,
        )
        return None

    aligned_true = true_df.loc[shared_ids, compare_columns].copy()
    aligned_pred = pred_df.loc[shared_ids, compare_columns].copy()

    merged = pd.concat(
        [
            aligned_true.add_prefix("true_"),
            aligned_pred.add_prefix("pred_"),
        ],
        axis=1,
    )
    merged.to_csv(test_output_dir / "cell_prop_comparison.csv")

    metrics = compare_y_y_pred_plot(
        y_true=aligned_true,
        y_pred=aligned_pred,
        show_columns=compare_columns,
        result_file_dir=str(test_output_dir),
        x_label="True cell proportion",
        y_label="Predicted cell proportion",
        model_name="DeSide",
        show_metrics=True,
        return_metrics=True,
        figsize=(8, 8),
        figure_format=config.comparison_figure_format,
    )
    if metrics is None:
        return None
    pd.DataFrame([metrics]).to_csv(
        test_output_dir / "prediction_metrics.csv",
        index=False,
        float_format="%.6f",
    )
    print_msg(
        f"Saved truth comparison for '{test_name}' with corr={metrics['corr']:.3f}, "
        f"rmse={metrics['rmse']:.3f}, ccc={metrics['ccc']:.3f}.",
        log_file_path=log_file_path,
    )
    return metrics


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
        comparison_summary = []
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
            test_result_dir = out_dir / test_name
            check_dir(str(test_result_dir))
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
            pred_df = pd.read_csv(output_file, index_col=0)
            pred_df.to_csv(test_result_dir / "predicted_fractions.csv")
            if config.compare_with_truth:
                true_df = _load_truth_cell_fraction(
                    config=config,
                    test_name=test_name,
                    test_path=test_path,
                    log_file_path=deside.log_file_path,
                )
                if true_df is None or true_df.empty:
                    print_msg(
                        f"[WARN] No ground-truth cell fractions found for test set '{test_name}'. "
                        "Saved predictions only.",
                        log_file_path=deside.log_file_path,
                    )
                else:
                    metrics = _save_test_set_comparison(
                        config=config,
                        test_name=test_name,
                        pred_df=pred_df,
                        true_df=true_df,
                        test_output_dir=test_result_dir,
                        log_file_path=deside.log_file_path,
                    )
                    if metrics is not None:
                        comparison_summary.append(
                            {
                                "test_set": test_name,
                                "n_samples": int(pred_df.index.isin(true_df.index).sum()),
                                **metrics,
                            }
                        )
        if comparison_summary:
            pd.DataFrame(comparison_summary).to_csv(
                out_dir / "prediction_metrics_summary.csv",
                index=False,
                float_format="%.6f",
            )

    return deside


def train_from_config_file(config_file_path: Union[str, Path]) -> DeSide:
    """One-line helper to train DeSide from a YAML configuration file.

    Usage
    -----
    >>> from deside.workflow import train_from_config_file
    >>> model = train_from_config_file('deside/configs/example_model_training_config.yaml')
    """

    config = DeSideConfig.from_yaml(config_file_path)
    return train_from_config(config, config_file_path=config_file_path)
