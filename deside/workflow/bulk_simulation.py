"""Independent config-driven bulk simulation and filtering workflow."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

try:
    from ..configs import BulkSimulationConfig
except Exception:  # pragma: no cover - import robustness for direct invocation.
    from deside.configs import BulkSimulationConfig

try:
    from ..simulation.generate_data import (
        BulkGEPGenerator,
        SingleCellTypeGEPGenerator,
        filtering_by_gene_list_and_pca_plot,
        get_gene_list_for_filtering,
    )
except Exception:  # pragma: no cover - import robustness.
    from deside.simulation.generate_data import (
        BulkGEPGenerator,
        SingleCellTypeGEPGenerator,
        filtering_by_gene_list_and_pca_plot,
        get_gene_list_for_filtering,
    )

try:
    from ..utility import check_dir, print_msg
except Exception:  # pragma: no cover - import robustness.
    from deside.utility import check_dir, print_msg

try:
    from ..utility.read_file import ReadExp, ReadH5AD
except Exception:  # pragma: no cover - import robustness.
    from deside.utility.read_file import ReadExp, ReadH5AD


__all__ = [
    "run_bulk_simulation_from_config",
    "run_bulk_simulation_from_config_file",
]


def _maybe_copy_source_config(config: BulkSimulationConfig, config_file_path: Optional[Union[str, Path]]) -> None:
    if config_file_path is None:
        return
    src = Path(config_file_path).resolve()
    if not src.exists():
        return
    dst_dir = Path(config.gene_filtering_result_dir)
    check_dir(str(dst_dir))
    for dst_name in ("example_bulk_simulation_config.yaml", src.name, "bulk_simulation_config_used.yaml"):
        try:
            shutil.copy2(src, dst_dir / dst_name)
        except Exception:  # pragma: no cover - best effort only.
            continue


def _read_gene_list_file(file_path: Union[str, Path]) -> List[str]:
    file_path = Path(file_path)
    gene_df = pd.read_csv(file_path)
    if "gene_name" in gene_df.columns:
        genes = gene_df["gene_name"].dropna().astype(str).tolist()
    elif gene_df.shape[1] >= 1:
        genes = gene_df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        genes = []
    if not genes:
        raise ValueError(f"No gene names were found in gene list file: {file_path}")
    return genes


def _write_gene_list(gene_list: List[str], output_file: Union[str, Path]) -> None:
    output_file = Path(output_file)
    check_dir(str(output_file.parent))
    pd.DataFrame({"gene_name": gene_list}).to_csv(output_file, index=False)


def _materialize_sct_output(
    source_file: Union[str, Path],
    requested_output_file: Optional[Union[str, Path]],
) -> str:
    source_path = Path(source_file)
    if requested_output_file is None:
        return str(source_path)
    target_path = Path(requested_output_file)
    if source_path.resolve() == target_path.resolve():
        return str(target_path)
    check_dir(str(target_path.parent))
    shutil.copy2(source_path, target_path)
    return str(target_path)


def _resolve_sct_dataset_reference(config: BulkSimulationConfig) -> Dict[str, Any]:
    resolved_input_path = config.sct_dataset_file_path
    if resolved_input_path:
        return {
            "resolved_sct_dataset_file": resolved_input_path,
            "resolved_sct_dataset_action": "reused",
            "resolved_sct_dataset_source": "provided",
        }

    sct_generator = SingleCellTypeGEPGenerator(**config.build_sct_generator_kwargs())
    default_output_path = Path(sct_generator.generated_bulk_gep_fp)
    requested_output_path = Path(config.sct_output_file_path) if config.sct_output_file_path else None

    if config.skip_if_done:
        if requested_output_path is not None and requested_output_path.exists():
            return {
                "resolved_sct_dataset_file": str(requested_output_path),
                "resolved_sct_dataset_action": "reused",
                "resolved_sct_dataset_source": "generated",
            }
        if default_output_path.exists():
            resolved_file = _materialize_sct_output(default_output_path, requested_output_path)
            return {
                "resolved_sct_dataset_file": resolved_file,
                "resolved_sct_dataset_action": "reused",
                "resolved_sct_dataset_source": "generated",
            }

    print_msg(
        f"Generating single-cell-type reference dataset '{config.sct_dataset_name}'",
        log_file_path=config.log_file_path,
    )
    sct_generator.generate_samples(**config.build_sct_generate_samples_kwargs())
    resolved_file = _materialize_sct_output(default_output_path, requested_output_path)
    return {
        "resolved_sct_dataset_file": resolved_file,
        "resolved_sct_dataset_action": "generated",
        "resolved_sct_dataset_source": "generated",
    }


def _build_summary(config: BulkSimulationConfig, outputs: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "bulk_dataset_name": config.bulk_dataset_name,
        "simu_bulk_dir": config.simu_bulk_dir,
        "resolved_sct_dataset_file": outputs["resolved_sct_dataset_file"],
        "resolved_sct_dataset_action": outputs["resolved_sct_dataset_action"],
        "resolved_sct_dataset_source": outputs["resolved_sct_dataset_source"],
        "generated_bulk_gep_file": outputs["generated_bulk_gep_file"],
        "generated_cell_fraction_file": outputs["generated_cell_fraction_file"],
        "gep_filtering_enabled": config.gep_filtering_enabled,
        "gene_filtering_enabled": config.gene_filtering_enabled,
        "gene_filtering_type": config.gene_filtering_type if config.gene_filtering_enabled else None,
    }
    if outputs.get("filtered_bulk_gep_file"):
        summary["filtered_bulk_gep_file"] = outputs["filtered_bulk_gep_file"]
    if outputs.get("gene_list_file"):
        summary["gene_list_file"] = outputs["gene_list_file"]
    generated_file = Path(outputs["generated_bulk_gep_file"])
    if generated_file.exists():
        generated_shape = ReadH5AD(str(generated_file)).get_h5ad().shape
        summary["generated_n_samples"] = int(generated_shape[0])
        summary["generated_n_genes"] = int(generated_shape[1])
    filtered_file = outputs.get("filtered_bulk_gep_file")
    if filtered_file and Path(filtered_file).exists():
        filtered_shape = ReadH5AD(str(filtered_file)).get_h5ad().shape
        summary["filtered_n_samples"] = int(filtered_shape[0])
        summary["filtered_n_genes"] = int(filtered_shape[1])
    gene_list_file = outputs.get("gene_list_file")
    if gene_list_file and Path(gene_list_file).exists():
        summary["filtered_gene_count"] = int(len(_read_gene_list_file(gene_list_file)))
    return summary


def _run_gene_filtering(config: BulkSimulationConfig, generated_bulk_gep_file: Union[str, Path]) -> Dict[str, Optional[str]]:
    check_dir(config.gene_filtering_result_dir)
    gene_list_path = Path(config.gene_filtering_gene_list_file)
    corr_result_path = Path(config.gene_filtering_corr_result_file)
    tcga_file = config.gene_filtering_tcga_file
    assert tcga_file is not None  # validated by config.validate()

    high_corr_file = config.gene_filtering_high_corr_gene_file
    gene_list: List[str]
    if high_corr_file and "high_corr_gene" in config.gene_filtering_type:
        high_corr_gene_list = _read_gene_list_file(high_corr_file)
        if config.gene_filtering_type == "high_corr_gene":
            gene_list = high_corr_gene_list
            _write_gene_list(gene_list, gene_list_path)
        elif config.gene_filtering_type == "high_corr_gene_and_quantile_range":
            quantile_gene_list = get_gene_list_for_filtering(
                bulk_exp_file=str(generated_bulk_gep_file),
                tcga_file=tcga_file,
                result_file_path=str(gene_list_path.with_name(f"{gene_list_path.stem}_quantile_range.csv")),
                q_col_name=config.gene_filtering_q_col_name,
                filtering_type="quantile_range",
                corr_threshold=config.gene_filtering_corr_threshold,
                n_gene_max=config.gene_filtering_n_gene_max,
                corr_result_fp=str(corr_result_path),
                quantile_range=config.gene_filtering_quantile_range,
            )
            quantile_gene_set = set(quantile_gene_list)
            gene_list = [gene for gene in high_corr_gene_list if gene in quantile_gene_set]
            _write_gene_list(gene_list, gene_list_path)
        else:
            gene_list = get_gene_list_for_filtering(
                bulk_exp_file=str(generated_bulk_gep_file),
                tcga_file=tcga_file,
                result_file_path=str(gene_list_path),
                q_col_name=config.gene_filtering_q_col_name,
                filtering_type=config.gene_filtering_type,
                corr_threshold=config.gene_filtering_corr_threshold,
                n_gene_max=config.gene_filtering_n_gene_max,
                corr_result_fp=str(corr_result_path),
                quantile_range=config.gene_filtering_quantile_range,
            )
    else:
        gene_list = get_gene_list_for_filtering(
            bulk_exp_file=str(generated_bulk_gep_file),
            tcga_file=tcga_file,
            result_file_path=str(gene_list_path),
            q_col_name=config.gene_filtering_q_col_name,
            filtering_type=config.gene_filtering_type,
            corr_threshold=config.gene_filtering_corr_threshold,
            n_gene_max=config.gene_filtering_n_gene_max,
            corr_result_fp=str(corr_result_path),
            quantile_range=config.gene_filtering_quantile_range,
        )

    filtered_h5ad_file = config.gene_filtering_filtered_h5ad_file if config.gene_filtering_save_filtered_h5ad else None
    if filtered_h5ad_file and config.skip_if_done and Path(filtered_h5ad_file).exists():
        return {
            "filtered_bulk_gep_file": filtered_h5ad_file,
            "gene_list_file": str(gene_list_path),
        }

    bulk_exp_obj = ReadH5AD(str(generated_bulk_gep_file))
    bulk_exp = bulk_exp_obj.get_df(round_decimals=None, copy=False)
    bulk_exp_cell_frac = bulk_exp_obj.get_cell_fraction(copy=True)
    tcga_exp = ReadExp(tcga_file, exp_type="TPM").get_exp(round_decimals=None, copy=False)
    filtering_by_gene_list_and_pca_plot(
        bulk_exp=bulk_exp,
        tcga_exp=tcga_exp,
        gene_list=gene_list,
        result_dir=config.gene_filtering_result_dir,
        simu_dataset_name=config.bulk_dataset_name,
        n_components=config.gene_filtering_pca_n_components,
        pca_model_name_postfix=config.gene_filtering_type,
        bulk_exp_type="log_space",
        tcga_exp_type="TPM",
        pca_model_file_path=config.gene_filtering_pca_model_file,
        pca_data_file_path=config.gene_filtering_pca_data_file,
        h5ad_file_path=filtered_h5ad_file,
        cell_frac_file=bulk_exp_cell_frac,
        figsize=config.gene_filtering_pca_figsize,
        if_plot_pca=config.gene_filtering_plot_pca,
    )
    return {
        "filtered_bulk_gep_file": filtered_h5ad_file,
        "gene_list_file": str(gene_list_path),
    }


def run_bulk_simulation_from_config(
    config: BulkSimulationConfig,
    config_file_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run the independent bulk simulation + filtering workflow from a config object."""

    config.validate()
    check_dir(config.simu_bulk_dir)
    check_dir(config.gene_filtering_result_dir)
    _maybe_copy_source_config(config, config_file_path)

    print_msg(
        f"Running bulk simulation workflow for dataset '{config.bulk_dataset_name}' under {config.simu_bulk_dir}",
        log_file_path=config.log_file_path,
    )
    gep_high_corr_gene_list = None
    if config.gep_high_corr_gene_list_file:
        gep_high_corr_gene_list = _read_gene_list_file(config.gep_high_corr_gene_list_file)

    sct_resolution = _resolve_sct_dataset_reference(config)
    bulk_generator_kwargs = config.build_bulk_generator_kwargs()
    bulk_generator_kwargs["sct_dataset_file_path"] = sct_resolution["resolved_sct_dataset_file"]
    bulk_generator = BulkGEPGenerator(**bulk_generator_kwargs)
    generated_bulk_gep_file = bulk_generator.generated_bulk_gep_fp
    if config.skip_if_done and Path(generated_bulk_gep_file).exists():
        print_msg(
            f"Skip bulk simulation because output already exists: {generated_bulk_gep_file}",
            log_file_path=config.log_file_path,
        )
    else:
        bulk_generator.generate_gep(**config.build_generate_gep_kwargs(high_corr_gene_list=gep_high_corr_gene_list))

    outputs: Dict[str, Any] = {
        "resolved_sct_dataset_file": sct_resolution["resolved_sct_dataset_file"],
        "resolved_sct_dataset_action": sct_resolution["resolved_sct_dataset_action"],
        "resolved_sct_dataset_source": sct_resolution["resolved_sct_dataset_source"],
        "generated_bulk_gep_file": generated_bulk_gep_file,
        "generated_cell_fraction_file": bulk_generator.generated_cell_fraction_fp,
        "filtered_bulk_gep_file": None,
        "gene_list_file": None,
    }
    if config.gene_filtering_enabled:
        print_msg(
            f"Running gene-level filtering ({config.gene_filtering_type}) for dataset '{config.bulk_dataset_name}'",
            log_file_path=config.log_file_path,
        )
        outputs.update(_run_gene_filtering(config, generated_bulk_gep_file))

    summary = _build_summary(config, outputs)
    summary_path = Path(config.simulation_summary_file_path)
    check_dir(str(summary_path.parent))
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print_msg(f"Saved bulk simulation summary to {summary_path}", log_file_path=config.log_file_path)
    return summary


def run_bulk_simulation_from_config_file(config_file_path: Union[str, Path]) -> Dict[str, Any]:
    """One-line helper to run bulk simulation from a YAML configuration file."""

    config = BulkSimulationConfig.from_yaml(config_file_path)
    return run_bulk_simulation_from_config(config, config_file_path=config_file_path)
