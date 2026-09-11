from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from deside.configs import BulkSimulationConfig

bulk_simulation_workflow = importlib.import_module("deside.workflow.bulk_simulation")


def _base_config_dict(tmp_path: Path) -> dict:
    merged_sc_file = tmp_path / "merged_sc.h5ad"
    merged_sc_file.write_text("merged", encoding="utf-8")
    return {
        "input": {
            "merged_sc_dataset_file_path": str(merged_sc_file),
            "sct_dataset_file_path": "",
            "tcga2cancer_type_file_path": "",
            "cell_type2subtype": {"Cancer Cells": ["Cancer Cells"], "CD8 T": ["CD8 T"]},
            "sc_dataset_ids": ["demo_sc"],
            "total_rna_coefficient": {},
            "cell_type_col_name": "cell_type",
            "subtype_col_name": "cell_type",
        },
        "output": {
            "simu_bulk_dir": str(tmp_path / "simu"),
            "bulk_dataset_name": "Mixed_demo",
            "gene_filtering_result_dir": str(tmp_path / "results"),
            "summary_file_path": str(tmp_path / "results" / "summary.json"),
        },
        "simulation": {
            "n_samples": 8,
            "sampling_method": "segment",
            "n_threads": 1,
            "simu_method": "mul",
        },
        "sct_generation": {
            "sct_dataset_name": "mixed_sctGEP_nbase100",
            "n_sample_each_cell_type": 4,
            "n_base_for_positive_samples": 2,
            "sample_type": "positive",
            "sep_by_patient": False,
            "simu_method": "ave",
            "test_set": False,
            "minimum_n_base": 1,
            "ref_gene_list_file_path": "",
            "output_file_path": "",
        },
        "gep_filtering": {"enable": False},
        "gene_filtering": {"enable": False},
        "runtime": {"skip_if_done": True, "check_basic_info": False},
    }


def test_bulk_simulation_config_allows_empty_sct_dataset_path(tmp_path):
    config = BulkSimulationConfig.from_dict(_base_config_dict(tmp_path))

    assert config.sct_dataset_file_path is None
    assert config.auto_generate_sct_dataset is True
    assert config.sct_dataset_name == "mixed_sctGEP_nbase100"
    assert config.build_bulk_generator_kwargs()["sct_dataset_file_path"] is None


def test_bulk_simulation_config_rejects_missing_provided_sct_path(tmp_path):
    config_dict = _base_config_dict(tmp_path)
    config_dict["input"]["sct_dataset_file_path"] = str(tmp_path / "missing_sct.h5ad")

    config = BulkSimulationConfig.from_dict(config_dict)

    with pytest.raises(ValueError, match="input.sct_dataset_file_path points to a missing file"):
        config.validate()


def test_bulk_simulation_config_requires_existing_merged_sc_for_bootstrap(tmp_path):
    config_dict = _base_config_dict(tmp_path)
    config_dict["input"]["merged_sc_dataset_file_path"] = str(tmp_path / "missing_merged_sc.h5ad")

    config = BulkSimulationConfig.from_dict(config_dict)

    with pytest.raises(ValueError, match="input.merged_sc_dataset_file_path points to a missing file"):
        config.validate()


def test_bulk_simulation_config_accepts_legacy_gene_filtering_tcga_file(tmp_path):
    config_dict = _base_config_dict(tmp_path)
    reference_file = tmp_path / "reference_tpm.csv"
    reference_file.write_text("gene,sample\nG1,1.0\n", encoding="utf-8")
    config_dict["gene_filtering"] = {
        "enable": True,
        "filtering_type": "quantile_range",
        "tcga_file": str(reference_file),
        "quantile_range": [0.005, 0.5, 0.995],
        "q_col_name": ["q_0.5", "q_50.0", "q_99.5"],
    }

    config = BulkSimulationConfig.from_dict(config_dict)

    assert config.gene_filtering_reference_file == str(reference_file)
    assert config.gene_filtering_tcga_file == str(reference_file)
    config.validate()


def test_bulk_simulation_config_allows_gep_filtering_without_cancer_type_subset(tmp_path):
    config_dict = _base_config_dict(tmp_path)
    reference_file = tmp_path / "reference_tpm.csv"
    reference_file.write_text("gene,sample\nG1,1.0\n", encoding="utf-8")
    config_dict["gep_filtering"] = {
        "enable": True,
        "reference_file": str(reference_file),
        "ref_exp_type": "TPM",
        "filtering_method": "median_gep",
        "filtering_ref_types": [],
    }

    config = BulkSimulationConfig.from_dict(config_dict)

    assert config.gep_filtering_ref_types is None
    config.validate()
    assert config.build_generate_gep_kwargs(high_corr_gene_list=None)["filtering_ref_types"] is None


def test_bulk_simulation_config_requires_mapping_when_gep_filtering_ref_types_provided(tmp_path):
    config_dict = _base_config_dict(tmp_path)
    reference_file = tmp_path / "reference_tpm.csv"
    reference_file.write_text("gene,sample\nG1,1.0\n", encoding="utf-8")
    config_dict["gep_filtering"] = {
        "enable": True,
        "reference_file": str(reference_file),
        "ref_exp_type": "TPM",
        "filtering_method": "median_gep",
        "filtering_ref_types": ["LUAD"],
    }

    config = BulkSimulationConfig.from_dict(config_dict)

    with pytest.raises(
        ValueError,
        match="input.tcga2cancer_type_file_path must be set when gep_filtering.filtering_ref_types is provided.",
    ):
        config.validate()


def test_resolve_sct_dataset_reference_generates_then_reuses(tmp_path, monkeypatch):
    config = BulkSimulationConfig.from_dict(_base_config_dict(tmp_path))
    generated_sct_file = tmp_path / "simu" / "simu_bulk_exp_mixed_sctGEP_nbase100_log2cpm1p.h5ad"

    class FakeSCTGenerator:
        def __init__(self, **kwargs):
            self.generated_bulk_gep_fp = str(generated_sct_file)

        def generate_samples(self, **kwargs):
            generated_sct_file.parent.mkdir(parents=True, exist_ok=True)
            generated_sct_file.write_text("generated", encoding="utf-8")

    monkeypatch.setattr(bulk_simulation_workflow, "SingleCellTypeGEPGenerator", FakeSCTGenerator)

    first = bulk_simulation_workflow._resolve_sct_dataset_reference(config)
    second = bulk_simulation_workflow._resolve_sct_dataset_reference(config)

    assert first["resolved_sct_dataset_action"] == "generated"
    assert first["resolved_sct_dataset_file"] == str(generated_sct_file)
    assert second["resolved_sct_dataset_action"] == "reused"
    assert second["resolved_sct_dataset_file"] == str(generated_sct_file)


def test_run_bulk_simulation_uses_resolved_sct_dataset_path(tmp_path, monkeypatch):
    config = BulkSimulationConfig.from_dict(_base_config_dict(tmp_path))
    generated_sct_file = tmp_path / "simu" / "simu_bulk_exp_mixed_sctGEP_nbase100_log2cpm1p.h5ad"
    generated_bulk_file = tmp_path / "simu" / "simu_bulk_exp_Mixed_demo_log2cpm1p.h5ad"
    generated_frac_file = tmp_path / "simu" / "generated_frac_Mixed_demo.csv"
    captured: dict = {}

    class FakeSCTGenerator:
        def __init__(self, **kwargs):
            self.generated_bulk_gep_fp = str(generated_sct_file)

        def generate_samples(self, **kwargs):
            generated_sct_file.parent.mkdir(parents=True, exist_ok=True)
            generated_sct_file.write_text("generated", encoding="utf-8")

    class FakeBulkGenerator:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.generated_bulk_gep_fp = str(generated_bulk_file)
            self.generated_cell_fraction_fp = str(generated_frac_file)

        def generate_gep(self, **kwargs):
            generated_bulk_file.parent.mkdir(parents=True, exist_ok=True)
            generated_bulk_file.write_text("bulk", encoding="utf-8")
            generated_frac_file.write_text("frac", encoding="utf-8")

    monkeypatch.setattr(bulk_simulation_workflow, "SingleCellTypeGEPGenerator", FakeSCTGenerator)
    monkeypatch.setattr(bulk_simulation_workflow, "BulkGEPGenerator", FakeBulkGenerator)
    monkeypatch.setattr(
        bulk_simulation_workflow,
        "_build_summary",
        lambda current_config, outputs: dict(outputs, bulk_dataset_name=current_config.bulk_dataset_name),
    )

    summary = bulk_simulation_workflow.run_bulk_simulation_from_config(config)

    assert captured["sct_dataset_file_path"] == str(generated_sct_file)
    assert summary["resolved_sct_dataset_action"] == "generated"
    assert summary["generated_bulk_gep_file"] == str(generated_bulk_file)
    written_summary = json.loads(Path(config.simulation_summary_file_path).read_text(encoding="utf-8"))
    assert written_summary["resolved_sct_dataset_file"] == str(generated_sct_file)
