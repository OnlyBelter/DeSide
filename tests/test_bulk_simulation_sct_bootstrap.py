from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deside.configs import BulkSimulationConfig
from deside.simulation.generate_data import BulkGEPGenerator, filtering_by_gene_list_and_pca_plot

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


def test_bulk_generator_resolves_subtype_column_alias():
    generator = BulkGEPGenerator.__new__(BulkGEPGenerator)
    generator.cell_subtype_used = ["Non-plasma B cells", "Plasma B cells"]
    generator.subtype_col_name = "cell_subtype"
    generator.cell_type_col_name = "cell_type"

    obs_df = pd.DataFrame(
        {
            "cell_type": ["B Cells", "B Cells"],
            "subtype": ["Non-plasma B cells", "Plasma B cells"],
        }
    )

    generator._resolve_subtype_column_name(obs_df)

    assert generator.subtype_col_name == "subtype"


def test_sc_sampling_uses_cell_type_column_for_sct_dataset(monkeypatch):
    generator = BulkGEPGenerator.__new__(BulkGEPGenerator)
    generator.cell_subtype_used = ["Non-plasma B cells", "Plasma B cells"]
    generator.subtype_col_name = "cell_subtype"
    generator.cell_type_col_name = "cell_type"
    generator.total_cell_number = 1

    cell_frac = pd.DataFrame(
        [{"Non-plasma B cells": 1.0, "Plasma B cells": 0.0}],
        index=["sample_1"],
    )
    obs_df = pd.DataFrame(
        {"cell_type": ["Non-plasma B cells", "Plasma B cells"]},
        index=["sct_1", "sct_2"],
    )
    captured = {}

    def fake_get_cell_num(cell_type_frac, total_num):
        return pd.DataFrame(
            {"Non-plasma B cells": [1], "Plasma B cells": [0]},
            index=cell_type_frac.index,
        )

    class FakePool:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starmap(self, func, paras):
            captured["paras"] = paras
            return [("sct_1",), tuple()]

    class FakeContext:
        def Pool(self, *_args, **_kwargs):
            return FakePool()

    monkeypatch.setattr("deside.simulation.generate_data.get_cell_num", fake_get_cell_num)
    monkeypatch.setattr("deside.simulation.generate_data.multiprocessing.cpu_count", lambda: 4)
    monkeypatch.setattr("deside.simulation.generate_data.multiprocessing.get_context", lambda _mode: FakeContext())

    sampled = generator._sc_sampling(cell_frac=cell_frac, obs_df=obs_df, n_threads=1, sc_dataset="sct_dataset")

    assert captured["paras"][0][4] == "cell_type"
    assert sampled.loc[("sample_1", "Non-plasma B cells"), "selected_cell_id"] == "sct_1"


def test_bulk_generator_log_status_only_once_for_same_key():
    generator = BulkGEPGenerator.__new__(BulkGEPGenerator)
    messages = []

    class FakeProgressBar:
        def write(self, message):
            messages.append(message)

    generator._active_progress_bar = FakeProgressBar()

    generator._log_status("message", once_key="same")
    generator._log_status("message", once_key="same")

    assert messages == ["message"]


def test_filtering_by_gene_list_reruns_pca_when_model_file_is_missing(tmp_path, monkeypatch):
    bulk_exp = pd.DataFrame([[1.0, 2.0], [1.5, 2.5]], index=["simu_1", "simu_2"], columns=["g1", "g2"])
    tcga_exp = pd.DataFrame([[2.0, 3.0], [2.5, 3.5]], index=["tcga_1", "tcga_2"], columns=["g1", "g2"])
    pca_data_file = tmp_path / "pca_data.csv"
    pca_data_file.write_text("PC1,PC2\n", encoding="utf-8")
    pca_model_file = tmp_path / "missing_model.joblib"
    calls = {"count": 0, "plot": 0}

    class FakePCA:
        explained_variance_ratio_ = np.array([0.7, 0.3])

        def transform(self, exp_df):
            return np.arange(exp_df.shape[0] * 2, dtype=float).reshape(exp_df.shape[0], 2)

    def fake_do_pca_analysis(exp_df, n_components, pca_result_fp):
        calls["count"] += 1
        assert n_components == 2
        assert pca_result_fp == str(pca_model_file)
        return FakePCA()

    fake_plot_module = types.ModuleType("deside.plot")

    def fake_plot_pca(**_kwargs):
        calls["plot"] += 1

    fake_plot_module.plot_pca = fake_plot_pca

    monkeypatch.setattr("deside.simulation.generate_data.do_pca_analysis", fake_do_pca_analysis)
    monkeypatch.setitem(sys.modules, "deside.plot", fake_plot_module)

    filtering_by_gene_list_and_pca_plot(
        bulk_exp=bulk_exp,
        tcga_exp=tcga_exp,
        gene_list=["g1", "g2"],
        result_dir=str(tmp_path),
        simu_dataset_name="Mixed_demo",
        n_components=2,
        pca_model_name_postfix="demo",
        bulk_exp_type="TPM",
        tcga_exp_type="TPM",
        pca_model_file_path=str(pca_model_file),
        pca_data_file_path=str(pca_data_file),
        if_plot_pca=True,
    )

    assert calls["count"] == 1
    assert calls["plot"] == 1


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
