"""Configuration utilities for DeSide in the PyTorch/Lightning 2.x line.

The layout mirrors the VAEDecon ``configs`` package: plain YAML files plus a
lightweight loader that converts them into the ``hyper_params`` dict and
training kwargs that the :class:`deside.decon_cf.DeSide` facade expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

__all__ = [
    "DeSideConfig",
    "BulkSimulationConfig",
    "load_deside_yaml",
    "load_bulk_simulation_yaml",
    "deside_config_from_dict",
    "deside_config_from_yaml",
    "bulk_simulation_config_from_dict",
    "bulk_simulation_config_from_yaml",
]


PathLike = Union[str, Path]


def _load_yaml(yaml_path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - PyYAML is a dep.
        raise ModuleNotFoundError(
            "DeSide configuration loading requires PyYAML. Please install PyYAML or use "
            "the migrated package dependencies (deside>=2.0.0a0)."
        ) from exc

    class _NoDuplicateSafeLoader(yaml.SafeLoader):
        pass

    def _construct_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate key '{key}' in YAML: {yaml_path}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _NoDuplicateSafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
    )
    with open(yaml_path, "r", encoding="utf-8") as handle:
        config_dict = yaml.load(handle, Loader=_NoDuplicateSafeLoader)
    if config_dict is None:
        config_dict = {}
    if not isinstance(config_dict, Mapping):
        raise ValueError(f"Top-level YAML content must be a mapping, got {type(config_dict).__name__}.")
    return dict(config_dict)


def _coerce_optional_list(value: Any, allow_empty: bool = True) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, (tuple, set)):
        return list(value)
    if allow_empty and isinstance(value, str) and value.strip() == "":
        return []
    return [value]


def _normalize_naming_postfix(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text and not text.startswith("_"):
        text = "_" + text
    return text


@dataclass
class DeSideConfig:
    """Configuration container used to train a DeSide model from a single YAML file."""

    raw: Dict[str, Any]
    data: Dict[str, Any]
    training: Dict[str, Any]
    model: Dict[str, Any]
    evaluation: Dict[str, Any]

    # ── Resolved paths ─────────────────────────────────────────────────
    @property
    def data_dir(self) -> str:
        return str(self.data.get("data_dir", "./datasets"))

    @property
    def training_set_file_paths(self) -> List[str]:
        training_sets = self.data.get("training_sets") or {}
        if isinstance(training_sets, Mapping):
            paths = [str(p) for p in training_sets.values()]
        elif isinstance(training_sets, (list, tuple)):
            paths = [str(p) for p in training_sets]
        else:
            raise ValueError(
                "data.training_sets must be a mapping (name->path) or a list of paths."
            )
        if not paths:
            raise ValueError("data.training_sets must define at least one training set file.")
        return paths

    @property
    def group_cell_types(self) -> Optional[Dict[str, List[str]]]:
        groups = self.data.get("group_cell_types") or None
        if not groups:
            return None
        if not isinstance(groups, Mapping):
            raise ValueError("data.group_cell_types must be a dict mapping group -> [subtypes].")
        normalized: Dict[str, List[str]] = {}
        for key, subtypes in groups.items():
            normalized[str(key)] = [str(s) for s in _coerce_optional_list(subtypes, allow_empty=False)]
        return normalized or None

    @property
    def cell_types(self) -> Optional[List[str]]:
        values = _coerce_optional_list(self.data.get("cell_types"))
        if not values:
            return None
        return [str(v) for v in values]

    @property
    def pathway_gene_set_files(self) -> Optional[List[str]]:
        values = _coerce_optional_list(self.data.get("pathway_gene_set_files"))
        if not values:
            return None
        return [str(v) for v in values]

    @property
    def filtered_gene_list(self) -> Optional[List[str]]:
        import pandas as pd

        file_path = self.data.get("filtered_gene_list_file")
        if not file_path:
            return None
        file_path = str(file_path)
        index_col = self.data.get("filtered_gene_list_index_col", 0)
        df = pd.read_csv(file_path, index_col=index_col)
        return [str(gene) for gene in df.index.to_list()]

    @property
    def input_gene_list_mode(self) -> Optional[str]:
        mode = self.data.get("input_gene_list")
        if mode is None:
            return None
        text = str(mode).strip()
        if text == "":
            return None
        return text

    @property
    def method_adding_pathway(self) -> str:
        return str(self.data.get("method_adding_pathway", "add_to_end"))

    @property
    def scaling_by_sample(self) -> bool:
        return bool(self.data.get("scaling_by_sample", False))

    @property
    def scaling_by_constant(self) -> bool:
        return bool(self.data.get("scaling_by_constant", True))

    @property
    def remove_cancer_cell(self) -> bool:
        return bool(self.data.get("remove_cancer_cell", False))

    # ── Evaluation (post-training predict) ─────────────────────────────

    @property
    def test_sets(self) -> Optional[Dict[str, str]]:
        raw = self.evaluation.get("test_sets") or self.evaluation.get("test_set_files") or None
        if raw is None:
            return None
        if isinstance(raw, Mapping):
            out: Dict[str, str] = {}
            for k, v in raw.items():
                if v is None or str(v).strip() == "":
                    continue
                out[str(k)] = str(v)
            return out or None
        if isinstance(raw, (list, tuple)):
            out = {}
            for i, p in enumerate(raw):
                if p is None or str(p).strip() == "":
                    continue
                name = Path(str(p)).stem or f"test_{i + 1}"
                out[name] = str(p)
            return out or None
        if isinstance(raw, str) and raw.strip():
            p = raw.strip()
            return {Path(p).stem or "test_1": p}
        return None

    @property
    def test_predict_output_dir(self) -> str:
        raw = self.evaluation.get("test_predict_output_dir")
        if raw:
            return str(raw)
        return str(Path(self.model_dir) / "predictions")

    @property
    def compare_with_truth(self) -> bool:
        return bool(self.evaluation.get("compare_with_truth", True))

    @property
    def test_truth_files(self) -> Optional[Dict[str, str]]:
        raw = (
            self.evaluation.get("test_truth_files")
            or self.evaluation.get("test_set_truth_files")
            or None
        )
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("evaluation.test_truth_files must be a dict mapping test-set name -> truth file path.")
        out: Dict[str, str] = {}
        for k, v in raw.items():
            if v is None or str(v).strip() == "":
                continue
            out[str(k)] = str(v)
        return out or None

    @property
    def test_truth_index_col(self) -> int:
        return int(self.evaluation.get("test_truth_index_col", 0))

    @property
    def comparison_figure_format(self) -> str:
        return str(self.evaluation.get("comparison_figure_format", "png"))

    @property
    def test_exp_type(self) -> str:
        val = self.evaluation.get("exp_type") or self.evaluation.get("test_exp_type") or "log_space"
        return str(val)

    @property
    def test_transpose(self) -> bool:
        return bool(self.evaluation.get("transpose", False))

    @property
    def test_add_cell_type(self) -> bool:
        return bool(self.evaluation.get("add_cell_type", False))

    @property
    def cell_prop_threshold(self) -> float:
        return float(self.evaluation.get("cell_prop_threshold", 0.0001))

    @property
    def cancer_cell_type_name(self) -> str:
        return str(self.evaluation.get("cancer_cell_type_name", "Cancer Cells"))

    @property
    def fill_cancer_as_residual(self) -> bool:
        return bool(self.evaluation.get("fill_cancer_as_residual", True))

    @property
    def renormalize_non_cancer_after_threshold(self) -> bool:
        return bool(self.evaluation.get("renormalize_non_cancer_after_threshold", True))

    def build_predict_kwargs(self, pathway_mask: Optional[Any] = None) -> Dict[str, Any]:
        """Bundle kwargs suitable for ``DeSide.predict()`` for a single test set.

        Callers override ``input_file`` / ``output_file_path`` per test set.
        """
        hp = self.build_hyper_params()
        return {
            "exp_type": self.test_exp_type,
            "transpose": self.test_transpose,
            "print_info": True,
            "add_cell_type": self.test_add_cell_type,
            "scaling_by_constant": self.scaling_by_constant,
            "scaling_by_sample": self.scaling_by_sample,
            "one_minus_alpha": bool(self.training.get("one_minus_alpha", False)),
            "pathway_mask": pathway_mask,
            "method_adding_pathway": self.method_adding_pathway,
            "hyper_params": hp,
            "cell_prop_threshold": self.cell_prop_threshold,
            "cancer_cell_type_name": self.cancer_cell_type_name,
            "fill_cancer_as_residual": self.fill_cancer_as_residual,
            "renormalize_non_cancer_after_threshold": self.renormalize_non_cancer_after_threshold,
        }

    # ── Training outputs and hyper-params ──────────────────────────────
    @property
    def output_dir(self) -> str:
        return str(self.training.get("output_dir", "./output/deside"))

    @property
    def model_name(self) -> str:
        return str(self.training.get("model_name", "DeSide"))

    @property
    def naming_postfix(self) -> str:
        return _normalize_naming_postfix(self.training.get("naming_postfix", ""))

    @property
    def model_dir(self) -> str:
        import os

        dir_name = self.model_name + self.naming_postfix
        return os.path.join(self.output_dir, dir_name)

    @property
    def log_file_path(self) -> Optional[str]:
        value = self.training.get("log_file_path")
        if value:
            return str(value)
        return None

    @property
    def n_epoch(self) -> int:
        return int(self.training.get("num_epochs", self.training.get("n_epoch", 10000)))

    @property
    def n_patience(self) -> int:
        return int(
            self.training.get("n_early_stopping_patience", self.training.get("n_patience", 100))
        )

    def build_hyper_params(self) -> Dict[str, Any]:
        model = self.model
        training = self.training
        architecture = (
            [int(u) for u in _coerce_optional_list(model.get("hidden_units"), allow_empty=False)],
            [float(d) for d in _coerce_optional_list(model.get("dropout_rates"), allow_empty=False)],
        )
        hidden_len = len(architecture[0])
        normalization = model.get("normalization")
        if normalization is not None:
            normalization = str(normalization)
            if normalization.lower() in {"none", "null"}:
                normalization = None
        normalization_layer = _coerce_optional_list(model.get("normalization_layer"))
        if not normalization_layer:
            normalization_layer = [1] * (hidden_len + 1)
        else:
            normalization_layer = [1 if int(v) else 0 for v in normalization_layer]
        if len(normalization_layer) != hidden_len + 1:
            raise ValueError(
                "model.normalization_layer must have length len(hidden_units) + 1 "
                f"(expected {hidden_len + 1}, got {len(normalization_layer)})."
            )

        pathway_network = bool(model.get("pathway_network", False))
        if pathway_network:
            pathway_arch = (
                [int(u) for u in _coerce_optional_list(model.get("pathway_hidden_units"), allow_empty=False)],
                [float(d) for d in _coerce_optional_list(model.get("pathway_dropout_rates"), allow_empty=False)],
            )
        else:
            pathway_arch = ([], [])

        hyper_params: Dict[str, Any] = {
            "architecture": architecture,
            "architecture_for_pathway_network": pathway_arch,
            "loss_function_alpha": float(training.get("loss_function_alpha", 0.5)),
            "normalization": normalization,
            "normalization_layer": normalization_layer,
            "pathway_network": pathway_network,
            "last_layer_activation": str(model.get("last_layer_activation", "sigmoid")),
            "learning_rate": float(training.get("learning_rate", 1e-4)),
            "optimizer": str(training.get("optimizer", "adamw")),
            "weight_decay": float(training.get("weight_decay", 1e-5)),
            "lr_scheduler": str(training.get("lr_scheduler", "reduce_lr_on_plateau")),
            "lr_scheduler_factor": float(training.get("lr_scheduler_factor", 0.5)),
            "lr_scheduler_patience": int(training.get("lr_scheduler_patience", 20)),
            "lr_scheduler_min_lr": float(training.get("lr_scheduler_min_lr", 1e-6)),
            "batch_size": int(training.get("batch_size", 128)),
            "validation_split": float(training.get("validation_split", 0.2)),
            "validation_seed": int(training.get("validation_seed", 42)),
        }
        return hyper_params

    # ── train_model callable bundle ────────────────────────────────────
    def build_train_model_kwargs(self) -> Dict[str, Any]:
        import os

        try:
            from ..utility.read_file import read_gene_set
        except Exception:  # pragma: no cover - import only needed during runtime.
            from deside.utility.read_file import read_gene_set

        pathway_mask = None
        pathway_files = self.pathway_gene_set_files
        if self.build_hyper_params()["pathway_network"]:
            if not pathway_files:
                raise ValueError(
                    "model.pathway_network=true requires data.pathway_gene_set_files to be set."
                )
            pathway_mask = read_gene_set(pathway_files)

        kwargs: Dict[str, Any] = {
            "training_set_file_path": self.training_set_file_paths,
            "hyper_params": self.build_hyper_params(),
            "cell_types": self.cell_types,
            "scaling_by_sample": self.scaling_by_sample,
            "callback": bool(self.training.get("callback", True)),
            "n_epoch": self.n_epoch,
            "n_patience": self.n_patience,
            "scaling_by_constant": self.scaling_by_constant,
            "remove_cancer_cell": self.remove_cancer_cell,
            "fine_tune": bool(self.training.get("fine_tune", False)),
            "one_minus_alpha": bool(self.training.get("one_minus_alpha", False)),
            "verbose": int(self.training.get("verbose", 1)),
            "pathway_mask": pathway_mask,
            "method_adding_pathway": self.method_adding_pathway,
            "input_gene_list": self.input_gene_list_mode,
            "filtered_gene_list": self.filtered_gene_list,
            "group_cell_types": self.group_cell_types,
        }
        return kwargs

    def instantiate_deside(self):
        """Return a ``DeSide`` facade pre-configured for this YAML file."""
        try:
            from ..decon_cf import DeSide
        except Exception:  # pragma: no cover - import path tolerance.
            from deside.decon_cf import DeSide

        return DeSide(model_dir=self.model_dir, log_file_path=self.log_file_path, model_name=self.model_name)

    # ── Factory helpers ────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> "DeSideConfig":
        raw = dict(config_dict)
        return cls(
            raw=raw,
            data=dict(raw.get("data", {}) or {}),
            training=dict(raw.get("training", {}) or {}),
            model=dict(raw.get("model", {}) or {}),
            evaluation=dict(raw.get("evaluation", {}) or {}),
        )

    @classmethod
    def from_yaml(cls, yaml_path: PathLike) -> "DeSideConfig":
        yaml_path = Path(yaml_path)
        if not yaml_path.is_absolute():
            yaml_path = yaml_path.resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"DeSide YAML config not found: {yaml_path}")
        return cls.from_dict(_load_yaml(yaml_path))


@dataclass
class BulkSimulationConfig:
    """Configuration container for independent bulk simulation and filtering workflows."""

    raw: Dict[str, Any]
    input: Dict[str, Any]
    output: Dict[str, Any]
    simulation: Dict[str, Any]
    sct_generation: Dict[str, Any]
    gep_filtering: Dict[str, Any]
    gene_filtering: Dict[str, Any]
    runtime: Dict[str, Any]

    @property
    def simu_bulk_dir(self) -> str:
        return str(self.output.get("simu_bulk_dir", "./datasets/simulated_bulk_cell_dataset"))

    @property
    def bulk_dataset_name(self) -> str:
        value = self.output.get("bulk_dataset_name")
        if not value or str(value).strip() == "":
            raise ValueError("output.bulk_dataset_name must be set in the bulk simulation config.")
        return str(value)

    @property
    def log_file_path(self) -> Optional[str]:
        value = self.output.get("log_file_path")
        if not value:
            return None
        return str(value)

    @property
    def generated_bulk_gep_file_path(self) -> str:
        prefix = f"simu_bulk_exp_{self.bulk_dataset_name}_log2cpm1p"
        return str(Path(self.simu_bulk_dir) / f"{prefix}.h5ad")

    @property
    def gene_filtering_result_dir(self) -> str:
        value = self.output.get("gene_filtering_result_dir")
        if value:
            return str(value)
        return str(Path(self.simu_bulk_dir) / self.bulk_dataset_name)

    @property
    def generated_cell_fraction_file_path(self) -> str:
        return str(Path(self.simu_bulk_dir) / f"generated_frac_{self.bulk_dataset_name}.csv")

    @property
    def simulation_summary_file_path(self) -> str:
        value = self.output.get("summary_file_path")
        if value:
            return str(value)
        return str(Path(self.gene_filtering_result_dir) / "bulk_simulation_summary.json")

    @property
    def skip_if_done(self) -> bool:
        return bool(self.runtime.get("skip_if_done", True))

    @property
    def check_basic_info(self) -> bool:
        return bool(self.runtime.get("check_basic_info", True))

    @property
    def zero_ratio_threshold(self) -> float:
        return float(self.runtime.get("zero_ratio_threshold", 0.97))

    @property
    def sc_dataset_gep_type(self) -> str:
        return str(self.runtime.get("sc_dataset_gep_type", "log_space"))

    @property
    def merged_sc_dataset_file_path(self) -> Optional[str]:
        value = self.input.get("merged_sc_dataset_file_path")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def sct_dataset_file_path(self) -> Optional[str]:
        value = self.input.get("sct_dataset_file_path")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def auto_generate_sct_dataset(self) -> bool:
        return self.sct_dataset_file_path is None

    @property
    def tcga2cancer_type_file_path(self) -> Optional[str]:
        value = self.input.get("tcga2cancer_type_file_path")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def cell_type2subtype(self) -> Dict[str, List[str]]:
        raw = self.input.get("cell_type2subtype")
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("input.cell_type2subtype must be a non-empty dict.")
        return {
            str(cell_type): [str(subtype) for subtype in _coerce_optional_list(subtypes, allow_empty=False)]
            for cell_type, subtypes in raw.items()
        }

    @property
    def sc_dataset_ids(self) -> List[str]:
        values = _coerce_optional_list(self.input.get("sc_dataset_ids"), allow_empty=False)
        if not values:
            raise ValueError("input.sc_dataset_ids must define at least one single-cell dataset ID.")
        return [str(value) for value in values]

    @property
    def total_rna_coefficient(self) -> Optional[Dict[str, float]]:
        raw = self.input.get("total_rna_coefficient")
        if raw in (None, ""):
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("input.total_rna_coefficient must be a dict when provided.")
        return {str(key): float(value) for key, value in raw.items()}

    @property
    def subtype_col_name(self) -> Optional[str]:
        value = self.input.get("subtype_col_name")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def cell_type_col_name(self) -> Optional[str]:
        value = self.input.get("cell_type_col_name")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def n_samples(self) -> int:
        return int(self.simulation.get("n_samples", 0))

    @property
    def sct_dataset_name(self) -> str:
        return str(self.sct_generation.get("sct_dataset_name", "mixed_sctGEP_nbase100"))

    @property
    def sct_n_sample_each_cell_type(self) -> int:
        return int(self.sct_generation.get("n_sample_each_cell_type", 10000))

    @property
    def sct_n_base_for_positive_samples(self) -> int:
        return int(self.sct_generation.get("n_base_for_positive_samples", 100))

    @property
    def sct_sample_type(self) -> str:
        return str(self.sct_generation.get("sample_type", "positive"))

    @property
    def sct_sep_by_patient(self) -> bool:
        return bool(self.sct_generation.get("sep_by_patient", False))

    @property
    def sct_simu_method(self) -> str:
        return str(self.sct_generation.get("simu_method", "ave"))

    @property
    def sct_test_set(self) -> bool:
        return bool(self.sct_generation.get("test_set", False))

    @property
    def sct_minimum_n_base(self) -> int:
        return int(self.sct_generation.get("minimum_n_base", 1))

    @property
    def sct_ref_gene_list_file_path(self) -> Optional[str]:
        value = self.sct_generation.get("ref_gene_list_file_path")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def sct_output_file_path(self) -> Optional[str]:
        value = self.sct_generation.get("output_file_path")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def sampling_method(self) -> str:
        return str(self.simulation.get("sampling_method", "segment"))

    @property
    def sampling_range(self) -> Optional[Dict[str, List[float]]]:
        raw = self.simulation.get("sampling_range")
        if not raw:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("simulation.sampling_range must be a dict when provided.")
        return {
            str(key): [float(v) for v in _coerce_optional_list(value, allow_empty=False)]
            for key, value in raw.items()
        }

    @property
    def total_cell_number(self) -> int:
        return int(self.simulation.get("total_cell_number", 100))

    @property
    def n_threads(self) -> int:
        return int(self.simulation.get("n_threads", 10))

    @property
    def simu_method(self) -> str:
        return str(self.simulation.get("simu_method", "mul"))

    @property
    def add_noise(self) -> bool:
        return bool(self.simulation.get("add_noise", False))

    @property
    def noise_params(self) -> tuple:
        return tuple(_coerce_optional_list(self.simulation.get("noise_params")))

    @property
    def cell_prop_prior(self) -> Optional[Dict[str, List[float]]]:
        raw = self.simulation.get("cell_prop_prior")
        if not raw:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("simulation.cell_prop_prior must be a dict when provided.")
        return {
            str(key): [float(v) for v in _coerce_optional_list(value, allow_empty=False)]
            for key, value in raw.items()
        }

    @property
    def gep_filtering_enabled(self) -> bool:
        return bool(self.gep_filtering.get("enable", True))

    @property
    def gep_reference_file(self) -> Optional[str]:
        value = self.gep_filtering.get("reference_file")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def gep_ref_exp_type(self) -> Optional[str]:
        value = self.gep_filtering.get("ref_exp_type")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def gep_filtering_quantile(self) -> tuple[Optional[float], Optional[float]]:
        values = _coerce_optional_list(self.gep_filtering.get("gep_filtering_quantile"))
        if not values:
            return (None, 0.95)
        if len(values) != 2:
            raise ValueError("gep_filtering.gep_filtering_quantile must contain exactly two values.")
        lower = None if values[0] in (None, "", "null") else float(values[0])
        upper = None if values[1] in (None, "", "null") else float(values[1])
        return (lower, upper)

    @property
    def gep_filtering_method(self) -> str:
        return str(self.gep_filtering.get("filtering_method", "median_gep"))

    @property
    def gep_filtering_ref_types(self) -> Optional[List[str]]:
        values = _coerce_optional_list(self.gep_filtering.get("filtering_ref_types"))
        if not values:
            return None
        normalized_values = [str(value).strip() for value in values]
        if len(normalized_values) == 1 and normalized_values[0].lower() == "all":
            import pandas as pd

            tcga_map_file = self.tcga2cancer_type_file_path
            if not tcga_map_file:
                raise ValueError(
                    "input.tcga2cancer_type_file_path must be set when "
                    "gep_filtering.filtering_ref_types='all'."
                )
            tcga_df = pd.read_csv(tcga_map_file, index_col=0)
            if "cancer_type" not in tcga_df.columns:
                raise ValueError(
                    "The TCGA cancer-type mapping file must contain a 'cancer_type' column "
                    "when gep_filtering.filtering_ref_types='all'."
                )
            return sorted(tcga_df["cancer_type"].dropna().astype(str).unique().tolist())
        return normalized_values

    @property
    def gep_show_filtering_info(self) -> bool:
        return bool(self.gep_filtering.get("show_filtering_info", False))

    @property
    def gep_n_top(self) -> int:
        return int(self.gep_filtering.get("n_top", 20))

    @property
    def gep_high_corr_gene_list_file(self) -> Optional[str]:
        value = self.gep_filtering.get("high_corr_gene_list_file")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def gep_filtering_by_gene_range(self) -> bool:
        return bool(self.gep_filtering.get("filtering_by_gene_range", False))

    @property
    def gep_min_percentage_within_gene_range(self) -> float:
        return float(self.gep_filtering.get("min_percentage_within_gene_range", 0.95))

    @property
    def gep_gene_quantile_range(self) -> Optional[List[float]]:
        values = _coerce_optional_list(self.gep_filtering.get("gene_quantile_range"))
        if not values:
            return None
        return [float(value) for value in values]

    @property
    def gep_filtering_in_pca_space(self) -> bool:
        return bool(self.gep_filtering.get("filtering_in_pca_space", False))

    @property
    def gep_pca_n_components(self) -> Union[int, float]:
        value = self.gep_filtering.get("pca_n_components", 0.9)
        if isinstance(value, float):
            return value
        if isinstance(value, int):
            return value
        text = str(value).strip()
        return float(text) if "." in text else int(text)

    @property
    def gep_norm_ord(self) -> int:
        return int(self.gep_filtering.get("norm_ord", 1))

    @property
    def gene_filtering_enabled(self) -> bool:
        return bool(self.gene_filtering.get("enable", True))

    @property
    def gene_filtering_type(self) -> str:
        return str(self.gene_filtering.get("filtering_type", "high_corr_gene_and_quantile_range"))

    @property
    def gene_filtering_tcga_file(self) -> Optional[str]:
        value = self.gene_filtering.get("tcga_file")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def gene_filtering_quantile_range(self) -> Optional[List[float]]:
        values = _coerce_optional_list(self.gene_filtering.get("quantile_range"))
        if not values:
            return None
        return [float(value) for value in values]

    @property
    def gene_filtering_q_col_name(self) -> Optional[List[str]]:
        values = _coerce_optional_list(self.gene_filtering.get("q_col_name"))
        if not values:
            return None
        return [str(value) for value in values]

    @property
    def gene_filtering_corr_threshold(self) -> float:
        return float(self.gene_filtering.get("corr_threshold", 0.3))

    @property
    def gene_filtering_n_gene_max(self) -> int:
        return int(self.gene_filtering.get("n_gene_max", 1000))

    @property
    def gene_filtering_high_corr_gene_file(self) -> Optional[str]:
        value = self.gene_filtering.get("high_corr_gene_file")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def gene_filtering_save_filtered_h5ad(self) -> bool:
        return bool(self.gene_filtering.get("save_filtered_h5ad", True))

    @property
    def gene_filtering_filtered_dataset_postfix(self) -> str:
        return str(self.gene_filtering.get("filtered_dataset_postfix", "filtered"))

    @property
    def gene_filtering_plot_pca(self) -> bool:
        return bool(self.gene_filtering.get("plot_pca", True))

    @property
    def gene_filtering_pca_n_components(self) -> int:
        return int(self.gene_filtering.get("pca_n_components", 2))

    @property
    def gene_filtering_pca_figsize(self) -> tuple[float, float]:
        values = _coerce_optional_list(self.gene_filtering.get("pca_figsize", [5, 5]), allow_empty=False)
        if len(values) != 2:
            raise ValueError("gene_filtering.pca_figsize must contain two values.")
        return (float(values[0]), float(values[1]))

    @property
    def gene_filtering_gene_list_file(self) -> str:
        value = self.gene_filtering.get("gene_list_file")
        if value:
            return str(value)
        file_name = f"gene_list_filtered_by_{self.gene_filtering_type}.csv"
        return str(Path(self.gene_filtering_result_dir) / file_name)

    @property
    def gene_filtering_corr_result_file(self) -> str:
        value = self.gene_filtering.get("corr_result_file")
        if value:
            return str(value)
        return str(Path(self.gene_filtering_result_dir) / "gene_corr_with_cell_fraction.csv")

    @property
    def gene_filtering_pca_model_file(self) -> str:
        value = self.gene_filtering.get("pca_model_file")
        if value:
            return str(value)
        return str(Path(self.gene_filtering_result_dir) / f"both_TCGA_and_simu_data_{self.bulk_dataset_name}_PCA_{self.gene_filtering_type}.joblib")

    @property
    def gene_filtering_pca_data_file(self) -> str:
        value = self.gene_filtering.get("pca_data_file")
        if value:
            return str(value)
        return str(Path(self.gene_filtering_result_dir) / f"{self.bulk_dataset_name}_PCA_with_TCGA_{self.gene_filtering_type}.csv")

    @property
    def gene_filtering_filtered_h5ad_file(self) -> str:
        value = self.gene_filtering.get("filtered_h5ad_file")
        if value:
            return str(value)
        postfix = self.gene_filtering_filtered_dataset_postfix.strip("_")
        if not postfix:
            postfix = "filtered"
        stem = Path(self.generated_bulk_gep_file_path).stem
        return str(Path(self.simu_bulk_dir) / f"{stem}_{postfix}.h5ad")

    def build_bulk_generator_kwargs(self) -> Dict[str, Any]:
        return {
            "simu_bulk_dir": self.simu_bulk_dir,
            "merged_sc_dataset_file_path": self.merged_sc_dataset_file_path,
            "sct_dataset_file_path": self.sct_dataset_file_path,
            "cell_type2subtype": self.cell_type2subtype,
            "sc_dataset_ids": self.sc_dataset_ids,
            "bulk_dataset_name": self.bulk_dataset_name,
            "check_basic_info": self.check_basic_info,
            "zero_ratio_threshold": self.zero_ratio_threshold,
            "sc_dataset_gep_type": self.sc_dataset_gep_type,
            "tcga2cancer_type_file_path": self.tcga2cancer_type_file_path,
            "total_rna_coefficient": self.total_rna_coefficient,
            "subtype_col_name": self.subtype_col_name,
            "cell_type_col_name": self.cell_type_col_name,
        }

    def build_sct_generator_kwargs(self) -> Dict[str, Any]:
        return {
            "merged_sc_dataset_file_path": self.merged_sc_dataset_file_path,
            "cell_type2subtype": self.cell_type2subtype,
            "sc_dataset_ids": self.sc_dataset_ids,
            "simu_bulk_dir": self.simu_bulk_dir,
            "bulk_dataset_name": self.sct_dataset_name,
            "zero_ratio_threshold": self.zero_ratio_threshold,
            "sc_dataset_gep_type": self.sc_dataset_gep_type,
            "subtype_col_name": self.subtype_col_name,
            "cell_type_col_name": self.cell_type_col_name,
        }

    def build_sct_generate_samples_kwargs(self) -> Dict[str, Any]:
        return {
            "n_sample_each_cell_type": self.sct_n_sample_each_cell_type,
            "n_base_for_positive_samples": self.sct_n_base_for_positive_samples,
            "sample_type": self.sct_sample_type,
            "sep_by_patient": self.sct_sep_by_patient,
            "simu_method": self.sct_simu_method,
            "test_set": self.sct_test_set,
            "minimum_n_base": self.sct_minimum_n_base,
            "ref_gene_list_file_path": self.sct_ref_gene_list_file_path,
        }

    def build_generate_gep_kwargs(self, *, high_corr_gene_list: Optional[List[str]]) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "sampling_range": self.sampling_range,
            "sampling_method": self.sampling_method,
            "total_cell_number": self.total_cell_number,
            "n_threads": self.n_threads,
            "filtering": self.gep_filtering_enabled,
            "reference_file": self.gep_reference_file,
            "ref_exp_type": self.gep_ref_exp_type,
            "gep_filtering_quantile": self.gep_filtering_quantile,
            "log_file_path": self.log_file_path,
            "n_top": self.gep_n_top,
            "simu_method": self.simu_method,
            "filtering_method": self.gep_filtering_method,
            "add_noise": self.add_noise,
            "noise_params": self.noise_params,
            "filtering_ref_types": self.gep_filtering_ref_types,
            "show_filtering_info": self.gep_show_filtering_info,
            "cell_prop_prior": self.cell_prop_prior,
            "high_corr_gene_list": high_corr_gene_list,
            "filtering_by_gene_range": self.gep_filtering_by_gene_range,
            "min_percentage_within_gene_range": self.gep_min_percentage_within_gene_range,
            "gene_quantile_range": self.gep_gene_quantile_range,
            "filtering_in_pca_space": self.gep_filtering_in_pca_space,
            "pca_n_components": self.gep_pca_n_components,
            "norm_ord": self.gep_norm_ord,
        }

    def validate(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("simulation.n_samples must be > 0.")
        if self.auto_generate_sct_dataset:
            if not self.merged_sc_dataset_file_path:
                raise ValueError(
                    "input.merged_sc_dataset_file_path must be set when input.sct_dataset_file_path is empty."
                )
            if not Path(self.merged_sc_dataset_file_path).exists():
                raise ValueError(
                    "input.merged_sc_dataset_file_path points to a missing file: "
                    f"{self.merged_sc_dataset_file_path}"
                )
            if self.sct_n_sample_each_cell_type <= 0:
                raise ValueError("sct_generation.n_sample_each_cell_type must be > 0.")
            if self.sct_n_base_for_positive_samples <= 0:
                raise ValueError("sct_generation.n_base_for_positive_samples must be > 0.")
            if self.sct_minimum_n_base <= 0:
                raise ValueError("sct_generation.minimum_n_base must be > 0.")
            if self.sct_sample_type not in {"positive", "negative"}:
                raise ValueError("sct_generation.sample_type must be either 'positive' or 'negative'.")
        elif not Path(self.sct_dataset_file_path).exists():
            raise ValueError(
                "input.sct_dataset_file_path points to a missing file: "
                f"{self.sct_dataset_file_path}"
            )
        if self.gep_filtering_enabled:
            if not self.gep_reference_file:
                raise ValueError("gep_filtering.reference_file must be set when gep_filtering.enable=true.")
            if not self.gep_ref_exp_type:
                raise ValueError("gep_filtering.ref_exp_type must be set when gep_filtering.enable=true.")
            if not self.gep_filtering_ref_types:
                raise ValueError("gep_filtering.filtering_ref_types must be set when gep_filtering.enable=true.")
        if self.gene_filtering_enabled:
            if not self.gene_filtering_tcga_file:
                raise ValueError("gene_filtering.tcga_file must be set when gene_filtering.enable=true.")
            if "quantile_range" in self.gene_filtering_type and not self.gene_filtering_quantile_range:
                raise ValueError(
                    "gene_filtering.quantile_range must be set when the filtering type uses quantile_range."
                )

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> "BulkSimulationConfig":
        raw = dict(config_dict)
        return cls(
            raw=raw,
            input=dict(raw.get("input", {}) or {}),
            output=dict(raw.get("output", {}) or {}),
            simulation=dict(raw.get("simulation", {}) or {}),
            sct_generation=dict(raw.get("sct_generation", {}) or {}),
            gep_filtering=dict(raw.get("gep_filtering", {}) or {}),
            gene_filtering=dict(raw.get("gene_filtering", {}) or {}),
            runtime=dict(raw.get("runtime", {}) or {}),
        )

    @classmethod
    def from_yaml(cls, yaml_path: PathLike) -> "BulkSimulationConfig":
        yaml_path = Path(yaml_path)
        if not yaml_path.is_absolute():
            yaml_path = yaml_path.resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"DeSide bulk simulation YAML config not found: {yaml_path}")
        return cls.from_dict(_load_yaml(yaml_path))


# Aliases for users coming from VAEDecon style naming.
load_deside_yaml = DeSideConfig.from_yaml
deside_config_from_dict = DeSideConfig.from_dict
deside_config_from_yaml = DeSideConfig.from_yaml
load_bulk_simulation_yaml = BulkSimulationConfig.from_yaml
bulk_simulation_config_from_dict = BulkSimulationConfig.from_dict
bulk_simulation_config_from_yaml = BulkSimulationConfig.from_yaml
