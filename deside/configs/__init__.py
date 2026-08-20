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
    "load_deside_yaml",
    "deside_config_from_dict",
    "deside_config_from_yaml",
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


# Aliases for users coming from VAEDecon style naming.
load_deside_yaml = DeSideConfig.from_yaml
deside_config_from_dict = DeSideConfig.from_dict
deside_config_from_yaml = DeSideConfig.from_yaml
