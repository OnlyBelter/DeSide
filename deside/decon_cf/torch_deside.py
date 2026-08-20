from __future__ import annotations

import os as _os

def _cap_blas_threads():
    """Cap BLAS / numpy / torch worker threads before either library is fully
    imported, to prevent CPU oversubscription on HPC nodes. Oversubscription of
    OpenBLAS/MKL threads on a shared node (e.g. 32 threads requested but 64
    launched) multiplies context-switch overhead and, when combined with the
    2.8–5.6 GB peak RSS of the pathway matmul stage, pushes the process into
    swap thrash that presents as a 20+ minute silent stall with no stdout
    updates (the exact symptom reported in this investigation).

    Policy: if the user has not explicitly set any of the standard knobs
    (OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, NUMEXPR_NUM_THREADS,
    VECLIB_MAXIMUM_THREADS), clamp to min(physical cores, 8, torch intra-op parallelism).
    """
    knobs = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    explicitly_set = any(_os.environ.get(k) for k in knobs)
    if explicitly_set:
        return
    try:
        import multiprocessing as _mp
        phys = _mp.cpu_count() or 4
    except Exception:
        phys = 4
    cap = min(phys, 8)
    for k in knobs:
        _os.environ.setdefault(k, str(cap))
    try:
        import torch as _torch
        if hasattr(_torch, "set_num_threads") and not _os.environ.get("PYTORCH_NO_AUTOTHREAD_CAP"):
            try:
                _torch.set_num_threads(cap)
                _torch.set_num_interop_threads(max(1, min(cap // 2, 4)))
            except Exception:
                pass
    except Exception:
        pass

_cap_blas_threads()
del _cap_blas_threads

import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from ..data import DeSideDataset, split_deside_dataset
from ..models import build_deside_model
from ..trainers import load_model_state_from_checkpoint
from ..utility import check_dir, get_x_by_pathway_network, print_msg
from ..utility.read_file import ReadExp, ReadH5AD, read_gene_set


_DEFAULT_PRETRAINED_PATHWAY_FILES = (
    "c2.cp.kegg.v2023.1.Hs.symbols.gmt",
    "c2.cp.reactome.v2023.1.Hs.symbols.gmt",
)

_DEFAULT_PRETRAINED_MODEL_DOI = "10.6084/m9.figshare.25117862.v1"
_DEFAULT_GITHUB_RAW_BASE = "https://raw.githubusercontent.com/OnlyBelter/DeSide_mini_example/main"


def _plot_loss(history_df, output_dir=None, x_label="n_epoch", y_label="MSE", file_name=None):
    plt.figure(figsize=(8, 6))
    if "loss" in history_df.columns:
        plt.plot(history_df["epoch"], history_df["loss"], label="loss")
    if "val_loss" in history_df.columns:
        plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label.upper())
    plt.tight_layout()
    if output_dir:
        target_name = "loss.png" if file_name is None else file_name
        plt.savefig(os.path.join(output_dir, target_name), dpi=200)
        plt.close()
    else:
        return plt


def _get_default_pretrained_hyper_params() -> dict:
    return {
        "architecture": ([200, 2000, 2000, 2000, 50], [0.05, 0.05, 0.05, 0.2, 0]),
        "architecture_for_pathway_network": ([50, 500, 500, 500, 50], [0, 0, 0, 0, 0]),
        "loss_function_alpha": 0.5,
        "normalization": "layer_normalization",
        "normalization_layer": [0, 0, 1, 1, 1, 1],
        "pathway_network": True,
        "last_layer_activation": "sigmoid",
        "learning_rate": 1e-4,
        "optimizer": "adamw",
        "weight_decay": 1e-5,
        "lr_scheduler": "reduce_lr_on_plateau",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 20,
        "lr_scheduler_min_lr": 1e-6,
        "batch_size": 128,
        "validation_split": 0.2,
        "validation_seed": 42,
    }


def _get_default_pathway_file_paths(dataset_dir: str) -> list:
    gene_set_dir = os.path.join(dataset_dir, "gene_set")
    return [os.path.join(gene_set_dir, i) for i in _DEFAULT_PRETRAINED_PATHWAY_FILES]


def _md5_file(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def _download_file(
    url: str,
    dst_file_path: str,
    timeout: int = 120,
    overwrite: bool = False,
    print_info: bool = True,
):
    if os.path.exists(dst_file_path) and (not overwrite):
        return
    check_dir(os.path.dirname(os.path.abspath(dst_file_path)))
    tmp_file_path = dst_file_path + ".part"
    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)
    if print_info:
        print(f"Downloading: {url} -> {dst_file_path}")
    req = urllib.request.Request(url, headers={"User-Agent": "DeSide"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp_file_path, "wb") as out:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp_file_path, dst_file_path)
    if print_info:
        size = os.path.getsize(dst_file_path)
        md5 = _md5_file(dst_file_path)
        print(f"Downloaded: {dst_file_path} | size={size} bytes | md5={md5}")


def _resolve_figshare_file_download_url(doi: str, target_file_name: str, timeout: int = 30) -> str:
    q = urllib.parse.quote(doi, safe="")
    query_url = f"https://api.figshare.com/v2/articles?doi={q}"
    req = urllib.request.Request(query_url, headers={"User-Agent": "DeSide"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        articles = json.loads(r.read().decode("utf-8"))
    if not articles:
        raise RuntimeError(f"Cannot resolve Figshare article by DOI: {doi}")
    article_id = articles[0].get("id")
    if article_id is None:
        raise RuntimeError(f"Invalid Figshare API response for DOI: {doi}")
    article_url = f"https://api.figshare.com/v2/articles/{article_id}"
    req = urllib.request.Request(article_url, headers={"User-Agent": "DeSide"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        article = json.loads(r.read().decode("utf-8"))
    files = article.get("files", [])
    if not files:
        raise RuntimeError(f"No files found for Figshare article: {doi}")
    for f in files:
        if f.get("name") == target_file_name:
            return f.get("download_url")
    if len(files) == 1:
        return files[0].get("download_url")
    available = [f.get("name") for f in files]
    raise RuntimeError(
        f'Cannot find file "{target_file_name}" in Figshare article: {doi}. Available files: {available}'
    )


def _ensure_pretrained_assets(
    model_dir: str,
    dataset_dir: str,
    model_name: str = "DeSide",
    pathway_file_paths: list = None,
    timeout: int = 120,
    overwrite: bool = False,
    print_info: bool = True,
):
    required_model_files = [
        f"model_{model_name}.ckpt",
        "model_config.json",
        "celltypes.txt",
        "genes.txt",
        "genes_for_gep.txt",
        "genes_for_pathway_profile.txt",
    ]
    if pathway_file_paths is None:
        pathway_file_paths = _get_default_pathway_file_paths(dataset_dir=dataset_dir)
    check_dir(model_dir)
    check_dir(os.path.join(dataset_dir, "gene_set"))

    model_file_name = f"model_{model_name}.ckpt"
    model_file_path = os.path.join(model_dir, model_file_name)
    if not os.path.exists(model_file_path) or overwrite:
        download_url = _resolve_figshare_file_download_url(
            doi=_DEFAULT_PRETRAINED_MODEL_DOI,
            target_file_name=model_file_name,
            timeout=min(30, timeout),
        )
        _download_file(download_url, model_file_path, timeout=timeout, overwrite=overwrite, print_info=print_info)

    for name in required_model_files:
        if name == model_file_name:
            continue
        url = f"{_DEFAULT_GITHUB_RAW_BASE}/DeSide_model/{name}"
        _download_file(url, os.path.join(model_dir, name), timeout=timeout, overwrite=overwrite, print_info=print_info)

    for local_fp, fn in zip(pathway_file_paths, _DEFAULT_PRETRAINED_PATHWAY_FILES):
        url = f"{_DEFAULT_GITHUB_RAW_BASE}/datasets/gene_set/{fn}"
        _download_file(url, local_fp, timeout=timeout, overwrite=overwrite, print_info=print_info)


def _validate_pretrained_assets(model_dir: str, pathway_file_paths: list = None, model_name: str = "DeSide"):
    required_model_files = [
        f"model_{model_name}.ckpt",
        "model_config.json",
        "celltypes.txt",
        "genes.txt",
        "genes_for_gep.txt",
        "genes_for_pathway_profile.txt",
    ]
    missing_model_files = [i for i in required_model_files if not os.path.exists(os.path.join(model_dir, i))]
    if missing_model_files:
        raise FileNotFoundError(
            f'Missing PyTorch pre-trained model files in "{model_dir}". '
            f"Missing files: {missing_model_files}. If you only have the legacy TensorFlow assets, "
            f"use the legacy TensorFlow release documented in README."
        )
    if pathway_file_paths is not None:
        missing_pathway_files = [i for i in pathway_file_paths if not os.path.exists(i)]
        if missing_pathway_files:
            raise FileNotFoundError(
                "Missing pathway gene-set files required by the pre-trained model. "
                f"Missing files: {missing_pathway_files}"
            )


def predict_with_pretrained_model(
    input_file,
    output_file_path: str = None,
    model_dir: str = "./DeSide_model",
    dataset_dir: str = "./datasets",
    exp_type: str = "TPM",
    transpose: bool = True,
    print_info: bool = True,
    scaling_by_constant: bool = True,
    scaling_by_sample: bool = False,
    pathway_mask: pd.DataFrame = None,
    pathway_file_paths: list = None,
    model_name: str = "DeSide",
    auto_download: bool = True,
    download_timeout: int = 120,
    overwrite: bool = False,
):
    if (pathway_mask is None) and (pathway_file_paths is None):
        pathway_file_paths = _get_default_pathway_file_paths(dataset_dir=dataset_dir)
    if auto_download:
        _ensure_pretrained_assets(
            model_dir=model_dir,
            dataset_dir=dataset_dir,
            model_name=model_name,
            pathway_file_paths=pathway_file_paths,
            timeout=download_timeout,
            overwrite=overwrite,
            print_info=print_info,
        )
    if pathway_mask is None:
        _validate_pretrained_assets(model_dir=model_dir, pathway_file_paths=pathway_file_paths, model_name=model_name)
        pathway_mask = read_gene_set(pathway_file_paths)
    else:
        _validate_pretrained_assets(model_dir=model_dir, model_name=model_name)
    if output_file_path is not None:
        check_dir(os.path.dirname(os.path.abspath(output_file_path)))
    deside_model = DeSide(model_dir=model_dir, model_name=model_name)
    return deside_model.predict(
        input_file=input_file,
        output_file_path=output_file_path,
        exp_type=exp_type,
        transpose=transpose,
        print_info=print_info,
        scaling_by_sample=scaling_by_sample,
        scaling_by_constant=scaling_by_constant,
        hyper_params=_get_default_pretrained_hyper_params(),
        pathway_mask=pathway_mask,
    )


class DeSide(object):
    def __init__(self, model_dir: str, log_file_path: str = None, model_name: str = "DeSide"):
        self.model_dir = model_dir
        self.model = None
        self.cell_types = None
        self.gene_list = None
        self.model_name = model_name
        self.min_cell_fraction = 0.0001
        self.model_file_path = os.path.join(self.model_dir, f"model_{model_name}.ckpt")
        self.model_config_file_path = os.path.join(self.model_dir, "model_config.json")
        self.training_config_file_path = os.path.join(self.model_dir, "training_config.json")
        self.checkpoint_manifest_file_path = os.path.join(self.model_dir, "checkpoint_paths.json")
        self.cell_type_file_path = os.path.join(self.model_dir, "celltypes.txt")
        self.gene_list_file_path = os.path.join(self.model_dir, "genes.txt")
        self.gene_list_for_gep_file_path = os.path.join(self.model_dir, "genes_for_gep.txt")
        self.gene_list_for_pathway_profile_file_path = os.path.join(
            self.model_dir,
            "genes_for_pathway_profile.txt",
        )
        self.training_set_file_path = None
        self.hyper_params = None
        self.model_config = None
        self.one_minus_alpha = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if log_file_path is None:
            log_file_path = os.path.join(self.model_dir, "log.txt")
        self.log_file_path = log_file_path
        check_dir(self.model_dir)

    def _build_model(self, input_shape, output_shape, hyper_params, n_pathway: int = 0):
        self.hyper_params = hyper_params
        hidden_units = list(hyper_params["architecture"][0])
        dropout_rates = list(hyper_params["architecture"][1])
        self.model_config = {
            "input_dim": int(input_shape),
            "output_dim": int(output_shape),
            "hidden_units": hidden_units,
            "dropout_rates": dropout_rates,
            "normalization": hyper_params.get("normalization"),
            "normalization_layer": list(
                hyper_params.get("normalization_layer", [1] * (len(hidden_units) + 1))
            ),
            "last_layer_activation": hyper_params.get("last_layer_activation"),
            "pathway_network": bool(hyper_params.get("pathway_network", False)),
            "n_pathway": int(n_pathway),
            "pathway_hidden_units": [],
            "pathway_dropout_rates": [],
        }
        if self.model_config["pathway_network"]:
            self.model_config["pathway_hidden_units"] = list(
                hyper_params["architecture_for_pathway_network"][0]
            )
            self.model_config["pathway_dropout_rates"] = list(
                hyper_params["architecture_for_pathway_network"][1]
            )
        self.model = build_deside_model(self.model_config).to(self.device)
        return self.model

    @staticmethod
    def _split_inputs_for_dataset(
        x: pd.DataFrame,
        pathway_network: bool,
        pathway_mask: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        x_by_branch = get_x_by_pathway_network(x, pathway_network=pathway_network, pathway_mask=pathway_mask)
        if isinstance(x_by_branch, dict):
            pathways = pathway_mask.columns.to_list()
            x_gep = x.loc[:, ~x.columns.isin(pathways)].copy()
            x_pathway = x.loc[:, x.columns.isin(pathways)].copy()
            return x_gep, x_pathway
        return x.copy(), None

    def _save_json(self, file_path: str, payload: dict):
        with open(file_path, "w", encoding="utf-8") as f_handle:
            json.dump(payload, fp=f_handle, indent=2)

    def _write_training_metadata(self, training_history, training_config: dict):
        if self.model_config is not None:
            self._save_json(self.model_config_file_path, self.model_config)
        self._save_json(self.training_config_file_path, training_config)
        checkpoint_manifest = {
            "best_model_path": os.path.basename(training_history.best_model_path)
            if training_history.best_model_path
            else "",
            "last_model_path": os.path.basename(training_history.last_model_path),
            "default_model_path": os.path.basename(self.model_file_path),
        }
        self._save_json(self.checkpoint_manifest_file_path, checkpoint_manifest)

    def _resolve_checkpoint_for_loading(self) -> str:
        if os.path.exists(self.model_file_path):
            return self.model_file_path
        if os.path.exists(self.checkpoint_manifest_file_path):
            with open(self.checkpoint_manifest_file_path, "r", encoding="utf-8") as f_handle:
                manifest = json.load(f_handle)
            for key in ("best_model_path", "last_model_path"):
                rel_path = manifest.get(key, "")
                if rel_path:
                    candidate = os.path.join(self.model_dir, rel_path)
                    if os.path.exists(candidate):
                        return candidate
        raise FileNotFoundError(f"No PyTorch checkpoint found under model_dir: {self.model_dir}")

    def _load_model_config(self) -> dict:
        if self.model_config is not None:
            return self.model_config
        if os.path.exists(self.model_config_file_path):
            with open(self.model_config_file_path, "r", encoding="utf-8") as f_handle:
                self.model_config = json.load(f_handle)
                return self.model_config
        key_params_path = os.path.join(self.model_dir, "key_params.txt")
        if os.path.exists(key_params_path):
            with open(key_params_path, "r", encoding="utf-8") as f_handle:
                key_params = json.load(f_handle)
            self.hyper_params = key_params.get("hyper_params")
        if self.hyper_params is None:
            raise FileNotFoundError(
                f"Cannot reconstruct model configuration. Missing {self.model_config_file_path}."
            )
        gene_list = self.get_gene_list()
        cell_types = self.get_cell_type()
        pathway_network = bool(self.hyper_params.get("pathway_network", False))
        gene_list_for_gep = self.get_gene_list_for_gep()
        if pathway_network and gene_list_for_gep:
            input_dim = len(gene_list_for_gep)
            n_pathway = max(0, len(gene_list) - input_dim)
        else:
            input_dim = len(gene_list)
            n_pathway = 0
        self._build_model(
            input_shape=input_dim,
            output_shape=len(cell_types),
            hyper_params=self.hyper_params,
            n_pathway=n_pathway,
        )
        return self.model_config

    def _load_trained_model(self):
        model_config = self._load_model_config()
        if self.model is None:
            self.model = build_deside_model(model_config).to(self.device)
        checkpoint_path = self._resolve_checkpoint_for_loading()
        self.model = load_model_state_from_checkpoint(self.model, checkpoint_path, map_location=str(self.device))
        self.model.to(self.device)
        return self.model

    def train_model(
        self,
        training_set_file_path: Union[str, list],
        hyper_params: dict,
        cell_types: list = None,
        scaling_by_sample: bool = False,
        callback: bool = True,
        n_epoch: int = 10000,
        metrics: str = "mse",
        n_patience: int = 100,
        scaling_by_constant=True,
        remove_cancer_cell=True,
        fine_tune=False,
        one_minus_alpha: bool = False,
        verbose=1,
        pathway_mask=None,
        method_adding_pathway="add_to_end",
        input_gene_list: str = None,
        filtered_gene_list: list = None,
        group_cell_types: dict = None,
    ):
        del metrics  # Lightning logs fixed regression metrics for the migrated implementation.
        self.one_minus_alpha = one_minus_alpha
        self.training_set_file_path = training_set_file_path
        if os.path.exists(self.model_file_path) and not fine_tune:
            print(f"Previous model existed: {self.model_file_path}")
            return

        print_msg("Start to training model...", log_file_path=self.log_file_path)
        learning_rate = hyper_params["learning_rate"]
        loss_function_alpha = hyper_params["loss_function_alpha"]
        batch_size = hyper_params["batch_size"]
        optimizer_name = str(hyper_params.get("optimizer", "adamw"))
        weight_decay = float(hyper_params.get("weight_decay", 1e-5))
        lr_scheduler_name = str(hyper_params.get("lr_scheduler", "reduce_lr_on_plateau"))
        enable_lr_scheduler = bool(lr_scheduler_name and lr_scheduler_name.lower() not in {"none", "null", "false"})
        lr_scheduler_factor = float(hyper_params.get("lr_scheduler_factor", 0.5))
        lr_scheduler_patience = int(hyper_params.get("lr_scheduler_patience", 20))
        lr_scheduler_min_lr = float(hyper_params.get("lr_scheduler_min_lr", 1e-6))

        if isinstance(training_set_file_path, str):
            training_set_file_path = [training_set_file_path]
        n_sets = len(training_set_file_path)
        print_msg("Start to reading training set...", log_file_path=self.log_file_path)
        # Read-only first pass: materialize only column sets and metadata we need for alignment.
        # For N5K x ~18K-gene H5AD files pd.concat(..., join="inner") copies the full three
        # matrices twice (once per input to reorder columns, once during concat). Precomputing
        # the inner column set and slicing up front avoids the large reorder copies.
        raw_h5ad = []
        x_columns_per_file: list[pd.Index] = []
        y_columns_per_file: list[pd.Index] = []
        counter = 0
        for file_path in training_set_file_path:
            file_obj = ReadH5AD(file_path)
            raw_h5ad.append(file_obj)
            x_columns_per_file.append(pd.Index(file_obj.dataset.var.index))
            y_columns_per_file.append(pd.Index(file_obj.dataset.obs.columns))
        common_x_cols = x_columns_per_file[0]
        for idx in x_columns_per_file[1:]:
            common_x_cols = common_x_cols.intersection(idx)
        common_y_cols = y_columns_per_file[0]
        for idx in y_columns_per_file[1:]:
            common_y_cols = common_y_cols.intersection(idx)
        # Keep the original gene/cell-type order from the first file for reproducibility.
        x_list, y_list = [], []
        for file_obj in raw_h5ad:
            # Use the optimized ReadH5AD entry point: no copy (concat will copy anyway),
            # no eager round(3), restricted to the intersection columns. This avoids two
            # elementwise passes and two full-matrix copies per input file.
            _x_full = file_obj.get_df(copy=False, round_decimals=None)
            _y_full = file_obj.get_cell_fraction(copy=False, round_decimals=None)
            _x = _x_full.loc[:, common_x_cols]
            _y = _y_full.loc[:, common_y_cols] if _y_full is not None else None
            # Disambiguate duplicate sample names across training set splits (e.g., a given
            # sample ID appears in both dirichlet and sparse h5ads). Safe only because we
            # never join on index after this point; training is order-based.
            _x.index = _x.index.astype(str) + "_" + str(counter)
            if _y is not None:
                _y.index = _y.index.astype(str) + "_" + str(counter)
            x_list.append(_x)
            y_list.append(_y)
            counter += 1
        # All frames now share identical column order; join="outer" is equivalent to inner and
        # avoids Pandas' join="inner" re-materialization of all inputs.
        x = pd.concat(x_list, axis=0, ignore_index=False, copy=False)
        y = pd.concat([yf for yf in y_list if yf is not None], axis=0, ignore_index=False, copy=False)
        del x_list, y_list, raw_h5ad

        if group_cell_types is not None:
            columns_set = set(y.columns)
            for g, _cell_types in group_cell_types.items():
                _cell_types_list = list(_cell_types)
                if not _cell_types_list:
                    continue
                # Ensure the grouped output column g is considered alongside its subtypes. The
                # group name can legitimately appear as a raw column in SimuTME outputs (both when
                # the file exports only the pre-grouped column, and when it exports both the group
                # name and raw subtypes alongside one another). Treat g as another subtype so the
                # sum is always written back to g cleanly and never silently dropped.
                subtype_candidates = list(_cell_types_list)
                if g in columns_set and g not in subtype_candidates:
                    subtype_candidates.append(g)
                missing_in_y = [c for c in _cell_types_list if c not in columns_set]
                # Which of (g plus subtypes) actually exist in the current y?
                present_any = [c for c in subtype_candidates if c in columns_set]
                if missing_in_y:
                    # Heterogeneous SimuTME outputs: at least one listed subtype is missing. We
                    # only proceed when the group name g itself exists in y; that column becomes
                    # the canonical accumulator, and any subtypes (other than g) that do exist are
                    # folded into it. We do NOT drop g here: g in y == g equals the group column
                    # itself, so dropping it would erase the final grouped output.
                    if g in columns_set:
                        present_subtypes_except_g = [c for c in _cell_types_list if c in columns_set and c != g]
                        if present_subtypes_except_g:
                            y[g] = y[g] + y[present_subtypes_except_g].sum(axis=1)
                            y = y.drop(columns=present_subtypes_except_g)
                            columns_set = set(y.columns)
                        # g still exists and is the merged column.
                        continue
                    if (
                        len(missing_in_y) == 1
                        and len(_cell_types_list) == 1
                        and missing_in_y[0] == g
                    ):
                        continue
                    raise KeyError(
                        f"group_cell_types[{g!r}] references subtypes not present in the merged training "
                        f"set cell-fraction columns: {missing_in_y}. Available columns: {sorted(columns_set)}."
                    )
                if len(present_any) > 1:
                    y[g] = y[present_any].sum(axis=1)
                    y = y.drop(columns=[c for c in present_any if c != g])
                    columns_set = set(y.columns)
                else:
                    single = present_any[0]
                    if single != g:
                        if g in columns_set:
                            raise KeyError(
                                f"group_cell_types[{g!r}] maps to single subtype {single!r}, but a column "
                                f"named {g!r} already exists in the training set y columns. Refusing to "
                                f"overwrite; rename one of the sides or merge subtypes explicitly."
                            )
                        y = y.rename(columns={single: g})
                        columns_set = set(y.columns)
        assert np.all(x.index == y.index), "The order of samples in x and y are not the same!"
        y = y.loc[y.sum(axis=1) > 0, :]
        x = x.loc[y.index, :]
        if self.one_minus_alpha:
            y = 1 - y

        x_obj = ReadExp(x, exp_type="log_space")
        if len(training_set_file_path) >= 2:
            x_obj.to_tpm()
            x_obj.to_log2cpm1p()

        if pathway_mask is not None:
            if input_gene_list == "intersection_with_pathway_genes":
                pathway_genes_set = set(pathway_mask.index)
                gep_gene_list = [gene for gene in x_obj.exp.columns if gene in pathway_genes_set]
                x_obj.align_with_gene_list(gene_list=gep_gene_list, fill_not_exist=True)
            elif input_gene_list == "filtered_genes" and filtered_gene_list is not None:
                gep_gene_list = list(filtered_gene_list)
            else:
                gep_gene_list = list(x_obj.exp.columns)
            pathway_profile_gene_list = list(x_obj.exp.columns)
            if method_adding_pathway == "add_to_end":
                pd.DataFrame(gep_gene_list).to_csv(self.gene_list_for_gep_file_path, sep="\t")
            pd.DataFrame(pathway_profile_gene_list).to_csv(self.gene_list_for_pathway_profile_file_path, sep="\t")
            x_obj = self._get_pathway_profiles(
                x_obj,
                pathway_mask,
                method=method_adding_pathway,
                filtered_gene_list=gep_gene_list,
            )

        if scaling_by_sample:
            x_obj.do_scaling(round_decimals=None)
        if scaling_by_constant:
            x_obj.do_scaling_by_constant()
        x = x_obj.get_exp(round_decimals=None)

        self.gene_list = x.columns.to_list()
        if cell_types is None:
            self.cell_types = y.columns.to_list()
        else:
            assert len(cell_types) > 0, "cell_types should not be empty."
            missing_cell_types = [i for i in cell_types if i not in y.columns.to_list()]
            assert len(missing_cell_types) == 0, (
                "Provided cell types (" + ", ".join(missing_cell_types) + ") are not in the training set."
            )
            self.cell_types = cell_types
        if remove_cancer_cell:
            self.cell_types = [i for i in self.cell_types if i != "Cancer Cells"]
        y = y.loc[:, self.cell_types]

        pd.DataFrame(self.cell_types).to_csv(self.cell_type_file_path, sep="\t")
        pd.DataFrame(self.gene_list).to_csv(self.gene_list_file_path, sep="\t")

        print(f"   Use the following cell types: {self.cell_types} during training.")
        print(f"   The shape of X is: {x.shape}, (n_sample, n_gene)")
        print(f"   The shape of y is: {y.shape}, (n_sample, n_cell_type)")

        pathway_network = bool(hyper_params["pathway_network"])
        x_gep, x_pathway = self._split_inputs_for_dataset(x, pathway_network=pathway_network, pathway_mask=pathway_mask)
        if not fine_tune:
            self._build_model(
                input_shape=x_gep.shape[1],
                output_shape=len(self.cell_types),
                hyper_params=hyper_params,
                n_pathway=0 if x_pathway is None else x_pathway.shape[1],
            )
        else:
            self._load_trained_model()

        dataset = DeSideDataset(
            gep=x_gep,
            pathway_profile=x_pathway,
            cell_fraction=y,
            sample_ids=list(x.index),
        )
        dataset_split = split_deside_dataset(
            dataset,
            validation_split=float(hyper_params.get("validation_split", 0.2)),
            seed=int(hyper_params.get("validation_seed", 42)),
        )
        print(
            "   The following loss function will be used:",
            loss_function_alpha,
            "* mae +",
            (1 - loss_function_alpha),
            "* rmse",
        )
        from ..trainers import train_deside_lightning

        training_history = train_deside_lightning(
            model=self.model,
            train_dataset=dataset_split.train,
            val_dataset=dataset_split.val,
            learning_rate=learning_rate,
            loss_alpha=loss_function_alpha,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            enable_lr_scheduler=enable_lr_scheduler,
            lr_scheduler_name=lr_scheduler_name,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_min_lr=lr_scheduler_min_lr,
            model_dir=self.model_dir,
            max_epochs=n_epoch,
            batch_size=batch_size,
            callback=callback,
            patience=n_patience,
            verbose=verbose,
        )

        best_or_last = training_history.best_model_path or training_history.last_model_path
        if best_or_last and os.path.abspath(best_or_last) != os.path.abspath(self.model_file_path):
            shutil.copy2(best_or_last, self.model_file_path)

        training_config = {
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "weight_decay": weight_decay,
            "lr_scheduler": lr_scheduler_name,
            "lr_scheduler_factor": lr_scheduler_factor,
            "lr_scheduler_patience": lr_scheduler_patience,
            "lr_scheduler_min_lr": lr_scheduler_min_lr,
            "loss_function_alpha": loss_function_alpha,
            "batch_size": batch_size,
            "n_epoch": n_epoch,
            "n_patience": n_patience,
            "validation_split": float(hyper_params.get("validation_split", 0.2)),
            "validation_seed": int(hyper_params.get("validation_seed", 42)),
            "scaling_by_sample": bool(scaling_by_sample),
            "scaling_by_constant": bool(scaling_by_constant),
            "remove_cancer_cell": bool(remove_cancer_cell),
            "fine_tune": bool(fine_tune),
            "framework": "pytorch-lightning",
        }
        self._write_training_metadata(training_history, training_config)

        _plot_loss(training_history.history_df, output_dir=self.model_dir, y_label="loss_function")
        key_params_file_path = os.path.join(self.model_dir, "key_params.txt")
        print(f"   Key parameters during model training will be saved in {key_params_file_path}.")
        self.save_params(key_params_file_path)
        print_msg("Training done.", log_file_path=self.log_file_path)

    @staticmethod
    def _get_pathway_profiles(x_obj, pathway_mask: pd.DataFrame, method="add_to_end", filtered_gene_list=None):
        if x_obj.file_type == "log_space":
            x_obj.to_tpm()
        if filtered_gene_list is not None:
            filtered_set = set(filtered_gene_list)
            intersect_before = [g for g in x_obj.exp.columns if g in filtered_set]
            needs_align = (
                len(intersect_before) != len(x_obj.exp.columns)
                or len(intersect_before) != len(filtered_gene_list)
            )
            if needs_align:
                x_obj.align_with_gene_list(gene_list=filtered_gene_list, fill_not_exist=True)
        x = x_obj.get_exp(round_decimals=None, copy=False)
        x_columns = x.columns
        x_cols_set = set(x_columns)
        pm_index_set = set(pathway_mask.index)
        common_genes = list(x_cols_set & pm_index_set)
        print("common genes between training set and pathway mask:", len(common_genes))
        genes_only_in_x = list(x_cols_set - pm_index_set)
        if len(genes_only_in_x) > 0:
            print("genes only in training set:", len(genes_only_in_x))
            desired_index = list(pathway_mask.index) + genes_only_in_x
            pathway_mask = pathway_mask.reindex(index=desired_index, fill_value=0.0, copy=False)
        pathway_mask = pathway_mask.reindex(index=x_columns, fill_value=0.0, copy=False)
        if method == "convert":
            x_values = x.to_numpy(dtype=np.float32, copy=False)
            pm_values = pathway_mask.to_numpy(dtype=np.float32, copy=False)
            x = pd.DataFrame(
                data=x_values @ pm_values,
                index=x.index,
                columns=pathway_mask.columns,
                copy=False,
            )
        elif method == "add_to_end":
            x_values = x.to_numpy(dtype=np.float32, copy=False)
            pm_values = pathway_mask.to_numpy(dtype=np.float32, copy=False)
            x_pathway_profiles = pd.DataFrame(
                data=x_values @ pm_values,
                index=x.index,
                columns=pathway_mask.columns,
                copy=False,
            )
            print(
                "   Pathway profile matmul complete.",
                f"x shape={x.shape}, pathway_profiles shape={x_pathway_profiles.shape}",
                flush=True,
            )
            x = pd.concat([x, x_pathway_profiles], axis=1, copy=False)
            print(f"   Concat [GEP + pathway profiles] complete. Combined shape={x.shape}", flush=True)
        _log_in = x.to_numpy(dtype=np.float32, copy=False)
        _log_out = np.log2(_log_in + 1.0)
        x = pd.DataFrame(
            data=_log_out,
            index=x.index,
            columns=x.columns,
            copy=False,
        )
        del _log_in, _log_out
        print("x shape:", x.shape, flush=True)
        result = ReadExp(x, exp_type="log_space")
        return result

    def get_x_before_predict(
        self,
        input_file,
        exp_type,
        transpose: bool = False,
        print_info: bool = True,
        scaling_by_sample: bool = False,
        scaling_by_constant: bool = True,
        pathway_mask: pd.DataFrame = None,
        method_adding_pathway: str = "add_to_end",
    ):
        if self.gene_list is None:
            self.gene_list = self.get_gene_list()
        if exp_type not in ["TPM", "log_space"]:
            raise ValueError(f'exp_type should be "TPM" or "log_space", "{exp_type}" is invalid.')
        if isinstance(input_file, str) and ".h5ad" in input_file:
            read_h5ad_obj = ReadH5AD(input_file)
            _input_data = read_h5ad_obj.get_df()
            read_df_obj = ReadExp(_input_data, exp_type=exp_type, transpose=transpose)
        elif (
            isinstance(input_file, str)
            and np.any([i in input_file for i in [".csv", ".txt", ".tsv"]])
        ) or isinstance(input_file, pd.DataFrame):
            read_df_obj = ReadExp(input_file, exp_type=exp_type, transpose=transpose)
        else:
            raise Exception(
                f"The current file path of raw data is {input_file}, only '*.csv', '*.txt', '*.tsv', or '*.h5ad' "
                f"is supported."
            )

        if pathway_mask is not None:
            gene_list_for_pathway_profile = self.get_gene_list_for_pathway_profile()
            gene_list_for_gep = None
            if method_adding_pathway == "add_to_end":
                gene_list_for_gep = self.get_gene_list_for_gep()
            intersection_genes = list(set(gene_list_for_pathway_profile) & set(read_df_obj.exp.columns.to_list()))
            if len(intersection_genes) != len(gene_list_for_pathway_profile) or len(intersection_genes) != len(
                read_df_obj.exp.columns.to_list()
            ):
                read_df_obj.align_with_gene_list(gene_list=gene_list_for_pathway_profile, fill_not_exist=True)
            print(f"   {read_df_obj.exp.shape[1]} genes will be used to construct the pathway profiles.")
            read_df_obj = self._get_pathway_profiles(
                read_df_obj,
                pathway_mask,
                method=method_adding_pathway,
                filtered_gene_list=gene_list_for_gep,
            )

        pathway_list = pathway_mask is not None
        read_df_obj.align_with_gene_list(gene_list=self.gene_list, fill_not_exist=True, pathway_list=pathway_list)
        if pathway_mask is None and exp_type != "log_space":
            read_df_obj.to_log2cpm1p()
        if scaling_by_sample:
            read_df_obj.do_scaling()
        if scaling_by_constant:
            read_df_obj.do_scaling_by_constant()
        x = read_df_obj.get_exp()
        _gene_list = x.columns.to_list()
        if len(_gene_list) != len(set(_gene_list)):
            x = x.loc[:, ~x.columns.duplicated(keep="first")]
        assert np.all(x.columns == self.gene_list), (
            "The gene list in input file is not the same as the gene list in pre-trained model."
        )
        if print_info:
            print(f"   > {len(self.gene_list)} genes included in pre-trained model and will be used for prediction.")
            print(f"   The shape of X is: {x.shape}, (n_sample, n_gene)")
        return x

    def predict(
        self,
        input_file,
        exp_type,
        output_file_path: str = None,
        transpose: bool = False,
        print_info: bool = True,
        add_cell_type: bool = False,
        scaling_by_constant=True,
        scaling_by_sample=False,
        one_minus_alpha: bool = False,
        pathway_mask: pd.DataFrame = None,
        method_adding_pathway: str = "add_to_end",
        hyper_params: dict = None,
        cell_prop_threshold: float = None,
        cancer_cell_type_name: str = "Cancer Cells",
        fill_cancer_as_residual: bool = True,
        renormalize_non_cancer_after_threshold: bool = True,
    ):
        self.one_minus_alpha = one_minus_alpha
        if print_info:
            print("   Start to predict cell fractions by pre-trained model...")
        if self.cell_types is None:
            self.cell_types = self.get_cell_type()

        x = self.get_x_before_predict(
            input_file,
            exp_type,
            transpose=transpose,
            print_info=print_info,
            scaling_by_constant=scaling_by_constant,
            scaling_by_sample=scaling_by_sample,
            pathway_mask=pathway_mask,
            method_adding_pathway=method_adding_pathway,
        )

        if self.model is None:
            self._load_trained_model()
            print(f"   Pre-trained model loaded from {self._resolve_checkpoint_for_loading()}.")

        pathway_network = bool(self.model_config.get("pathway_network", False))
        x_gep, x_pathway = self._split_inputs_for_dataset(x, pathway_network=pathway_network, pathway_mask=pathway_mask)
        dataset = DeSideDataset(
            gep=x_gep,
            pathway_profile=x_pathway,
            sample_ids=list(x.index),
        )
        batch_size = 1024
        if hyper_params is not None:
            batch_size = int(hyper_params.get("batch_size", batch_size))

        pred_batches = []
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for batch in loader:
                gep = batch["gep"].to(self.device)
                pathway_profile = batch.get("pathway_profile")
                if pathway_profile is not None:
                    pathway_profile = pathway_profile.to(self.device)
                pred_batches.append(self.model(gep=gep, pathway_profile=pathway_profile).cpu().numpy())
        pred_result = np.vstack(pred_batches)
        pred_df = pd.DataFrame(pred_result, index=x.index.copy(), columns=self.cell_types)
        if self.one_minus_alpha:
            pred_df = 1 - pred_df
        threshold = self.min_cell_fraction if cell_prop_threshold is None else float(cell_prop_threshold)
        pred_df[pred_df.values < threshold] = 0
        if renormalize_non_cancer_after_threshold:
            for sample_id, row in pred_df.iterrows():
                row_sum = np.sum(row)
                if row_sum > 1:
                    pred_df.loc[sample_id] = row / row_sum

        if fill_cancer_as_residual and cancer_cell_type_name not in pred_df.columns:
            pred_df_with_1_others = pred_df.loc[:, [i for i in pred_df.columns if i != cancer_cell_type_name]].copy()
            pred_df_with_1_others["1-others"] = 1 - np.vstack(pred_df_with_1_others.sum(axis=1))
            pred_df_with_1_others.loc[pred_df_with_1_others["1-others"] < 0, "1-others"] = 0
            pred_df_with_1_others[cancer_cell_type_name] = pred_df_with_1_others["1-others"]
            pred_df = pred_df_with_1_others.copy()
        if add_cell_type:
            pred_df["pred_cell_type"] = self._pred_cell_type_by_cell_frac(pred_cell_frac=pred_df)
        if print_info:
            print("   Model prediction done.")
        if output_file_path is not None:
            pred_df.to_csv(output_file_path, float_format="%.3f")
            return None
        return pred_df

    def get_model(self):
        if self.model is None and os.path.exists(self._resolve_checkpoint_for_loading()):
            self._load_trained_model()
            print(f"   Pre-trained model loaded from {self._resolve_checkpoint_for_loading()}.")
        return self.model

    def get_parameters(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_file_path": self.model_file_path,
            "hyper_params": self.hyper_params,
            "training_set_file_path": self.training_set_file_path,
            "cell_type_file_path": self.cell_type_file_path,
            "gene_list_file_path": self.gene_list_file_path,
            "model_config_file_path": self.model_config_file_path,
            "training_config_file_path": self.training_config_file_path,
            "log_file_path": self.log_file_path,
        }

    def get_gene_list(self) -> list:
        if (self.gene_list is None) and os.path.exists(self.gene_list_file_path):
            self.gene_list = list(pd.read_csv(self.gene_list_file_path, sep="\t", index_col=0)["0"])
        return self.gene_list

    def get_gene_list_for_gep(self) -> list:
        gene_list_for_gep = []
        if os.path.exists(self.gene_list_for_gep_file_path):
            gene_list_for_gep = list(pd.read_csv(self.gene_list_for_gep_file_path, sep="\t", index_col=0)["0"])
        return gene_list_for_gep

    def get_gene_list_for_pathway_profile(self) -> list:
        gene_list_for_pathway_profile = []
        if os.path.exists(self.gene_list_for_pathway_profile_file_path):
            gene_list_for_pathway_profile = list(
                pd.read_csv(self.gene_list_for_pathway_profile_file_path, sep="\t", index_col=0)["0"]
            )
        return gene_list_for_pathway_profile

    def get_cell_type(self) -> list:
        if (self.cell_types is None) and os.path.exists(self.cell_type_file_path):
            self.cell_types = list(pd.read_csv(self.cell_type_file_path, sep="\t", index_col=0)["0"])
        return self.cell_types

    def save_params(self, output_file_path: str):
        key_params = self.get_parameters()
        with open(output_file_path, "w", encoding="utf-8") as f_handle:
            json.dump(key_params, fp=f_handle, indent=2)

    def _pred_cell_type_by_cell_frac(self, pred_cell_frac: pd.DataFrame) -> list:
        id2cell_type = {i: self.cell_types[i] for i in range(len(self.cell_types))}
        pred_id = pred_cell_frac.values.argmax(axis=1)
        return [id2cell_type[i] for i in pred_id]


def loss_fn_mae_rmse(y_true, y_pred, alpha=0.5):
    y_true_t = torch.as_tensor(y_true, dtype=torch.float32)
    y_pred_t = torch.as_tensor(y_pred, dtype=torch.float32)
    mae = torch.mean(torch.abs(y_true_t - y_pred_t))
    rmse = torch.sqrt(torch.mean((y_true_t - y_pred_t) ** 2))
    return alpha * mae + (1 - alpha) * rmse
