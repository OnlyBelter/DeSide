from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _build_norm(normalization: Optional[str], n_features: int) -> Optional[nn.Module]:
    if normalization == "batch_normalization":
        return nn.BatchNorm1d(n_features)
    if normalization == "layer_normalization":
        return nn.LayerNorm(n_features)
    return None


def _get_activation_module(name: Optional[str]) -> Optional[nn.Module]:
    if name is None:
        return None
    name = name.lower()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "hard_sigmoid":
        return nn.Hardsigmoid()
    if name == "relu":
        return nn.ReLU()
    if name == "softmax":
        return nn.Softmax(dim=1)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation function: {name}")


def _init_linear(layer: nn.Linear) -> None:
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class DenseNormBlock(nn.Module):
    """Linear layer with optional normalization and ReLU to mirror Keras blocks."""

    def __init__(self, in_features: int, out_features: int, normalization: Optional[str], use_norm: bool):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=not use_norm)
        _init_linear(self.linear)
        self.norm = _build_norm(normalization, out_features) if use_norm else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.activation(x)


class FeatureBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: list[int],
        dropout_rates: list[float],
        normalization: Optional[str],
        normalization_layer: list[int],
    ):
        super().__init__()
        self.input_norm = None
        if normalization is not None and normalization_layer[0] == 1:
            self.input_norm = _build_norm(normalization, input_dim)

        blocks = []
        dropouts = []
        in_dim = input_dim
        for i, out_dim in enumerate(hidden_units):
            use_norm = bool(normalization is not None and normalization_layer[i + 1] == 1)
            blocks.append(DenseNormBlock(in_dim, out_dim, normalization, use_norm))
            dropouts.append(nn.Dropout(float(dropout_rates[i])) if float(dropout_rates[i]) > 0 else nn.Identity())
            in_dim = out_dim
        self.blocks = nn.ModuleList(blocks)
        self.dropouts = nn.ModuleList(dropouts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm is not None:
            x = self.input_norm(x)
        for block, dropout in zip(self.blocks, self.dropouts):
            x = block(x)
            x = dropout(x)
        return x


class DeSideRegressor(nn.Module):
    """PyTorch implementation of the DeSide feed-forward regressor."""

    def __init__(self, model_config: dict):
        super().__init__()
        self.model_config = model_config
        hidden_units = list(model_config["hidden_units"])
        dropout_rates = list(model_config["dropout_rates"])
        normalization = model_config.get("normalization")
        normalization_layer = list(
            model_config.get("normalization_layer", [1] * (len(hidden_units) + 1))
        )
        self.pathway_network = bool(model_config.get("pathway_network", False))
        self.output_activation = _get_activation_module(model_config.get("last_layer_activation"))

        self.gep_branch = FeatureBranch(
            input_dim=int(model_config["input_dim"]),
            hidden_units=hidden_units,
            dropout_rates=dropout_rates,
            normalization=normalization,
            normalization_layer=normalization_layer,
        )

        if self.pathway_network:
            p_hidden_units = list(model_config["pathway_hidden_units"])
            p_dropout_rates = list(model_config["pathway_dropout_rates"])
            self.pathway_branch = FeatureBranch(
                input_dim=int(model_config["n_pathway"]),
                hidden_units=p_hidden_units,
                dropout_rates=p_dropout_rates,
                normalization=normalization,
                normalization_layer=normalization_layer,
            )
            merge_in_dim = hidden_units[-1] + p_hidden_units[-1]
            self.merge_layer = DenseNormBlock(
                merge_in_dim,
                hidden_units[-1],
                normalization=None,
                use_norm=False,
            )
        else:
            self.pathway_branch = None
            self.merge_layer = None

        self.output_layer = nn.Linear(hidden_units[-1], int(model_config["output_dim"]))
        _init_linear(self.output_layer)

    def forward(self, gep: torch.Tensor, pathway_profile: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.gep_branch(gep)
        if self.pathway_network:
            if pathway_profile is None:
                raise ValueError("pathway_profile is required when pathway_network=True")
            p_features = self.pathway_branch(pathway_profile)
            features = torch.cat([features, p_features], dim=1)
            features = self.merge_layer(features)
        output = self.output_layer(features)
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output


def build_deside_model(model_config: dict) -> DeSideRegressor:
    return DeSideRegressor(model_config=model_config)
