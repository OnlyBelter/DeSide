from .workflow import run_step4, run_step3
from .workflow import tcga_evaluation
from .bulk_simulation import run_bulk_simulation_from_config, run_bulk_simulation_from_config_file

__all__ = [
    "run_step3",
    "run_step4",
    "tcga_evaluation",
    "run_bulk_simulation_from_config",
    "run_bulk_simulation_from_config_file",
    "train_from_config",
    "train_from_config_file",
]


def train_from_config(*args, **kwargs):
    from .train import train_from_config as _train_from_config

    return _train_from_config(*args, **kwargs)


def train_from_config_file(*args, **kwargs):
    from .train import train_from_config_file as _train_from_config_file

    return _train_from_config_file(*args, **kwargs)
