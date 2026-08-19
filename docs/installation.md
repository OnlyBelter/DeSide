# Installation

This part shows how to install DeSide in a virtual environment.

***

It is recommended to install DeSide in a virtual environment, such as `conda`.
This isolates the package from your system Python installation and prevents
any conflicts with other packages.

## Create a virtual environment

1. Install `conda` if you do not already have it. For the usage of `conda`,
   see the
   [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. Create a new virtual environment named `deside` and activate it:

```shell
conda create -n deside python=3.10
conda activate deside
```

## Update pip

DeSide uses `pyproject.toml` to manage dependencies, so you need `pip` version
21.3 or later.

```shell
python3 -m pip install --upgrade pip
```

## Install DeSide

Install the PyTorch-based DeSide 2.x line:

```shell
pip install deside>=2.0.0a0
```

The new line depends on PyTorch and PyTorch Lightning.

> [!NOTE]
> DeSide 2.x is a major upgrade. It does not load legacy TensorFlow `.h5`
> checkpoints from DeSide 1.x. If you need the TensorFlow-based release, see
> the legacy install guidance below.

## Troubleshooting on Apple Silicon

If installation fails on Apple Silicon macOS when building dependencies such as
`tables`, install `hdf5` and `pytables` from `conda-forge` first, then install
DeSide again:

```shell
conda install -c conda-forge hdf5 pytables
pip install deside>=2.0.0a0
```

## Legacy install (TensorFlow-based DeSide 1.x)

Use the instructions in this section only if you need the older TensorFlow
line. DeSide 1.x uses TensorFlow 2.11 and the legacy `.h5` model format.

1. Create and activate a compatible environment:

```shell
conda create -n deside-tf python=3.8
conda activate deside-tf
```

2. Install the 1.x release:

```shell
pip install deside<2.0
```

For reproducibility, use the pre-release tag or branch of the legacy line if
you need a specific snapshot of DeSide 1.x.
