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

## Install PyTorch FIRST (before DeSide)

DeSide 2.x depends on PyTorch and PyTorch Lightning. PyTorch must be installed
separately because the correct wheel depends on your GPU driver / CUDA version.
Running `pip install deside` without doing this first will almost always pull
the **CPU-only** PyTorch wheel (no GPU kernels), which causes the silent
`CUBLAS_STATUS_ARCH_MISMATCH` / `cudaErrorNoKernelImageForDevice` crash at
training start even though `torch.cuda.is_available()` reports `True`.

Pick your OS below, or use the official PyTorch selector to generate a
custom command exactly matching your driver:
<https://pytorch.org/get-started/locally/>
(for older releases, see <https://pytorch.org/get-started/previous-versions/>).

### Linux / Windows (GPU with CUDA 12.6 — recommended for modern NVIDIA GPUs)

```shell
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

> For other CUDA versions (12.8, 13.0, ROCm, or CPU-only), replace the wheel tag
> and URL as shown on the PyTorch install page. For **older drivers that only
> support CUDA ≤ 11.8**, install torch 2.3.1 with `--index-url
> https://download.pytorch.org/whl/cu118` (CUDA 11.8 works on drivers ≥ 450.80).

### macOS (Apple Silicon / Intel — no CUDA, uses CPU or MPS)

```shell
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
```

### Verify PyTorch works on your hardware (REQUIRED)

Run this check. **Do not install DeSide until this prints PASS.**

```shell
python3 - <<'PY'
import torch, sys
print(f"torch.__version__      = {torch.__version__}")
print(f"torch.version.cuda     = {torch.version.cuda}")
print(f"torch.cuda.is_available= {torch.cuda.is_available()}")
for i in range(min(torch.cuda.device_count() if torch.cuda.is_available() else 0, 4)):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name:30s}  SM={p.major}.{p.minor}  mem={p.total_memory//1024**3:>3d}GiB")
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = torch.randn(4, 4, device=dev, dtype=torch.float32)
b = torch.randn(4, 4, device=dev, dtype=torch.float32)
val = (a @ b).sum().item()
print(f"PASS: kernel ran on {dev.type}. 4x4@4x4 sum = {val:.6f}")
PY
```

If this raises an exception (e.g. `CUBLAS_STATUS_ARCH_MISMATCH`,
`no kernel image is available for execution on the device`), **reinstall PyTorch
with the correct CUDA wheel tag for your driver** — do NOT proceed to install
DeSide yet. DeSide ships a CUDA preflight probe that falls back to CPU on kernel
launch failures, so training will technically succeed but you will lose 15–20×
throughput.

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
