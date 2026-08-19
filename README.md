# DeSide: Cellular Deconvolution of Bulk RNA-seq
<img src="https://raw.githubusercontent.com/OnlyBelter/DeSide/main/docs/_static/logo.png" width="300">

![PyPI version](https://img.shields.io/pypi/v/deside)
![Install with pip](https://img.shields.io/badge/Install%20with-pip-blue)
![MIT](https://img.shields.io/badge/License-MIT-black)

## What is DeSide?

DeSide is a DEep-learning and SIngle-cell based DEconvolution method for solid tumors, which can be used to infer cellular proportions of different cell types from bulk RNA-seq data.

DeSide consists of the following four parts (see figure below):
- DNN Model
- Single Cell Dataset Integration
- Cell Proportion Generation
- Bulk Tumor Synthesis

<img src="https://raw.githubusercontent.com/OnlyBelter/DeSide/main/Fig.1a_b.svg" width="800" alt="Overview of DeSide">

In this repository, we provide the code for implementing these four parts and visualizing the results.

## Requirements

DeSide requires Python 3.9 or higher. It is tested on Linux and macOS and is
expected to work on Windows as well. The current development line on branch
`pytorch-dev` uses PyTorch and PyTorch Lightning instead of TensorFlow.

- torch>=2.0.0
- lightning==2.5.1
- scikit-learn>=0.24.2
- anndata>=0.8.0
- scanpy>=1.8.0
- umap-learn==0.5.1
- pandas>=1.5.3
- numpy>=1.22
- matplotlib>=3.6,<3.10
- seaborn>=0.11.2
- bbknn>=1.5.1
- SciencePlots

> [!NOTE]
> This is a major upgrade. DeSide 2.x does not load legacy TensorFlow `.h5`
> checkpoints from DeSide 1.x. If you need the older TensorFlow-based line,
> install `deside<2.0` in a Python 3.8 environment.

## Installation

`pip` should work out of the box.

1. Create a virtual environment:

```shell
conda create -n deside python=3.10
conda activate deside
```

2. Update `pip`:

```shell
python3 -m pip install --upgrade pip
```

3. Install DeSide:

```shell
pip install deside>=2.0.0a0
```

### Troubleshooting on Apple Silicon

If installation fails on Apple Silicon macOS when building dependencies such as
`tables`, install `hdf5` and `pytables` from `conda-forge` first, then install
DeSide again:

```shell
conda install -c conda-forge hdf5 pytables
pip install deside>=2.0.0a0
```

### Legacy install (TensorFlow-based DeSide 1.x)

Use this section only if you need the older TensorFlow line.

```shell
conda create -n deside-tf python=3.8
conda activate deside-tf
python3 -m pip install --upgrade pip
pip install deside<2.0
```

## Usage Examples

Usage examples can be found at [DeSide_mini_example](https://github.com/OnlyBelter/DeSide_mini_example).

Three examples are provided:

- Using a pre-trained model
- Training a model from scratch
- Generating a synthetic dataset

Example 1 can be run with one function call after import. The current PyTorch
development branch keeps the same user-facing helper, but saves and loads
models in a PyTorch Lightning checkpoint directory instead of TensorFlow `.h5`
files.

```python
import deside

deside.predict_with_pretrained_model(
    input_file="path/xx_TPM.csv",
    output_file_path="./results/y_pred.csv"
)
```

This helper expects the same local assets used in the mini example:

- `./DeSide_model/` for the pre-trained model directory
- `./datasets/gene_set/` for the pathway `.gmt` files

By default, missing Example 1 assets are downloaded automatically into those
folders with explicit download logs. To disable auto-download, pass
`auto_download=False`.

You can also run Example 1 from this repository as a script:

```bash
python examples/example1_pretrained_model.py \
  --input-file path/xx_TPM.csv \
  --output-file ./results/y_pred.csv
```

## Documentation

For detailed documentation, see https://deside.readthedocs.io/. The
documentation covers installation, usage examples, datasets, and the public
classes and functions.

## Changelog

See [docs/changelog.md](docs/changelog.md) for the full change history.

- v2.0.0a0 (August 19, 2026, branch `pytorch-dev`): start the major upgrade
  from TensorFlow/Keras to PyTorch and PyTorch Lightning, keep DeSide training
  and prediction facades stable, introduce Lightning checkpoint directories,
  and document the legacy TensorFlow-based 1.x install path.
- v1.3.3 (June 16, 2026): add a one-call pre-trained model API, automatic
  downloads for Example 1 assets, and a runnable Example 1 script.


## License
DeSide can be used under the terms of the MIT License.

## Contact
Any questions or suggestions about DeSide are welcomed! Please report it on [issues](https://github.com/OnlyBelter/DeSide/issues), or contact Xin Xiong (onlybelter@outlook.com) or Xuefei Li (xuefei.li@siat.ac.cn).

## Manuscript
```text
@article{Xiong2023.05.11.540466,
	author = {Xin Xiong and Yerong Liu and Dandan Pu and Zhu Yang and Zedong Bi and Liang Tian and Xuefei Li},
	title = {DeSide: A unified deep learning approach for cellular decomposition of bulk tumors based on limited scRNA-seq data},
	elocation-id = {2023.05.11.540466},
	year = {2023},
	doi = {10.1101/2023.05.11.540466},
	URL = {https://www.biorxiv.org/content/early/2023/05/14/2023.05.11.540466},
	eprint = {https://www.biorxiv.org/content/early/2023/05/14/2023.05.11.540466.full.pdf},
	journal = {bioRxiv}
}

@article{Xiong2024-nq,
  title        = {{DeSide}: A unified deep learning approach for cellular
                  deconvolution of tumor microenvironment},
  author       = {Xiong, Xin* and Liu, Yerong* and Pu, Dandan and Yang, Zhu and
                  Bi, Zedong and Tian, Liang# and Li, Xuefei#},
  journaltitle = {Proc. Natl. Acad. Sci. U. S. A.},
  volume       = {121},
  issue        = {46},
  pages        = {e2407096121},
  date         = {2024},
  doi          = {10.1073/pnas.2407096121},
  URL          = {https://www.pnas.org/doi/10.1073/pnas.2407096121}
}

```
