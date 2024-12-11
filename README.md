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
DeSide requires Python 3.8 or higher. It has been tested on Linux and MacOS, but should work on Windows as well.
- tensorflow>=2.11.1
- scikit-learn==0.24.2
- anndata>=0.8.0
- scanpy==1.8.0
- umap-learn==0.5.1
- pandas==1.5.3
- numpy>=1.22
- matplotlib
- seaborn>=0.11.2
- bbknn==1.5.1
- SciencePlots
- matplotlib<3.7

## Installation

pip should work out of the box:
```shell
# creating a virtual environment is recommended
conda create -n deside python=3.8
conda activate deside
# update pip
python3 -m pip install --upgrade pip
# install deside
pip install deside
```

## Usage Examples
Usage examples can be found: [DeSide_mini_example](https://github.com/OnlyBelter/DeSide_mini_example)

Three examples are provided:
- Using pre-trained model
- Training a model from scratch
- Generating a synthetic dataset

## Documentation
For all detailed documentation, please check https://deside.readthedocs.io/. The documentation will demonstrate the usage of DeSide from the following aspects:
- Installation in a virtual environment
- Usage examples
- Datasets used in DeSide
- Functions and classes in DeSide


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


