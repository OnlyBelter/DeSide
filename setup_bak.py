import io
import os
import sys
from setuptools import setup, find_packages
# from deside._version import __version__

install_requires = ["matplotlib", "pandas==1.2.5", "scikit-learn==0.24.0",
                    "seaborn>=0.11.2", "requests", "tensorflow>=2.8.0", "scipy", "numpy<1.22",
                    "anndata>=0.8.0", "scanpy==1.8.0", "joblib", "leidenalg", "jupyterlab",
                    "umap-learn==0.5.1", "bbknn==1.5.1", "openpyxl", "tensorboard",
                    "build", "sphinx==3.5.4", "myst-parser>=0.13.6", "fsspec",
                    "h5py", "sphinx_book_theme", "tables>=3.6.1", "statsmodels", "matplotlib-venn",
                    "SciencePlots"]
if sys.version_info < (3, 4, 0):
    install_requires.append("enum34")


# def get_version():
#     with open(os.path.join("deside", "_version.txt"), 'r') as f:
#         return f.readline().strip()


def get_test_data():
    for p, _, fs in os.walk(os.path.join("deside", "tests", "references")):
        p = p.split(os.sep)[2:]

        for f in fs:
            yield os.path.join(*(p + [f]))


# README_rst = ""
fndoc = os.path.join(os.path.dirname(__file__), "README.rst")
with io.open(fndoc, mode="r", encoding="utf-8") as fd:
    README_rst = fd.read()

setup(
    name="deside",
    version='1.0.0',
    description="DEep-learning and SIngle-cell based DEconvolution",
    long_description=README_rst,
    license="BSD-3-Clause",
    author="Xin (Belter) Xiong",
    author_email="onlybelter@outlook.com",
    url="",
    platforms=["any"],
    keywords="cancer biology, bioinformatics",
    packages=find_packages(),
    package_data={
        "emtdecode": ["demo_data/*.txt", "_version.txt"],
        "emtdecode.tests": list(get_test_data()),
    },
    install_requires=install_requires,
    tests_require=["nose==1.*", "PyYaml>=4.2b1"],
    extras_require={"full": ["py", "tqdm"]},
)
