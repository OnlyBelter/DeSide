from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from deside.utility.read_file import ReadExp, ReadH5AD


def test_read_h5ad_get_df_supports_sparse_x(tmp_path):
    x = csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    obs = pd.DataFrame(index=["sample_1", "sample_2"])
    var = pd.DataFrame(index=["gene_1", "gene_2"])
    h5ad_file = tmp_path / "sparse_test.h5ad"
    AnnData(X=x, obs=obs, var=var).write_h5ad(h5ad_file)

    df = ReadH5AD(str(h5ad_file)).get_df()

    assert list(df.index) == ["sample_1", "sample_2"]
    assert list(df.columns) == ["gene_1", "gene_2"]
    assert np.allclose(df.values, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


def test_read_exp_align_with_gene_list_can_suppress_logging(capsys):
    exp = pd.DataFrame([[1.0, 2.0]], index=["sample_1"], columns=["gene_1", "gene_2"])

    read_exp = ReadExp(exp_file=exp, exp_type="TPM")
    read_exp.align_with_gene_list(gene_list=["gene_1"], log_info=False)

    captured = capsys.readouterr()

    assert captured.out == ""
