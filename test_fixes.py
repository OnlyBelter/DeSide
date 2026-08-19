"""Verify all three fixes for the log-stall investigation."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from deside.utility.read_file import ReadExp
from deside.decon_cf.torch_deside import DeSide


def test1_readexp_ndarray_wrap():
    arr = np.random.rand(5, 10).astype(np.float32)
    x_obj = ReadExp(arr, exp_type="log_space")
    assert isinstance(x_obj.exp, pd.DataFrame), "ReadExp did not wrap ndarray"
    print("TEST 1 PASS: ReadExp(ndarray) wraps to DataFrame")
    print("        columns sample:", x_obj.exp.columns[:3].tolist())
    print("        index sample:",  x_obj.exp.index[:3].tolist())


def test2_align_on_hardened():
    arr = np.random.rand(5, 10).astype(np.float32)
    x_obj = ReadExp(arr, exp_type="log_space")
    x_obj.align_with_gene_list(
        gene_list=x_obj.exp.columns.tolist(),
        fill_not_exist=True,
        pathway_list=True,
    )
    print("TEST 2 PASS: align_with_gene_list works on ndarray-backed ReadExp")


def test3_get_pathway_profiles_end_to_end():
    N, G, P = 20, 50, 15
    genes = [f"g{i}" for i in range(G)]
    samples = [f"s{i}" for i in range(N)]
    pathways = [f"pw{i}" for i in range(P)]
    x_df = pd.DataFrame(
        np.random.rand(N, G).astype(np.float32) * 100 + 1,
        index=samples,
        columns=genes,
    )
    pm = pd.DataFrame(
        np.random.randint(0, 2, size=(G, P)).astype(np.float32),
        index=genes,
        columns=pathways,
    )
    x_obj = ReadExp(x_df.copy(), exp_type="TPM")
    filtered_genes = genes[:40] + [f"g_missing{i}" for i in range(5)]
    result = DeSide._get_pathway_profiles(
        x_obj, pm, method="add_to_end", filtered_gene_list=filtered_genes
    )
    assert isinstance(result.exp, pd.DataFrame), "result.exp is not DataFrame"
    expected_cols = 45 + P
    assert result.exp.shape == (N, expected_cols), f"shape {result.exp.shape} != ({N},{expected_cols})"
    print(f"TEST 3 PASS: _get_pathway_profiles E2E, shape={result.exp.shape}")
    print("        first 5 columns:", result.exp.columns[:5].tolist())
    print("        last 5 columns (pw):", result.exp.columns[-5:].tolist())


def test4_convert_method_preserves_dataframe():
    N, G, P = 12, 40, 10
    genes = [f"g{i}" for i in range(G)]
    samples = [f"s{i}" for i in range(N)]
    pathways = [f"pw{i}" for i in range(P)]
    x_df = pd.DataFrame(
        np.random.rand(N, G).astype(np.float32) * 100 + 1,
        index=samples,
        columns=genes,
    )
    pm = pd.DataFrame(
        np.random.randint(0, 2, size=(G, P)).astype(np.float32),
        index=genes,
        columns=pathways,
    )
    x_obj = ReadExp(x_df.copy(), exp_type="TPM")
    result = DeSide._get_pathway_profiles(x_obj, pm, method="convert")
    assert isinstance(result.exp, pd.DataFrame)
    assert result.exp.shape == (N, P), f"convert shape {result.exp.shape}"
    print(f"TEST 4 PASS: convert method; result shape={result.exp.shape}")


def main():
    test1_readexp_ndarray_wrap()
    test2_align_on_hardened()
    test3_get_pathway_profiles_end_to_end()
    test4_convert_method_preserves_dataframe()
    print()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
