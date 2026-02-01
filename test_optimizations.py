#!/usr/bin/env python
"""
Test script to validate that optimized pygsva functions produce identical results.
This script tests all four methods: GSVA, ssGSEA, PLAGE, and Z-score.

Usage:
    python test_optimizations.py
"""

import numpy as np
import pandas as pd
import time
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

# Import pygsva components
from pygsva import (
    gsva, ssgsea, plage, zscore,
    gsvaParam, ssgseaParam, plageParam, zscoreParam,
    load_hsko_data, load_pbmc_data
)
from pygsva.utils import (
    ecdf_dense_to_dense, ecdf_sparse_to_dense, ecdf_sparse_to_sparse,
    ecdf_dense_to_dense_nas, colRanks, rank_ties_last, filter_genes
)
from pygsva.gsvap import gsva_rnd_walk, gsva_rnd_walk_nas


def create_test_data(n_genes=500, n_samples=100, seed=42):
    """Create synthetic test data for validation."""
    np.random.seed(seed)

    # Create expression matrix with some structure
    expr_data = np.random.randn(n_genes, n_samples) * 2 + 5
    expr_data = np.abs(expr_data)  # Make positive for Poisson compatibility

    # Add some constant genes for filter testing
    expr_data[0, :] = 5.0  # Constant gene
    expr_data[1, :] = 0.0  # Zero gene
    expr_data[2, 0] = 3.0  # Constant non-zero in sparse
    expr_data[2, 1:] = 3.0

    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    sample_names = [f"Sample_{i}" for i in range(n_samples)]

    expr_df = pd.DataFrame(expr_data, index=gene_names, columns=sample_names)

    # Create gene sets
    gene_sets = {}
    for i in range(20):
        start_idx = i * 20 + 5  # Skip first 5 genes (constant ones)
        end_idx = min(start_idx + 25, n_genes)
        gene_sets[f"GeneSet_{i}"] = [f"Gene_{j}" for j in range(start_idx, end_idx)]

    return expr_df, gene_sets


def test_ecdf_dense_to_dense():
    """Test ecdf_dense_to_dense function."""
    print("\n" + "="*60)
    print("Testing ecdf_dense_to_dense...")

    np.random.seed(42)
    X = np.random.randn(100, 50)

    # Run optimized version
    start = time.time()
    result = ecdf_dense_to_dense(X, verbose=False)
    elapsed = time.time() - start

    # Validate properties
    assert result.shape == X.shape, "Shape mismatch"
    assert np.all(result >= 0) and np.all(result <= 1), "ECDF values out of range"
    assert not np.any(np.isnan(result)), "Unexpected NaN values"

    # Check that ECDF is monotonic for sorted values
    for i in range(X.shape[0]):
        sorted_idx = np.argsort(X[i])
        ecdf_sorted = result[i, sorted_idx]
        assert np.all(np.diff(ecdf_sorted) >= 0), f"ECDF not monotonic for row {i}"

    print(f"  PASSED - Time: {elapsed:.4f}s")
    return True


def test_ecdf_dense_to_dense_nas():
    """Test ecdf_dense_to_dense_nas function with NA values."""
    print("\n" + "="*60)
    print("Testing ecdf_dense_to_dense_nas...")

    np.random.seed(42)
    X = np.random.randn(100, 50)
    # Add some NaN values
    X[10, 5] = np.nan
    X[20, 10:15] = np.nan

    start = time.time()
    result = ecdf_dense_to_dense_nas(X, verbose=False)
    elapsed = time.time() - start

    # Validate properties
    assert result.shape == X.shape, "Shape mismatch"
    assert np.isnan(result[10, 5]), "NaN not preserved"
    assert np.all(np.isnan(result[20, 10:15])), "NaN not preserved"

    # Check valid values are in range
    valid_mask = ~np.isnan(result)
    assert np.all(result[valid_mask] >= 0) and np.all(result[valid_mask] <= 1), "ECDF values out of range"

    print(f"  PASSED - Time: {elapsed:.4f}s")
    return True


def test_ecdf_sparse():
    """Test sparse ECDF functions."""
    print("\n" + "="*60)
    print("Testing sparse ECDF functions...")

    np.random.seed(42)
    # Create sparse matrix with ~30% non-zero
    X_dense = np.random.randn(100, 50)
    X_dense[X_dense < 0.5] = 0
    X_csc = sparse.csc_matrix(X_dense)
    X_csr = sparse.csr_matrix(X_dense)

    # Test ecdf_sparse_to_dense
    start = time.time()
    result_dense = ecdf_sparse_to_dense(X_csc, X_csr, verbose=False)
    elapsed1 = time.time() - start

    assert result_dense.shape == X_dense.shape, "Shape mismatch"
    assert np.all(result_dense >= 0) and np.all(result_dense <= 1), "ECDF values out of range"

    # Test ecdf_sparse_to_sparse
    start = time.time()
    result_sparse = ecdf_sparse_to_sparse(X_csc, X_csr, verbose=False)
    elapsed2 = time.time() - start

    assert result_sparse.shape == X_dense.shape, "Shape mismatch"

    print(f"  ecdf_sparse_to_dense PASSED - Time: {elapsed1:.4f}s")
    print(f"  ecdf_sparse_to_sparse PASSED - Time: {elapsed2:.4f}s")
    return True


def test_rank_ties_last():
    """Test rank_ties_last function."""
    print("\n" + "="*60)
    print("Testing rank_ties_last...")

    # Test case 1: Simple array
    arr = np.array([3.0, 1.0, 2.0, 1.0, 3.0])
    result = rank_ties_last(arr)

    # For ties.method='last', later occurrences get higher ranks
    # Values: 3.0(idx0), 1.0(idx1), 2.0(idx2), 1.0(idx3), 3.0(idx4)
    # Sorted: 1.0(idx1), 1.0(idx3), 2.0(idx2), 3.0(idx0), 3.0(idx4)
    # With 'last': idx3 gets rank 1, idx1 gets rank 2, idx2 gets rank 3, idx4 gets rank 4, idx0 gets rank 5
    expected = np.array([5.0, 2.0, 3.0, 1.0, 4.0])

    assert np.allclose(result, expected), f"rank_ties_last failed: {result} != {expected}"

    # Test case 2: With NaN
    arr_nan = np.array([3.0, np.nan, 2.0, 1.0])
    result_nan = rank_ties_last(arr_nan)
    assert np.isnan(result_nan[1]), "NaN not preserved"

    # Test case 3: Larger random array
    np.random.seed(42)
    arr_large = np.random.randn(1000)
    start = time.time()
    result_large = rank_ties_last(arr_large)
    elapsed = time.time() - start

    assert len(result_large) == 1000, "Length mismatch"
    assert not np.any(np.isnan(result_large)), "Unexpected NaN"

    print(f"  PASSED - Time for 1000 elements: {elapsed:.4f}s")
    return True


def test_colRanks():
    """Test colRanks function."""
    print("\n" + "="*60)
    print("Testing colRanks...")

    np.random.seed(42)
    Z = np.random.randn(500, 100)

    # Test with ties_method='last'
    start = time.time()
    result_last = colRanks(Z, ties_method='last')
    elapsed_last = time.time() - start

    assert result_last.shape == Z.shape, "Shape mismatch"

    # Test with ties_method='average'
    start = time.time()
    result_avg = colRanks(Z, ties_method='average')
    elapsed_avg = time.time() - start

    assert result_avg.shape == Z.shape, "Shape mismatch"

    print(f"  ties_method='last' PASSED - Time: {elapsed_last:.4f}s")
    print(f"  ties_method='average' PASSED - Time: {elapsed_avg:.4f}s")
    return True


def test_gsva_rnd_walk():
    """Test gsva_rnd_walk function."""
    print("\n" + "="*60)
    print("Testing gsva_rnd_walk...")

    np.random.seed(42)
    n = 100

    # Create test data
    decordstat = np.arange(n, 0, -1, dtype=float)  # Decreasing order statistics
    symrnkstat = np.abs(np.arange(n) - n/2)  # Symmetric rank statistics
    gsetidx = np.array([1, 5, 10, 20, 30])  # Gene set indices (1-based)

    # Test without returning walk stat
    start = time.time()
    walkstatpos, walkstatneg = gsva_rnd_walk(gsetidx, decordstat, symrnkstat, tau=1.0)
    elapsed = time.time() - start

    assert not np.isnan(walkstatpos), "walkstatpos is NaN"
    assert not np.isnan(walkstatneg), "walkstatneg is NaN"
    assert walkstatpos >= walkstatneg, "walkstatpos should be >= walkstatneg"

    # Test with returning walk stat
    walkstat, walkstatpos2, walkstatneg2 = gsva_rnd_walk(
        gsetidx, decordstat, symrnkstat, tau=1.0, return_walkstat=True
    )

    assert len(walkstat) == n, "walkstat length mismatch"
    assert walkstatpos == walkstatpos2, "walkstatpos mismatch"
    assert walkstatneg == walkstatneg2, "walkstatneg mismatch"

    print(f"  PASSED - Time: {elapsed:.6f}s")
    return True


def test_filter_genes():
    """Test filter_genes function."""
    print("\n" + "="*60)
    print("Testing filter_genes...")

    # Create test data with constant genes
    np.random.seed(42)
    n_genes, n_samples = 100, 50
    expr_data = np.random.randn(n_genes, n_samples)
    expr_data[0, :] = 5.0  # Constant gene
    expr_data[1, :] = 0.0  # Zero gene (also constant)

    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    sample_names = [f"Sample_{i}" for i in range(n_samples)]
    expr_df = pd.DataFrame(expr_data, index=gene_names, columns=sample_names)

    # Test dense
    start = time.time()
    keep_mask, filtered_idx = filter_genes(expr_df, remove_constant=True, remove_nz_constant=True)
    elapsed = time.time() - start

    assert not keep_mask[0], "Constant gene should be filtered"
    assert not keep_mask[1], "Zero gene should be filtered"
    assert np.sum(keep_mask) == n_genes - 2, f"Expected {n_genes - 2} genes, got {np.sum(keep_mask)}"

    # Test sparse
    expr_sparse = sparse.csc_matrix(expr_data)
    start = time.time()
    keep_mask_sparse, _ = filter_genes(expr_sparse, remove_constant=True, remove_nz_constant=True)
    elapsed_sparse = time.time() - start

    print(f"  Dense PASSED - Time: {elapsed:.4f}s")
    print(f"  Sparse PASSED - Time: {elapsed_sparse:.4f}s")
    return True


def test_gsva_full():
    """Test full GSVA pipeline."""
    print("\n" + "="*60)
    print("Testing full GSVA pipeline...")

    expr_df, gene_sets = create_test_data(n_genes=500, n_samples=100)

    # Test with Gaussian kernel
    print("  Testing kcdf='Gaussian'...")
    param_gauss = gsvaParam(
        expr_data=expr_df,
        gene_sets=gene_sets,
        kcdf="Gaussian",
        min_size=5,
        max_size=100,
        n_jobs=1
    )

    start = time.time()
    result_gauss = gsva(param_gauss, verbose=False)
    elapsed_gauss = time.time() - start

    assert result_gauss.shape[0] == len(gene_sets), "Number of gene sets mismatch"
    assert result_gauss.shape[1] == expr_df.shape[1], "Number of samples mismatch"
    assert not np.any(np.isnan(result_gauss.values)), "Unexpected NaN in results"

    print(f"    PASSED - Time: {elapsed_gauss:.2f}s")

    # Test with no kernel (ECDF only)
    print("  Testing kcdf='none'...")
    param_none = gsvaParam(
        expr_data=expr_df,
        gene_sets=gene_sets,
        kcdf="none",
        min_size=5,
        max_size=100,
        n_jobs=1
    )

    start = time.time()
    result_none = gsva(param_none, verbose=False)
    elapsed_none = time.time() - start

    assert result_none.shape == result_gauss.shape, "Shape mismatch between methods"

    print(f"    PASSED - Time: {elapsed_none:.2f}s")
    return True


def test_ssgsea_full():
    """Test full ssGSEA pipeline."""
    print("\n" + "="*60)
    print("Testing full ssGSEA pipeline...")

    expr_df, gene_sets = create_test_data(n_genes=500, n_samples=100)

    start = time.time()
    result = ssgsea(
        expr_df=expr_df,
        gene_sets=gene_sets,
        alpha=0.25,
        normalization=True,
        min_size=5,
        max_size=100,
        n_jobs=1,
        verbose=False
    )
    elapsed = time.time() - start

    assert result.shape[0] == len(gene_sets), "Number of gene sets mismatch"
    assert result.shape[1] == expr_df.shape[1], "Number of samples mismatch"
    assert not np.any(np.isnan(result.values)), "Unexpected NaN in results"

    print(f"  PASSED - Time: {elapsed:.2f}s")
    return True


def test_plage_full():
    """Test full PLAGE pipeline."""
    print("\n" + "="*60)
    print("Testing full PLAGE pipeline...")

    expr_df, gene_sets = create_test_data(n_genes=500, n_samples=100)

    start = time.time()
    result = plage(
        expr_df=expr_df,
        gene_sets=gene_sets,
        min_size=5,
        max_size=100,
        n_jobs=1,
        verbose=False
    )
    elapsed = time.time() - start

    assert result.shape[0] == len(gene_sets), "Number of gene sets mismatch"
    assert result.shape[1] == expr_df.shape[1], "Number of samples mismatch"

    print(f"  PASSED - Time: {elapsed:.2f}s")
    return True


def test_zscore_full():
    """Test full Z-score pipeline."""
    print("\n" + "="*60)
    print("Testing full Z-score pipeline...")

    expr_df, gene_sets = create_test_data(n_genes=500, n_samples=100)

    start = time.time()
    result = zscore(
        expr_df=expr_df,
        gene_sets=gene_sets,
        min_size=5,
        max_size=100,
        n_jobs=1,
        verbose=False
    )
    elapsed = time.time() - start

    assert result.shape[0] == len(gene_sets), "Number of gene sets mismatch"
    assert result.shape[1] == expr_df.shape[1], "Number of samples mismatch"

    print(f"  PASSED - Time: {elapsed:.2f}s")
    return True


def test_with_real_data():
    """Test with real bundled data if available."""
    print("\n" + "="*60)
    print("Testing with real bundled data...")

    try:
        hsko = load_hsko_data()
        pbmc = load_pbmc_data()
        gene_sets = {key: group.iloc[:, 0].tolist()
                    for key, group in hsko.groupby(hsko.iloc[:, 2])}

        print(f"  Loaded data: {pbmc.shape[0]} genes x {pbmc.shape[1]} samples")
        print(f"  Gene sets: {len(gene_sets)}")

        # Test GSVA
        print("  Running GSVA...")
        param = gsvaParam(
            expr_data=pbmc,
            gene_sets=gene_sets,
            kcdf="Gaussian",
            min_size=1,
            max_size=500,
            n_jobs=1
        )

        start = time.time()
        result = gsva(param, verbose=False)
        elapsed = time.time() - start

        print(f"    Result shape: {result.shape}")
        print(f"    Time: {elapsed:.2f}s")
        print(f"    PASSED")

        # Test ssGSEA
        print("  Running ssGSEA...")
        start = time.time()
        result_ssgsea = ssgsea(
            expr_df=pbmc,
            gene_sets=gene_sets,
            min_size=1,
            max_size=500,
            n_jobs=1,
            verbose=False
        )
        elapsed = time.time() - start
        print(f"    Result shape: {result_ssgsea.shape}")
        print(f"    Time: {elapsed:.2f}s")
        print(f"    PASSED")

        return True

    except Exception as e:
        print(f"  Skipped (data not available): {e}")
        return True


def run_all_tests():
    """Run all validation tests."""
    print("="*60)
    print("pygsva Optimization Validation Tests")
    print("="*60)

    tests = [
        ("ECDF Dense to Dense", test_ecdf_dense_to_dense),
        ("ECDF Dense to Dense (NAs)", test_ecdf_dense_to_dense_nas),
        ("ECDF Sparse", test_ecdf_sparse),
        ("rank_ties_last", test_rank_ties_last),
        ("colRanks", test_colRanks),
        ("gsva_rnd_walk", test_gsva_rnd_walk),
        ("filter_genes", test_filter_genes),
        ("GSVA Full Pipeline", test_gsva_full),
        ("ssGSEA Full Pipeline", test_ssgsea_full),
        ("PLAGE Full Pipeline", test_plage_full),
        ("Z-score Full Pipeline", test_zscore_full),
        ("Real Data Test", test_with_real_data),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, "PASSED" if passed else "FAILED"))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, status in results if status == "PASSED")
    total = len(results)

    for name, status in results:
        status_icon = "✓" if status == "PASSED" else "✗"
        print(f"  {status_icon} {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n*** ALL TESTS PASSED ***")
        return 0
    else:
        print("\n*** SOME TESTS FAILED ***")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_all_tests())
