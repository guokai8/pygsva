#!/usr/bin/env python
"""
Benchmark script to measure performance improvements in pygsva.

Usage:
    python benchmark_optimizations.py
"""

import numpy as np
import pandas as pd
import time
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

from pygsva import (
    gsva, ssgsea, plage, zscore,
    gsvaParam, ssgseaParam, plageParam, zscoreParam,
    load_hsko_data, load_pbmc_data
)


def create_benchmark_data(n_genes=5000, n_samples=500, n_gene_sets=100, seed=42):
    """Create larger benchmark data."""
    np.random.seed(seed)

    # Create expression matrix
    expr_data = np.random.randn(n_genes, n_samples) * 2 + 10
    expr_data = np.abs(expr_data)

    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    sample_names = [f"Sample_{i}" for i in range(n_samples)]

    expr_df = pd.DataFrame(expr_data, index=gene_names, columns=sample_names)

    # Create gene sets of varying sizes
    gene_sets = {}
    for i in range(n_gene_sets):
        size = np.random.randint(20, 100)
        genes = np.random.choice(gene_names, size=size, replace=False)
        gene_sets[f"GeneSet_{i}"] = list(genes)

    return expr_df, gene_sets


def benchmark_gsva(expr_df, gene_sets, kcdf="Gaussian", n_runs=3):
    """Benchmark GSVA method."""
    times = []

    for i in range(n_runs):
        param = gsvaParam(
            expr_data=expr_df,
            gene_sets=gene_sets,
            kcdf=kcdf,
            min_size=5,
            max_size=500,
            n_jobs=1
        )

        start = time.time()
        result = gsva(param, verbose=False)
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        'method': f'GSVA (kcdf={kcdf})',
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'result_shape': result.shape
    }


def benchmark_ssgsea(expr_df, gene_sets, n_runs=3):
    """Benchmark ssGSEA method."""
    times = []

    for i in range(n_runs):
        start = time.time()
        result = ssgsea(
            expr_df=expr_df,
            gene_sets=gene_sets,
            min_size=5,
            max_size=500,
            n_jobs=1,
            verbose=False
        )
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        'method': 'ssGSEA',
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'result_shape': result.shape
    }


def benchmark_plage(expr_df, gene_sets, n_runs=3):
    """Benchmark PLAGE method."""
    times = []

    for i in range(n_runs):
        start = time.time()
        result = plage(
            expr_df=expr_df,
            gene_sets=gene_sets,
            min_size=5,
            max_size=500,
            n_jobs=1,
            verbose=False
        )
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        'method': 'PLAGE',
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'result_shape': result.shape
    }


def benchmark_zscore(expr_df, gene_sets, n_runs=3):
    """Benchmark Z-score method."""
    times = []

    for i in range(n_runs):
        start = time.time()
        result = zscore(
            expr_df=expr_df,
            gene_sets=gene_sets,
            min_size=5,
            max_size=500,
            n_jobs=1,
            verbose=False
        )
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        'method': 'Z-score',
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'result_shape': result.shape
    }


def run_benchmarks():
    """Run all benchmarks."""
    print("="*70)
    print("pygsva Performance Benchmark")
    print("="*70)

    # Small dataset benchmark
    print("\n" + "-"*70)
    print("SMALL DATASET: 1000 genes x 100 samples, 50 gene sets")
    print("-"*70)

    expr_small, gs_small = create_benchmark_data(
        n_genes=1000, n_samples=100, n_gene_sets=50
    )

    results_small = []

    print("\nRunning benchmarks (3 runs each)...")

    result = benchmark_gsva(expr_small, gs_small, kcdf="Gaussian")
    results_small.append(result)
    print(f"  {result['method']}: {result['mean_time']:.2f}s ± {result['std_time']:.2f}s")

    result = benchmark_gsva(expr_small, gs_small, kcdf="none")
    results_small.append(result)
    print(f"  {result['method']}: {result['mean_time']:.2f}s ± {result['std_time']:.2f}s")

    result = benchmark_ssgsea(expr_small, gs_small)
    results_small.append(result)
    print(f"  {result['method']}: {result['mean_time']:.2f}s ± {result['std_time']:.2f}s")

    result = benchmark_plage(expr_small, gs_small)
    results_small.append(result)
    print(f"  {result['method']}: {result['mean_time']:.2f}s ± {result['std_time']:.2f}s")

    result = benchmark_zscore(expr_small, gs_small)
    results_small.append(result)
    print(f"  {result['method']}: {result['mean_time']:.2f}s ± {result['std_time']:.2f}s")

    # Medium dataset benchmark
    print("\n" + "-"*70)
    print("MEDIUM DATASET: 5000 genes x 200 samples, 100 gene sets")
    print("-"*70)

    expr_medium, gs_medium = create_benchmark_data(
        n_genes=5000, n_samples=200, n_gene_sets=100
    )

    results_medium = []

    print("\nRunning benchmarks (3 runs each)...")

    result = benchmark_gsva(expr_medium, gs_medium, kcdf="Gaussian")
    results_medium.append(result)
    print(f"  {result['method']}: {result['mean_time']:.2f}s ± {result['std_time']:.2f}s")

    result = benchmark_gsva(expr_medium, gs_medium, kcdf="none")
    results_medium.append(result)
    print(f"  {result['method']}: {result['mean_time']:.2f}s ± {result['std_time']:.2f}s")

    result = benchmark_ssgsea(expr_medium, gs_medium)
    results_medium.append(result)
    print(f"  {result['method']}: {result['mean_time']:.2f}s ± {result['std_time']:.2f}s")

    result = benchmark_plage(expr_medium, gs_medium)
    results_medium.append(result)
    print(f"  {result['method']}: {result['mean_time']:.2f}s ± {result['std_time']:.2f}s")

    result = benchmark_zscore(expr_medium, gs_medium)
    results_medium.append(result)
    print(f"  {result['method']}: {result['mean_time']:.2f}s ± {result['std_time']:.2f}s")

    # Large dataset benchmark (optional - may take longer)
    print("\n" + "-"*70)
    print("LARGE DATASET: 10000 genes x 500 samples, 200 gene sets")
    print("-"*70)

    expr_large, gs_large = create_benchmark_data(
        n_genes=10000, n_samples=500, n_gene_sets=200
    )

    results_large = []

    print("\nRunning benchmarks (2 runs each for large dataset)...")

    result = benchmark_gsva(expr_large, gs_large, kcdf="Gaussian", n_runs=2)
    results_large.append(result)
    print(f"  {result['method']}: {result['mean_time']:.2f}s ± {result['std_time']:.2f}s")

    result = benchmark_gsva(expr_large, gs_large, kcdf="none", n_runs=2)
    results_large.append(result)
    print(f"  {result['method']}: {result['mean_time']:.2f}s ± {result['std_time']:.2f}s")

    result = benchmark_ssgsea(expr_large, gs_large, n_runs=2)
    results_large.append(result)
    print(f"  {result['method']}: {result['mean_time']:.2f}s ± {result['std_time']:.2f}s")

    result = benchmark_plage(expr_large, gs_large, n_runs=2)
    results_large.append(result)
    print(f"  {result['method']}: {result['mean_time']:.2f}s ± {result['std_time']:.2f}s")

    result = benchmark_zscore(expr_large, gs_large, n_runs=2)
    results_large.append(result)
    print(f"  {result['method']}: {result['mean_time']:.2f}s ± {result['std_time']:.2f}s")

    # Summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    print("\n{:<25} {:>12} {:>12} {:>12}".format(
        "Method", "Small", "Medium", "Large"
    ))
    print("-"*70)

    methods = ['GSVA (kcdf=Gaussian)', 'GSVA (kcdf=none)', 'ssGSEA', 'PLAGE', 'Z-score']
    for i, method in enumerate(methods):
        small_time = results_small[i]['mean_time']
        medium_time = results_medium[i]['mean_time']
        large_time = results_large[i]['mean_time']
        print("{:<25} {:>10.2f}s {:>10.2f}s {:>10.2f}s".format(
            method, small_time, medium_time, large_time
        ))

    print("\n" + "="*70)
    print("Benchmark complete!")
    print("="*70)


if __name__ == "__main__":
    run_benchmarks()
