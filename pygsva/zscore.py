import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix
from tqdm import tqdm
from typing import Union
import math
import multiprocessing
import sys
import warnings
import gc
from typing import Dict, List, Optional
from dataclasses import dataclass
from scipy.stats import rankdata
from .utils import map_gene_sets_to_features, filter_gene_sets, filter_and_map_gene_sets_s, filter_and_map_genes_and_gene_sets_s, convert_indices_to_genes
from scipy import sparse
import numpy as np

def sparse_column_standardize(matrix):
    """
    Highly optimized sparse matrix standardization.
    Vectorized version using numpy reduceat operations.
    """
    if not sparse.isspmatrix_csc(matrix):
        matrix = matrix.tocsc()
    else:
        matrix = matrix.copy()

    n_cols = matrix.shape[1]
    col_lengths = np.diff(matrix.indptr)

    # Vectorized computation of means and stds using reduceat
    # Only process columns that have non-zero elements
    non_empty_mask = col_lengths > 0
    non_empty_indices = np.where(non_empty_mask)[0]

    means = np.zeros(n_cols)
    stds = np.zeros(n_cols)

    if len(non_empty_indices) > 0:
        # Get the start indices for reduceat (only for non-empty columns)
        reduce_indices = matrix.indptr[non_empty_indices]

        # Compute sums for each column
        col_sums = np.add.reduceat(matrix.data, reduce_indices)

        # Handle the last column edge case
        if len(reduce_indices) > 1:
            # For columns that aren't the last non-empty one
            col_sums_corrected = col_sums.copy()
        else:
            col_sums_corrected = col_sums

        # Compute means
        means[non_empty_indices] = col_sums_corrected / col_lengths[non_empty_indices]

        # Compute variance using two-pass algorithm for numerical stability
        for i, col_idx in enumerate(non_empty_indices):
            start, end = matrix.indptr[col_idx:col_idx+2]
            if end > start:
                col_data = matrix.data[start:end]
                n = len(col_data)
                if n > 1:
                    stds[col_idx] = np.std(col_data, ddof=1)

    # Vectorized standardization
    col_indices = np.repeat(np.arange(n_cols), col_lengths)
    valid_std_mask = stds[col_indices] != 0
    matrix.data = np.where(valid_std_mask,
                          (matrix.data - means[col_indices]) / stds[col_indices],
                          0)

    return matrix

def sparse_apply_scale(X):
    """Apply scaling with transposition"""
    return sparse_column_standardize(X.transpose()).transpose()

def process_pathway(args):
    expr_z, pathway_name, genes = args
    genes_in_data = [gene for gene in genes if gene in expr_z.index]
    if not genes_in_data:
        return pathway_name, None
    expr_subset = expr_z.loc[genes_in_data]
    z_scores = expr_subset.sum(axis=0) / np.sqrt(len(genes_in_data))
    return pathway_name, z_scores.values
    
def zscore(expr_df, gene_sets, min_size=1, max_size=np.inf, remove_constant=True, 
           remove_nz_constant=True, n_jobs=1, use_sparse=False, verbose=True):
    filtered_result = filter_and_map_genes_and_gene_sets_s(
        expr_data=expr_df,
        gene_sets=gene_sets,
        min_size=min_size,
        max_size=max_size,
        remove_constant=remove_constant,
        remove_nz_constant=remove_nz_constant,
        use_sparse=use_sparse,
        verbose=verbose
    )
    
    expr_t = filtered_result['filtered_data_matrix']
    gene_sets = filtered_result['filtered_mapped_gene_sets']
    gene_sets = convert_indices_to_genes(filtered_result['filtered_mapped_gene_sets'],expr_t.index)
    def standardize_series(series):
        mean = series.mean()
        std = series.std(ddof=1)  # 设置 ddof=1，与 R 的 scale 函数一致
        return (series - mean) / std
    if verbose:
        print("Centering and scaling values")
    if use_sparse:
        Z = sparse_apply_scale(csc_matrix(expr_t))
        Z = pd.DataFrame(Z.toarray(),index=expr_t.index,
                           columns=expr_t.columns)
    else:
        Z = expr_t.apply(standardize_series, axis=1)
        
    process_args = [(Z, pathway_name, genes) 
                   for pathway_name, genes in gene_sets.items()]
    
    batch_size = max(1, min(100, len(gene_sets) // (n_jobs * 4)))
    
    if n_jobs == 1:
        if verbose:
            from tqdm import tqdm
            results = [process_pathway(args) for args in tqdm(process_args, desc="Calculating Z-scores")]
        else:
            results = [process_pathway(args) for args in process_args]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
            delayed(process_pathway)(args) for args in process_args
        )
    
    pathway_activity = {name: scores for name, scores in results if scores is not None}
    return pd.DataFrame(pathway_activity, index=expr_df.columns).T