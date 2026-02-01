import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix
from tqdm import tqdm
from typing import Union
import math
import multiprocessing
import sys
from joblib import Parallel, delayed
import gc
from typing import Dict, List, Optional
from dataclasses import dataclass
from scipy.stats import rankdata
from scipy.stats import zscore as czscore
from joblib import Parallel, delayed
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
    
def rightsingularsvdvectorgset(gset_idx, Z, verbose=False, progress_bar=None):
    """
    计算基因集的右奇异向量
    
    Parameters:
    -----------
    gset_idx : array-like
        基因集的索引
    Z : scipy.sparse.csc_matrix or numpy.ndarray
        输入矩阵
    verbose : bool
        是否显示进度
    progress_bar : tqdm
        进度条对象
    
    Returns:
    --------
    numpy.ndarray
        第一个右奇异向量
    """
    # 判断是否是稀疏矩阵
    if sparse.issparse(Z):
        # 对稀疏矩阵使用svds
        s = sparse.linalg.svds(Z[gset_idx], k=1)
        v = s[2].T  # 获取右奇异向量
    else:
        # 对密集矩阵使用普通svd
        s = np.linalg.svd(Z[gset_idx], full_matrices=False)
        v = s[2].T  # 获取右奇异向量
    
    # 更新进度条
    if verbose and progress_bar is not None:
        progress_bar.update(1)
    
    # 返回第一个右奇异向量
    return v[:, 0]
    
from scipy import sparse
from scipy.linalg import svd
import numpy as np

def safe_svd(x, nu, nv):
    if sparse.issparse(x):
        x = x.toarray()
    u, s, vt = svd(x, full_matrices=False)
    out = {'u': u, 'd': s, 'v': vt.T}
    if not nu:
        out['u'] = np.zeros((x.shape[0], 0))
    if not nv:
        out['v'] = np.zeros((x.shape[1], 0))
    return out

def svd_via_crossprod(x, k, nu=None, nv=None):
    if nu is None:
        nu = k
    if nv is None:
        nv = k
        
    if x.shape[0] > x.shape[1]:
        y = x.T @ x if sparse.issparse(x) else np.dot(x.T, x)
        res = safe_svd(y, nu=0, nv=max(nu, nv, k))
        res['d'] = np.sqrt(res['d'])
        u0 = x @ res['v'][:, :nu]
        res['u'] = u0 / res['d'][:nu]
        res['v'] = res['v'][:, :nv]
    else:
        y = x @ x.T if sparse.issparse(x) else np.dot(x, x.T)
        res = safe_svd(y, nu=max(nu, nv, k), nv=0)
        res['d'] = np.sqrt(res['d'])
        v0 = (x.T @ res['u'][:, :nv]) if sparse.issparse(x) else np.dot(x.T, res['u'][:, :nv])
        res['v'] = v0 / res['d'][:nv]
        res['u'] = res['u'][:, :nu]
    
    res['d'] = res['d'][:k]
    return res

def run_exact_svd(x, k=None, nu=None, nv=None):
    if k is None:
        k = min(x.shape)
    if nu is None:
        nu = k
    if nv is None:
        nv = k
    
    if x.shape[0] > x.shape[1]:
        res = svd_via_crossprod(x, k=k, nu=nu, nv=nv)
    else:
        res = safe_svd(x, nu=nu, nv=nv)
        res['d'] = res['d'][:k]
    return res

def right_singular_svd_vector(gset_idx, Z, progress_bar=None):
    
    gset_idx = np.array(gset_idx) - 1
    subset = Z[gset_idx, :]
    if sparse.issparse(Z):
        s = run_exact_svd(subset)
    else:
        u, s, vt = svd(subset)
        s = {'v': vt.T}
    
    if progress_bar:
        progress_bar.update(1)
    return s['v'][:, 0]
      
def plage(expr_df, gene_sets, min_size=1, max_size=np.inf, 
         remove_constant=True, remove_nz_constant=True,
         n_jobs=1, use_sparse=False, verbose=True):
    """
    Calculate pathway activities using PLAGE method (following R implementation)
    
    Parameters:
    -----------
    expr_df : pandas.DataFrame or scipy.sparse matrix
        Gene expression data
    gene_sets : dict
        Dictionary of pathway names to gene lists
    min_size : int, default=1
        Minimum size for gene sets
    max_size : float, default=inf
        Maximum size for gene sets
    remove_constant : bool
        Whether to remove constant genes
    remove_nz_constant : bool
        Whether to remove constant non-zero genes
    n_jobs : int
        Number of parallel jobs
    use_sparse : bool
        Whether to use sparse matrix operations
    verbose : bool
        Whether to show progress
    """
    # Filter and prepare data
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
    batch_size = max(1, len(gene_sets) // (n_jobs * 4))
    gene_sets_items = list(gene_sets.items())
    # Center and scale the data
    if verbose:
        print("Centering and scaling values")
        
    if use_sparse:
        Z = sparse_apply_scale(csc_matrix(expr_t))
    else:
        Z = expr_t.apply(czscore, axis=1, ddof=0)
        Z = Z.values
    if n_jobs > 1:
        es = Parallel(n_jobs=n_jobs,batch_size=batch_size)(
                delayed(right_singular_svd_vector)(genes, Z, None)
            for pathway_name, genes in gene_sets_items
        ) 
    else:
        es = [right_singular_svd_vector(genes, Z, progress_bar=None)
            for pathway_name, genes in gene_sets_items] 
            
    gene_sets = convert_indices_to_genes(filtered_result['filtered_mapped_gene_sets'], 
                                       expr_t.index)
    # Create results DataFrame
    result_df = pd.DataFrame(es, 
                           index=list(gene_sets.keys()),
                           columns=expr_df.columns)
    
    # Handle single gene set case
    if len(gene_sets) == 1:
        result_df = pd.DataFrame(result_df.values.reshape(1, -1),
                               index=[list(gene_sets.keys())[0]],
                               columns=expr_df.columns)
    
    return result_df