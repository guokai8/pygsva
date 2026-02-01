#PRECOMPUTE_RESOLUTION = 10000
#MAX_PRECOMPUTE = 10.0
#SIGMA_FACTOR = 4.0
#is_precomputed = False
#####
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.stats import norm, poisson, rankdata  # 添加 rankdata 的导入
from scipy.stats import poisson
from tqdm import tqdm
from typing import List, Union, Tuple
import math
import multiprocessing
import numpy as np
import pandas as pd
import sys
import warnings
from .config import sigma_factor, precompute_resolution,max_precompute,init_cdfs,get_config,update_config
precomputed_values =  init_cdfs()
is_precomputed = True
#from  import gsva_config
# 更新后的 precomputed_cdf 函数
def precomputed_cdf(x, sigma):
    if x is None or sigma is None or sigma < sys.float_info.epsilon:
        return None

    v = x / sigma
    if v < -max_precompute:
        return 0.0
    elif v > max_precompute:
        return 1.0
    else:
        idx = int(abs(v) / max_precompute * precompute_resolution)
        idx = max(min(idx, precompute_resolution), 0)  # Ensure valid index
        cdf = precomputed_values[idx]
        return 1.0 - cdf if v < 0 else cdf
        
def precomputed_cdf_vectorized(diff_matrix, bw):
    """Fully vectorized CDF computation using precomputed values"""
    scaled_x = np.abs(diff_matrix / bw)
    idx = np.clip((scaled_x * precompute_resolution / max_precompute).astype(int),
                 0, precompute_resolution - 1)

    # OPTIMIZATION: Use NumPy array indexing instead of nested loops
    precomputed_array = np.array(precomputed_values)
    result = precomputed_array[idx]

    # Handle negative values using boolean masking
    negative_mask = diff_matrix < 0
    result[negative_mask] = 1.0 - result[negative_mask]

    return result

def check_rownames(expr):
    if expr.index is None or expr.index.empty:
        raise ValueError("The input assay object doesn't have rownames")
    elif expr.index.has_duplicates:
        raise ValueError("The input assay object has duplicated rownames")
        

def validate_params(expr_data, gene_sets, kcdf, check_na, use):
    """
    Validate GSVA parameters
    
    Parameters:
    -----------
    expr_data : numpy.ndarray or scipy.sparse.spmatrix
        Expression data matrix
    gene_sets : dict
        Dictionary of gene sets
    kcdf : str
        Kernel to use
    check_na : str
        How to check for NA values
    use : str
        How to handle NA values
    """
    valid_kcdfs = ['auto', 'Gaussian', 'Poisson', 'none']
    if kcdf not in valid_kcdfs:
        raise ValueError(f"kcdf must be one of {valid_kcdfs}")
    
    valid_check_na = ['auto', 'yes', 'no']
    if check_na not in valid_check_na:
        raise ValueError(f"check_na must be one of {valid_check_na}")
    
    valid_use = ['everything', 'all.obs', 'na.rm']
    if use not in valid_use:
        raise ValueError(f"use must be one of {valid_use}")
    
    # Check for row names
    if hasattr(expr_data, 'index'):
        if expr_data.index.empty:
            raise ValueError("Expression data must have row names")
    else:
        import warnings
        warnings.warn("Expression data doesn't have row names")
        

# 标准差计算
def sd_plain(x, n):
    if n < 2:
        return 0.0

    # 第一次遍历：计算平均值
    total = sum(x[:n])
    tmp = total / n

    # 如果 tmp 是有限的，进行调整
    if math.isfinite(tmp):
        total = sum([xi - tmp for xi in x[:n]])
        tmp += total / n
    mean = tmp
    n1 = n - 1

    # 第二次遍历：计算方差
    total = sum([(xi - mean) ** 2 for xi in x[:n]])
    return math.sqrt(total / n1)

def remove_consecutive_duplicates(arr):
    """
    Remove consecutive duplicates from a sorted array.
    Mimics the C implementation's consecutive duplicate removal.
    """
    if len(arr) <= 1:
        return arr
    
    unique = [arr[0]]
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1]:
            unique.append(arr[i])
    return np.array(unique)

def ecdf_sparse_to_sparse(X_csc, X_csr, verbose=False):
    """
    Calculate empirical cumulative distribution function values on nonzero entries.
    Optimized version using vectorized numpy operations.
    """
    if not isinstance(X_csc, sparse.csc_matrix):
        X_csc = sparse.csc_matrix(X_csc)
    if not isinstance(X_csr, sparse.csr_matrix):
        X_csr = sparse.csr_matrix(X_csr)

    nr, nc = X_csc.shape
    result = X_csc.copy()
    result.data = result.data.astype(np.float64)

    row_iterator = tqdm(range(nr)) if verbose else range(nr)
    for i in row_iterator:
        # Get nonzero values in current row
        row_start = X_csr.indptr[i]
        row_end = X_csr.indptr[i + 1]
        row_data = X_csr.data[row_start:row_end]

        if len(row_data) == 0:
            continue

        # Use np.unique to get unique values, inverse indices, and counts
        unique_values, inverse_indices, counts = np.unique(row_data, return_inverse=True, return_counts=True)

        # Calculate ECDF values
        ecdf_values = np.cumsum(counts) / len(row_data)

        # Map ECDF values back to original positions
        row_ecdf = ecdf_values[inverse_indices]

        # Update result matrix - need to map back to CSC format
        for idx_in_row, idx in enumerate(range(row_start, row_end)):
            col = X_csr.indices[idx]

            # Find position in CSC format
            col_start = X_csc.indptr[col]
            col_end = X_csc.indptr[col + 1]
            row_pos = np.searchsorted(X_csc.indices[col_start:col_end], i)
            data_idx = col_start + row_pos

            result.data[data_idx] = row_ecdf[idx_in_row]

    return result

def ecdf_sparse_to_dense(X_csc, X_csr, verbose=False):
    """
    Calculate empirical cumulative distribution function values including zeros.
    Optimized version using vectorized numpy operations.
    """
    if not isinstance(X_csc, sparse.csc_matrix):
        X_csc = sparse.csc_matrix(X_csc)
    if not isinstance(X_csr, sparse.csr_matrix):
        X_csr = sparse.csr_matrix(X_csr)

    nr, nc = X_csc.shape
    result = np.zeros((nr, nc), dtype=np.float64)

    row_iterator = tqdm(range(nr)) if verbose else range(nr)
    for i in row_iterator:
        # Get full row including zeros
        row = X_csr[i].toarray().ravel()

        # Use np.unique to get unique values, inverse indices, and counts
        unique_values, inverse_indices, counts = np.unique(row, return_inverse=True, return_counts=True)

        # Calculate ECDF values
        ecdf_values = np.cumsum(counts) / nc

        # Map back to original positions using inverse indices
        result[i, :] = ecdf_values[inverse_indices]

    return result

def ecdf_dense_to_dense(X, verbose=False):
    """
    Calculate empirical cumulative distribution function values for dense matrix.
    Optimized version using vectorized numpy operations.
    """
    if sparse.issparse(X):
        X = X.toarray()

    nr, nc = X.shape
    result = np.zeros_like(X, dtype=np.float64)

    row_iterator = tqdm(range(nr)) if verbose else range(nr)
    for i in row_iterator:
        row = X[i]

        # Use np.unique to get unique values, inverse indices, and counts in one call
        unique_values, inverse_indices, counts = np.unique(row, return_inverse=True, return_counts=True)

        # Calculate ECDF values (cumulative counts / total)
        ecdf_values = np.cumsum(counts) / nc

        # Map back to original positions using inverse indices
        result[i, :] = ecdf_values[inverse_indices]

    return result

def ecdf_dense_to_dense_nas(X, verbose=False):
    """
    Calculate empirical cumulative distribution function values with NA handling.
    Optimized version using vectorized numpy operations.
    """
    if sparse.issparse(X):
        X = X.toarray()

    nr, nc = X.shape
    result = np.full_like(X, np.nan, dtype=np.float64)

    row_iterator = tqdm(range(nr)) if verbose else range(nr)
    for i in row_iterator:
        row = X[i]
        valid_mask = ~np.isnan(row)
        valid_values = row[valid_mask]

        if len(valid_values) == 0:
            continue

        # Use np.unique to get unique values, inverse indices, and counts
        unique_values, inverse_indices, counts = np.unique(valid_values, return_inverse=True, return_counts=True)

        # Calculate ECDF values (cumulative counts / total columns, matching original)
        ecdf_values = np.cumsum(counts) / nc

        # Create mapping array for valid positions
        valid_ecdf = ecdf_values[inverse_indices]

        # Fill result only at valid positions
        result[i, valid_mask] = valid_ecdf

    return result
#### sd with na rm
def sd_narm(x, n):
    x_slice = x[:n]
    x_valid = x_slice[~np.isnan(x_slice)]
    n_valid = len(x_valid)
    if n_valid < 2:
        return np.nan
    mean_x = np.sum(x_valid) / n_valid
    variance = np.sum((x_valid - mean_x) ** 2) / (n_valid - 1)
    return np.sqrt(variance)

# Core computation function
def row_d(x, y, size_density_n, size_test_n, Gaussk):
    """
    Core computation for row density with updated constants.
    """
    if Gaussk:
        bw = sd_plain(x, size_density_n) / sigma_factor
    else:
        bw = 0.5

    if Gaussk and not is_precomputed:
        init_cdfs()
        
    r = []
    for j in range(size_test_n):
        left_tail = 0.0
        for i in range(size_density_n):
            if Gaussk:
                cdf_val = precomputed_cdf(y[j] - x[i], bw)
                left_tail += cdf_val
            else:
                p_val = poisson.cdf(y[j], x[i] + bw)
                left_tail += p_val
        left_tail /= size_density_n
        if left_tail > 0 and left_tail < 1:
            rj = -np.log((1.0 - left_tail) / left_tail)
        else:
            rj = np.nan  # Avoid log(0) or division by zero
        r.append(rj)
    return r

# Row computation with NA propagation
def row_d_naprop(x, y, size_density_n, size_test_n, Gaussk):
    if Gaussk:
        if np.any(np.isnan(x[:size_density_n])):
            bw = np.nan
        else:
            bw = sd_plain(x, size_density_n) / sigma_factor
    else:
        bw = 0.5

    global is_precomputed
    if Gaussk and not is_precomputed:
        init_cdfs()
        is_precomputed = True

    r = [np.nan] * size_test_n

    if not np.isnan(bw):
        for j in range(size_test_n):
            if not np.isnan(y[j]):
                left_tail = 0.0
                i = 0
                valid_count = 0
                while i < size_density_n and not np.isnan(x[i]):
                    if Gaussk:
                        cdf_val = precomputed_cdf(y[j] - x[i], bw)
                        left_tail += cdf_val
                    else:
                        p_val = poisson.cdf(y[j], x[i] + bw)
                        left_tail += p_val
                    valid_count += 1
                    i += 1

                if valid_count == size_density_n:  # Only compute if we have all valid values
                    left_tail /= size_density_n
                    if left_tail > 0 and left_tail < 1:
                        rj = -1.0 * np.log((1.0 - left_tail) / left_tail)
                        r[j] = rj
    return r

# Row computation with NA removal
def row_d_narm(x, y, size_density_n, size_test_n, Gaussk):
    if Gaussk:
        bw = sd_narm(x, size_density_n) / sigma_factor
    else:
        bw = 0.5

    global is_precomputed
    if Gaussk and not is_precomputed:
        init_cdfs()
        is_precomputed = True

    r = [np.nan] * size_test_n
    for j in range(size_test_n):
        bw_is_nan = np.isnan(bw)
        yj_is_nan = np.isnan(y[j])
        if not bw_is_nan and not yj_is_nan:
            left_tail = 0.0
            valid_count = 0
            for i in range(size_density_n):
                xi_is_nan = np.isnan(x[i])
                if not xi_is_nan:
                    if Gaussk:
                        cdf_val = precomputed_cdf(y[j] - x[i], bw)
                        left_tail += cdf_val
                    else:
                        p_val = poisson.cdf(y[j], x[i] + bw)
                        left_tail += p_val
                    valid_count += 1
            if valid_count > 0:
                left_tail /= valid_count
                if 0 < left_tail < 1:
                    rj = -1.0 * np.log((1.0 - left_tail) / left_tail)
                    r[j] = rj
    return r
    
def sd(x):
    """Calculate standard deviation"""
    return np.std(x, ddof=1)

def sd_plain(x, n):
    if n < 2:
        return 0.0

    # 第一次遍历：计算平均值
    total = sum(x[:n])
    tmp = total / n

    # 如果 tmp 是有限的，进行调整
    if math.isfinite(tmp):
        total = sum([xi - tmp for xi in x[:n]])
        tmp += total / n
    mean = tmp
    n1 = n - 1

    # 第二次遍历：计算方差
    total = sum([(xi - mean) ** 2 for xi in x[:n]])
    return math.sqrt(total / n1)
    
 # row_d_nologodds 函数
def row_d_nologodds(x, y, size_density_n, size_test_n, Gaussk):
    # 计算带宽
    if Gaussk:
        sd_val = sd_plain(x, size_density_n)
        if sd_val < sys.float_info.epsilon:
            bw = 0.5
        else:
            bw = sd_val / sigma_factor
    else:
        bw = 0.5

    # 如有需要，初始化 CDF
    global is_precomputed
    if Gaussk and not is_precomputed:
        precomputed_values = init_cdfs()
        is_precomputed = True

    # 初始化结果向量
    r = [0.0] * size_test_n

    # 处理每个测试点
    for j in range(size_test_n):
        left_tail = 0.0

        # 累加密度点的贡献
        for i in range(size_density_n):
            if Gaussk:
                cdf_val = precomputed_cdf(y[j] - x[i], bw)
                if cdf_val is None or math.isnan(cdf_val):
                    continue  # 如果 cdf_val 是 None 或 NaN，跳过
                left_tail += cdf_val
            else:
                p_val = poisson.cdf(y[j], x[i] + bw)
                if p_val is None or math.isnan(p_val):
                    continue  # 如果 p_val 是 None 或 NaN，跳过
                left_tail += p_val

        # 计算平均贡献
        r[j] = left_tail / size_density_n

    return r

def compute_kcdf_row_vectorized(x, size_density_n, sigma_factor, Gaussk=True):
    if Gaussk:
        if size_density_n < 2:
            bw = 0.5
        else:
            mean = np.mean(x)
            var = np.var(x, ddof=1)
            sd_val = np.sqrt(var)
            bw = sd_val / sigma_factor if sd_val > np.finfo(float).eps else 0.5
        
        x_matrix = x.reshape(-1, 1)
        x_row = x.reshape(1, -1)
        z_scores = (x_matrix - x_row) / bw
        cdf_matrix = norm.cdf(z_scores)
        return np.mean(cdf_matrix, axis=1)
    else:
        # 向量化Poisson计算
        x_matrix = x.reshape(-1, 1)
        lambda_matrix = x.reshape(1, -1) + 0.5
        cdf_matrix = poisson.cdf(x_matrix, lambda_matrix)
        return np.mean(cdf_matrix, axis=1)

def kcdfvals_sparse_to_sparse(XCsp, XRsp, Gaussk=True, verbose=True):
    nr, nc = XCsp.shape
    new_data = np.zeros(XRsp.data.shape, dtype=np.float64)
    
    for i in tqdm(range(nr)) if verbose else range(nr):
        row_start, row_end = XRsp.indptr[i:i+2]
        nv = row_end - row_start
        
        if nv > 0:
            x = XRsp.data[row_start:row_end].astype(np.float64)
            new_data[row_start:row_end] = compute_kcdf_row_vectorized(x, nv, sigma_factor, Gaussk)
    
    return csr_matrix((new_data, XRsp.indices, XRsp.indptr), shape=XRsp.shape).tocsc()
    
def row_d_nologodds_vectorized(x, size_density_n, Gaussk):
    if Gaussk:
        sd_val = np.std(x) if size_density_n > 1 else 0
        bw = sd_val / sigma_factor if sd_val > sys.float_info.epsilon else 0.5
    else:
        bw = 0.5
    
    x_matrix = x.reshape(-1, 1)
    x_row = x.reshape(1, -1)
    diff_matrix = x_matrix - x_row
    
    if Gaussk:
        cdf_matrix = precomputed_cdf_vectorized(diff_matrix, bw)
    else:
        lambda_matrix = x_row + bw
        cdf_matrix = poisson.cdf(x_matrix, lambda_matrix)
    
    return np.mean(cdf_matrix, axis=1)

def kcdfvals_sparse_to_dense(XCsp, XRsp, Gaussk=True, verbose=True):
    nr, nc = XCsp.shape
    kcdf_vals = np.zeros((nr, nc), order='F')
    global is_precomputed
    if Gaussk and not is_precomputed:
        init_cdfs()
        is_precomputed = True
    
    for i in tqdm(range(nr)) if verbose else range(nr):
        x = np.zeros(nc)
        row_start, row_end = XRsp.indptr[i:i+2]
        
        if row_end > row_start:
            indices = XRsp.indices[row_start:row_end]
            x[indices] = XRsp.data[row_start:row_end]
            nz_values = row_d_nologodds_vectorized(x, nc, Gaussk)
            kcdf_vals[i, :] = nz_values
    
    return kcdf_vals
    
# Main matrix density function
def matrix_density(density_data, test_data, n_density_samples,
                   n_test_samples, n_genes, Gaussk=True,
                   any_na=False, na_use=1, verbose=False):
    result = np.zeros(n_test_samples * n_genes)

    if verbose:
        print("Processing genes...")

    for j in range(n_genes):
        offset_density_start = j * n_density_samples
        offset_density_end = (j + 1) * n_density_samples
        offset_test_start = j * n_test_samples
        offset_test_end = (j + 1) * n_test_samples

        # Extract current gene's data
        x = density_data[offset_density_start:offset_density_end]
        y = test_data[offset_test_start:offset_test_end]

        if not any_na:
            temp_result = row_d(x, y, n_density_samples, n_test_samples, Gaussk)
        elif na_use == 1:  # Propagate NAs
            temp_result = row_d_naprop(x, y, n_density_samples, n_test_samples, Gaussk)
        else:  # Remove NAs
            temp_result = row_d_narm(x, y, n_density_samples, n_test_samples, Gaussk)

        result[offset_test_start:offset_test_end] = temp_result

        if verbose and (j + 1) % 1000 == 0:
            print(f"Processed {j + 1}/{n_genes} genes")

    return result


import numpy as np
from scipy.stats import poisson
from tqdm import tqdm
import sys

def matrix_density_vectorized(density_data, test_data, n_density_samples,
                            n_test_samples, n_genes, Gaussk=True,
                            any_na=False, na_use=1, verbose=False):
    """
    向量化版本的matrix_density，确保结果准确性
    """
    # 重塑数据为更易处理的形式
    density_matrix = density_data.reshape(n_genes, n_density_samples)
    test_matrix = test_data.reshape(n_genes, n_test_samples)
    result = np.zeros(n_genes * n_test_samples)
    
    if verbose:
        print("Processing genes...")
        
    if not any_na:
        if Gaussk:
            # 计算每个基因的带宽
            bw = np.zeros(n_genes)
            for i in range(n_genes):
                sd_val = sd_plain(density_matrix[i], n_density_samples)
                bw[i] = sd_val / sigma_factor if sd_val > sys.float_info.epsilon else 0.5
            
            # OPTIMIZATION: Vectorized computation per gene
            for i in range(n_genes):
                x = density_matrix[i].reshape(1, -1)  # 1 x n_density
                y = test_matrix[i].reshape(-1, 1)     # n_test x 1

                # 计算所有差值
                diff_matrix = y - x  # n_test x n_density

                # OPTIMIZATION: Use vectorized CDF computation instead of nested loops
                cdf_matrix = precomputed_cdf_vectorized(diff_matrix, bw[i])

                # 计算每行的平均值
                left_tails = np.mean(cdf_matrix, axis=1)

                # 计算logodds
                valid_mask = (left_tails > 0) & (left_tails < 1)
                temp_result = np.full(n_test_samples, np.nan)
                temp_result[valid_mask] = -np.log((1.0 - left_tails[valid_mask]) / left_tails[valid_mask])

                result[i * n_test_samples:(i + 1) * n_test_samples] = temp_result

                if verbose and (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{n_genes} genes")
                    
        else:
            # 使用向量化的泊松分布计算
            for i in range(n_genes):
                x = density_matrix[i].reshape(1, -1)  # (1, n_density)
                y = test_matrix[i].reshape(-1, 1)     # (n_test, 1)
                
                # 一次性计算所有lambda值并广播
                lambda_matrix = x + 0.5  # broadcast to (n_test, n_density)
                
                # 使用poisson.cdf进行向量化计算
                cdf_matrix = poisson.cdf(y, lambda_matrix)
                
                # 计算每行的平均值
                left_tails = np.mean(cdf_matrix, axis=1)
                
                # 计算logodds
                valid_mask = (left_tails > 0) & (left_tails < 1)
                temp_result = np.full(n_test_samples, np.nan)
                temp_result[valid_mask] = -np.log((1.0 - left_tails[valid_mask]) / left_tails[valid_mask])
                
                result[i * n_test_samples:(i + 1) * n_test_samples] = temp_result
                
                if verbose and (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{n_genes} genes")
    else:
        # 对于含有NA的情况，仍使用原始的逐个处理方式
        for j in range(n_genes):
            offset_density_start = j * n_density_samples
            offset_density_end = (j + 1) * n_density_samples
            offset_test_start = j * n_test_samples
            offset_test_end = (j + 1) * n_test_samples
            
            x = density_data[offset_density_start:offset_density_end]
            y = test_data[offset_test_start:offset_test_end]
            
            if na_use == 1:
                temp_result = row_d_naprop(x, y, n_density_samples, n_test_samples, Gaussk)
            else:
                temp_result = row_d_narm(x, y, n_density_samples, n_test_samples, Gaussk)
                
            result[offset_test_start:offset_test_end] = temp_result
            
            if verbose and (j + 1) % 1000 == 0:
                print(f"Processed {j + 1}/{n_genes} genes")
    
    return result


def sufficient_ssize(expr, kcdf_min_ssize):
    """
    Check if the expression matrix has sufficient sample size.
    
    Parameters:
    -----------
    expr : numpy.ndarray or scipy.sparse.spmatrix
        Expression data matrix
    kcdf_min_ssize : int
        Minimum required sample size
        
    Returns:
    --------
    bool
        True if sample size is sufficient, False otherwise
    """
    from scipy import sparse
    
    # For sparse matrix
    if sparse.issparse(expr):
        # Calculate average number of non-zero values per row
        return (expr.nnz / expr.shape[0]) >= kcdf_min_ssize
    
    # For dense matrix or any other case
    return expr.shape[1] >= kcdf_min_ssize

def parse_kcdf_param(expr, kcdf, kcdf_min_ssize, sparse_output, verbose):
    """
    Parse kernel CDF parameters for GSVA analysis
    
    Parameters:
    -----------
    expr : numpy.ndarray, scipy.sparse.spmatrix, or pandas.DataFrame
        Expression data matrix
    kcdf : str
        Kernel to use
    kcdf_min_ssize : int
        Minimum sample size for kernel estimation
    sparse_output : bool
        Whether using sparse algorithm
    verbose : bool
        Whether to print progress messages
        
    Returns:
    --------
    dict
        Dictionary containing kernel and Gaussk parameters
    """
    from scipy import sparse
    import numpy as np
    import pandas as pd
    
    kernel = False
    Gaussk = True  # default (True) is Gaussian kernel, Poisson otherwise (False)
    
    if kcdf == "auto":
        if verbose:
            print("kcdf='auto' (default)")
        
        if not sufficient_ssize(expr, kcdf_min_ssize):
            kernel = True
            if sparse.issparse(expr):
                data = expr.data
                sample_size = min(1000, len(data))
                sampled_data = np.random.choice(data, size=sample_size, replace=False)
                Gaussk = np.any((sampled_data < 0) | (sampled_data != np.floor(sampled_data)))
            else:
                # Convert DataFrame to numpy array if needed
                if isinstance(expr, pd.DataFrame):
                    expr_values = expr.values
                else:
                    expr_values = expr
                    
                # Check if first element is integer type
                Gaussk = not np.issubdtype(expr_values.dtype, np.integer)
    
    elif kcdf == "Gaussian":
        kernel = True
        Gaussk = True
    
    elif kcdf == "Poisson":
        kernel = True
        Gaussk = False
    
    else:
        kernel = False
    
    if verbose:
        if sparse.issparse(expr) and sparse_output:
            print("GSVA sparse algorithm")
        else:
            print("GSVA dense (classical) algorithm")
        
        if kernel:
            if Gaussk:
                print("Row-wise ECDF estimation with Gaussian kernels")
            else:
                print("Row-wise ECDF estimation with Poisson kernels")
        else:
            print("Direct row-wise ECDFs estimation")
    
    return {'kernel': kernel, 'Gaussk': Gaussk}
####
def sparse2column_list(sparse_matrix):
    """
    Converts a CSC sparse matrix into a list of its columns.

    Parameters:
    -----------
    sparse_matrix : scipy.sparse.csc_matrix
        Sparse matrix.

    Returns:
    --------
    column_list : list of numpy.ndarray
        List where each element is an array of non-zero values in the corresponding column.
    """
    if not sparse.isspmatrix_csc(sparse_matrix):
        sparse_matrix = sparse_matrix.tocsc()

    column_list = []
    for i in range(sparse_matrix.shape[1]):
        start_idx = sparse_matrix.indptr[i]
        end_idx = sparse_matrix.indptr[i + 1]
        col_data = sparse_matrix.data[start_idx:end_idx]
        column_list.append(col_data)
    return column_list
    
####
def sparse_column_apply_and_replace(sparse_matrix, func, **kwargs):
    """
    Applies a function to each column of a sparse matrix and replaces the data.

    Parameters:
    -----------
    sparse_matrix : scipy.sparse.csc_matrix
        Sparse matrix.
    func : callable
        Function to apply to each column's data.
    **kwargs :
        Additional arguments to pass to the function.

    Returns:
    --------
    sparse_matrix : scipy.sparse.csc_matrix
        Sparse matrix with updated data.
    """
    if not sparse.isspmatrix_csc(sparse_matrix):
        sparse_matrix = sparse_matrix.tocsc()
    else:
        sparse_matrix = sparse_matrix.copy()

    for i in range(sparse_matrix.shape[1]):
        start_idx = sparse_matrix.indptr[i]
        end_idx = sparse_matrix.indptr[i + 1]
        col_data = sparse_matrix.data[start_idx:end_idx]

        # Apply the function to the column data
        new_col_data = func(col_data, **kwargs)

        # Replace the data
        sparse_matrix.data[start_idx:end_idx] = new_col_data

    # If data is integer, convert to float (double precision)
    if np.issubdtype(sparse_matrix.data.dtype, np.integer):
        sparse_matrix.data = sparse_matrix.data.astype(np.float64)

    return sparse_matrix



#####

def rank_ties_last(arr):
    """
    Implements R's rank function with ties.method = 'last'
    Exactly matches R's behavior for handling ties.
    Optimized version using vectorized numpy operations.

    Parameters:
    arr: array-like object
        The array to be ranked

    Returns:
    numpy.ndarray
        Array of ranks matching R's rank(x, ties.method='last')
    """
    arr = np.asarray(arr)
    n = len(arr)

    # Create mask for non-NA values
    mask = ~np.isnan(arr)

    if not mask.any():  # If all values are NA
        return np.full(n, np.nan)

    # Initialize ranks array
    ranks = np.full(n, np.nan)

    # Get valid values and their original indices
    valid_idx = np.where(mask)[0]
    valid_values = arr[mask]
    n_valid = len(valid_values)

    # Sort by value (ascending), then by original index (descending) for ties
    # This gives us "last" tie-breaking behavior
    sort_keys = np.lexsort((-valid_idx, valid_values))

    # Assign ranks based on sorted order
    sorted_ranks = np.empty(n_valid, dtype=float)
    sorted_ranks[sort_keys] = np.arange(1, n_valid + 1)

    # Place ranks back into original positions
    ranks[mask] = sorted_ranks

    return ranks

#####
def colRanks(Z, ties_method='last', preserve_shape=True):
    """
    Calculates ranks for each column in a matrix Z.
    Optimized version with vectorized operations for standard tie methods.

    Parameters:
    -----------
    Z : numpy.ndarray
        Input matrix.
    ties_method : str
        Method for handling ties ('average', 'min', 'max', 'dense', 'ordinal', 'last').
    preserve_shape : bool
        Whether to preserve the input shape.

    Returns:
    --------
    ranked_Z : numpy.ndarray
        Matrix of ranks.
    """
    n_rows, n_cols = Z.shape
    ranked_Z = np.empty_like(Z, dtype=float)

    if ties_method == 'last':
        # Use optimized rank_ties_last for each column
        for col_idx in range(n_cols):
            ranked_Z[:, col_idx] = rank_ties_last(Z[:, col_idx])
    else:
        # For standard tie methods, use scipy's rankdata which is already optimized
        # Process all columns at once using apply_along_axis for better performance
        ranked_Z = np.apply_along_axis(lambda x: rankdata(x, method=ties_method), 0, Z)

    return ranked_Z

def check_for_na_values(expr_data, check_na, use):
    """
    Check for NA values in expression data
    
    Parameters:
    -----------
    expr_data : numpy.ndarray or scipy.sparse.spmatrix
        Expression data matrix
    check_na : str
        How to check for NA values ('auto', 'yes', 'no')
    use : str
        How to handle NA values ('everything', 'all.obs', 'na.rm')
        
    Returns:
    --------
    dict
        Dictionary with did_check_na and any_na flags
    """
    from scipy import sparse
    import numpy as np
    
    if check_na == 'no':
        return {'did_check_na': False, 'any_na': False}
        
    # For 'auto' and 'yes'
    has_na = np.isnan(expr_data).any() if isinstance(expr_data, np.ndarray) else \
            np.isnan(expr_data.data).any() if sparse.issparse(expr_data) else False
            
    return {'did_check_na': True, 'any_na': has_na}

#####
def filter_genes_bk(expr, remove_constant=True, remove_nz_constant=True):
    """Filter out genes with constant values and constant non-zero values."""
    # Store original shape for debugging
    original_shape = expr.shape
    print(f"Original shape: {original_shape}")
    
    # Check data type
    print(f"Input type: {type(expr)}")
    
    # Handle sparse matrix conversion for min/max calculation
    if sparse.issparse(expr):
        gene_min = pd.Series(expr.min(axis=1).toarray().flatten())
        gene_max = pd.Series(expr.max(axis=1).toarray().flatten())
    else:
        gene_min = expr.min(axis=1, skipna=True)
        gene_max = expr.max(axis=1, skipna=True)
    
    constant_genes = (gene_min == gene_max)
    print(f"Number of constant genes: {constant_genes.sum()}")
    
    if constant_genes.any():
        invalid_genes = constant_genes
        msg = f"{invalid_genes.sum()} genes with constant values throughout the samples"
        warnings.warn(msg)
        if remove_constant:
            warnings.warn("Genes with constant values are discarded")
            if sparse.issparse(expr):
                expr = expr[~invalid_genes.values]
            else:
                expr = expr.loc[~invalid_genes]
            print(f"Shape after removing constant genes: {expr.shape}")
    
    # Convert to CSR format for consistent row-wise operations
    if sparse.issparse(expr):
        expr_csr = expr.tocsr()
        data = expr_csr
    else:
        data = sparse.csr_matrix(expr.values)
    
    # Find constant non-zero genes using consistent sparse matrix approach
    constant_nz_genes = []
    for i in range(data.shape[0]):
        row = data[i]
        nz_vals = row.data  # 获取非零值
        if len(nz_vals) > 0 and np.min(nz_vals) == np.max(nz_vals):
            constant_nz_genes.append(i)
    
    print(f"Number of constant non-zero genes: {len(constant_nz_genes)}")
    
    if constant_nz_genes:
        msg = f"{len(constant_nz_genes)} genes with constant non-zero values throughout the samples"
        warnings.warn(msg)
        if remove_nz_constant:
            warnings.warn("Genes with constant non-zero values are discarded")
            mask = np.ones(expr.shape[0], dtype=bool)
            mask[constant_nz_genes] = False
            if sparse.issparse(expr):
                expr = expr[mask]
            else:
                expr = expr[mask]
            print(f"Shape after removing constant non-zero genes: {expr.shape}")
    
    if sparse.issparse(expr):
        remaining_rows = expr.shape[0]
    else:
        remaining_rows = expr.shape[0]
        
    if remaining_rows < 2:
        raise ValueError("Less than two genes in the input assay object")
    
    return expr
import pandas as pd
import numpy as np
from scipy import sparse
import warnings
from joblib import Parallel, delayed
import multiprocessing

def filter_genes(expr, remove_constant=True, remove_nz_constant=True):
    """
    Filter constant genes and return mask and filtered indices.
    Optimized version using vectorized numpy operations.
    """
    original_index = expr.index if hasattr(expr, 'index') else np.arange(expr.shape[0])
    keep_genes = np.ones(expr.shape[0], dtype=bool)

    gene_ranges = np.vstack([
        expr.min(axis=1).toarray().flatten() if sparse.issparse(expr) else expr.min(axis=1),
        expr.max(axis=1).toarray().flatten() if sparse.issparse(expr) else expr.max(axis=1)
    ]).T

    constant_genes = (gene_ranges[:, 0] == gene_ranges[:, 1])
    if np.any(constant_genes) and remove_constant:
        print(f"Found {np.sum(constant_genes)} genes with constant values")
        keep_genes &= ~constant_genes

    if sparse.issparse(expr) and remove_nz_constant:
        expr_t = expr.T.tocsc()
        n_genes = expr.shape[0]

        # Vectorized computation of min/max for non-zero values per row
        # Using reduceat for efficient computation
        indptr = expr_t.indptr
        data = expr_t.data

        # Find non-empty rows (genes with non-zero values)
        row_lengths = np.diff(indptr)
        non_empty_mask = row_lengths > 0

        # Initialize arrays
        nz_min = np.zeros(n_genes)
        nz_max = np.zeros(n_genes)

        if np.any(non_empty_mask):
            non_empty_indices = np.where(non_empty_mask)[0]

            # Get start indices for reduceat
            reduce_starts = indptr[non_empty_indices]

            # Compute min and max using reduceat
            # Note: reduceat computes reduction between consecutive indices
            nz_min_vals = np.minimum.reduceat(data, reduce_starts)
            nz_max_vals = np.maximum.reduceat(data, reduce_starts)

            nz_min[non_empty_indices] = nz_min_vals
            nz_max[non_empty_indices] = nz_max_vals

        # Constant non-zero genes: min == max and min != 0 (has non-zero values)
        constant_nz_genes = (nz_min == nz_max) & non_empty_mask & (nz_min != 0)

        if np.any(constant_nz_genes):
            print(f"Found {np.sum(constant_nz_genes)} genes with constant non-zero values")
            keep_genes &= ~constant_nz_genes

    if keep_genes.sum() < 2:
        raise ValueError("Less than two genes in the input assay object")

    filtered_indices = original_index[keep_genes]
    return keep_genes, filtered_indices
    
def filter_genes2(expr, 
                remove_constant=True, 
                remove_nz_constant=True, 
                use_sparse = False,
                n_jobs=-1):
    """
    Filter constant genes and constant non-zero genes from expression matrix.
    Returns boolean mask and filtered gene indices.

    Parameters:
    - expr: pd.DataFrame/scipy.sparse.spmatrix/numpy.ndarray expression data
    - remove_constant: Whether to remove constant genes
    - remove_nz_constant: Whether to remove constant non-zero genes
    - n_jobs: Number of parallel jobs

    Returns:
    - keep_genes: Boolean mask array
    - filtered_indices: Filtered gene indices
    """
    if isinstance(expr, pd.DataFrame):
        original_index = expr.index
        if use_sparse:
            data_matrix = sparse.csr_matrix(expr.values)
        else:
            data_matrix = expr.values
    elif sparse.isspmatrix(expr):
        original_index = np.arange(expr.shape[0])
        data_matrix = expr.tocsr() if not sparse.isspmatrix_csr(expr) else expr
    elif isinstance(expr, np.ndarray):
        original_index = np.arange(expr.shape[0])
        if use_sparse:
            data_matrix = sparse.csr_matrix(expr)
        else:
            data_matrix = expr
    else:
        raise TypeError("Input must be DataFrame, sparse matrix or ndarray")

    print(f"Input data shape: {data_matrix.shape}")
    keep_genes = np.ones(data_matrix.shape[0], dtype=bool)
    
    if remove_constant:
        gene_min = data_matrix.min(axis=1).toarray().flatten()
        gene_max = data_matrix.max(axis=1).toarray().flatten()
        constant_mask = gene_min == gene_max
        num_constant = np.sum(constant_mask)
        print(f"Found {num_constant} constant genes")
        keep_genes &= ~constant_mask
    
    if remove_nz_constant:
        def is_constant_nz(row):
            return row.nnz > 0 and np.unique(row.data).size == 1
            
        num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        print(f"Using {num_cores} cores for parallel processing")
        
        constant_nz_flags = Parallel(n_jobs=num_cores)(
            delayed(is_constant_nz)(data_matrix.getrow(i)) 
            for i in range(data_matrix.shape[0])
        )
        constant_nz_mask = np.array(constant_nz_flags)
        num_constant_nz = np.sum(constant_nz_mask)
        print(f"Found {num_constant_nz} constant non-zero genes")
        keep_genes &= ~constant_nz_mask
    
    num_remaining = np.sum(keep_genes)
    print(f"Number of genes remaining: {num_remaining}")
    
    if num_remaining < 2:
        raise ValueError("Less than 2 genes remain after filtering")
        
    filtered_indices = original_index[keep_genes]
    return keep_genes, filtered_indices
        
def map_gene_sets_to_features(gene_sets, features):
    """
    Maps gene sets to feature indices (1-based to match R)
    """
    mapped_gene_sets = {}
    for gs_name, genes in gene_sets.items():
        # Add 1 to make indices 1-based
        mapped_genes = [features.get_loc(gene) + 1 for gene in genes if gene in features]
        if mapped_genes:
            mapped_gene_sets[gs_name] = mapped_genes
    if not any(mapped_gene_sets.values()):
        raise ValueError("No identifiers in the gene sets could be matched to the identifiers in the expression data.")
    return mapped_gene_sets

def filter_gene_sets(mapped_gene_sets, min_size=1, max_size=np.inf):
    filtered_mapped_gene_sets = {
        gs_name: genes for gs_name, genes in mapped_gene_sets.items()
        if min_size <= len(genes) <= max_size
    }
    return filtered_mapped_gene_sets

def filter_and_map_gene_sets(param, filtered_data_matrix, verbose=False):
    gene_sets = param.gene_sets
    min_size = param.min_size
    max_size = param.max_size
    
    mapped_gene_sets = map_gene_sets_to_features(gene_sets, filtered_data_matrix.index)
    
    filtered_mapped_gene_sets = filter_gene_sets(mapped_gene_sets,
                                                 min_size=min_size,
                                                 max_size=max_size)
    
    if not filtered_mapped_gene_sets:
        raise ValueError("The gene set list is empty! Filter may be too stringent.")
    
    if len(set(filtered_mapped_gene_sets.keys())) != len(filtered_mapped_gene_sets):
        raise ValueError("The gene set list contains duplicated gene set names.")
    
    if any(len(genes) == 1 for genes in filtered_mapped_gene_sets.values()):
        warnings.warn("Some gene sets have size one. Consider setting min_size > 1")
    
    return filtered_mapped_gene_sets

def filter_and_map_genes_and_gene_sets(param, remove_constant=True, remove_nz_constant=True, filtered_gene = True, verbose=False,use_sparse=False,chunk_size=1000):
    expr_data = param.expr_data
    data_matrix = expr_data  # 假设数据已经是解包的 DataFrame
    orig_index = data_matrix.index
    orig_columns = data_matrix.columns
    if filtered_gene:
        mask, filtered_indices = filter_genes(data_matrix,remove_constant=remove_constant,remove_nz_constant=remove_nz_constant,n_jobs = param.n_jobs)
        if isinstance(data_matrix, pd.DataFrame):
            filtered_data_matrix = data_matrix.loc[filtered_indices]
        else:
            filtered_data_matrix = data_matrix[mask]
        if use_sparse:
            filtered_data_matrix = filtered_data_matrix.toarray()
            new_index = orig_index[mask]
            filtered_data_matrix = pd.DataFrame(filtered_data_matrix,
                          index=new_index,
                          columns=orig_columns)
    else:
        filtered_data_matrix = data_matrix
   # filtered_data_matrix = filter_genes(data_matrix,
   #                                    remove_constant=remove_constant,
   #                                     remove_nz_constant=remove_nz_constant)
    
    filtered_mapped_gene_sets = filter_and_map_gene_sets(param,
                                                         filtered_data_matrix=filtered_data_matrix,
                                                         verbose=verbose)
    
    return {
        'filtered_data_matrix': filtered_data_matrix,
        'filtered_mapped_gene_sets': filtered_mapped_gene_sets
    }

def split_indices(n: int, chunks: int) -> List[range]:
    """Split n indices into chunks"""
    chunk_size = n // chunks
    remainder = n % chunks
    indices = []
    start = 0
    
    for i in range(chunks):
        size = chunk_size + (1 if i < remainder else 0)
        indices.append(range(start, start + size))
        start += size
        
    return indices

def order_rankstat(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Python implementation of order_rankstat_R
    
    Parameters:
    -----------
    x : np.ndarray
        Input array
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (ordering, rank statistics)
    """
    n = len(x)
    # Get ordering (convert to 1-based indexing)
    ord_idx = np.argsort(-x) + 1
    
    # Calculate rank statistics
   # rank_stats = np.abs(np.arange(n, dtype=float) - (n/2))
    rank_stats = np.abs(np.arange(1, len(expr) + 1, dtype=float) - (len(expr)/2))
    return ord_idx, rank_stats

def order_rankstat_sparse_to_sparse(X: sparse.csc_matrix, j: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Python implementation of order_rankstat_sparse_to_sparse_R
    """
    j = j - 1  # Convert to 0-based indexing
    col_start = X.indptr[j]
    col_end = X.indptr[j+1]
    
    # Get non-zero values and their indices
    values = X.data[col_start:col_end]
    rows = X.indices[col_start:col_end]
    
    # Sort non-zero values
    nnz = len(values)
    sort_idx = np.argsort(-values)
    
    # Create full ordering
    n = X.shape[0]
    ordering = np.zeros(n, dtype=int)
    rank_stats = np.full(n, (nnz/2.0 + 1.0))
    
    # Fill in non-zero entries
    for i, idx in enumerate(sort_idx):
        ordering[i] = rows[idx] + 1  # Convert to 1-based
        rank_stats[rows[idx]] = abs(nnz - i - (nnz/2.0))
    
    # Fill in zero entries
    zero_idx = n - nnz
    for i in range(n):
        if i not in rows:
            ordering[zero_idx] = i + 1  # Convert to 1-based
            zero_idx += 1
    
    return ordering, rank_stats


###confirm is correct
def ranks2stats(r, use_sparse=False):
    """
    Convert ranks into decreasing order statistics and symmetric rank statistics
    
    Parameters:
    -----------
    r : array-like
        Input ranks
    use_sparse : bool
        Whether to use sparse computation
        
    Returns:
    --------
    dict
        Dictionary with 'dos' and 'srs' keys
    """
    r = np.asarray(r)
    p = len(r)
    
    # Find mask for rank 0
    mask = r == 0
    
    # Convert to integer ranks
    r_dense = r.astype(int)
    
    # Handle sparse ranks
    if np.any(mask):
        nzs = np.sum(mask)
        r_dense[~mask] += nzs  # shift ranks of nonzero values
        r_dense[mask] = np.arange(1, nzs + 1)  # zeros get increasing ranks
    
    # Calculate decreasing order statistics
    dos = p - r_dense + 1
    
    # Calculate symmetric rank statistics
    if np.any(mask) and use_sparse:
        r_copy = r.copy()
        r_copy[~mask] += 1  # shift ranks of nonzero values by one
        r_copy[mask] = 1    # all zeros get the same first rank
        srs = np.abs(np.max(r_copy)/2 - r_copy)
    else:
        srs = np.abs(p/2 - r_dense)
    
    return {'dos': dos, 'srs': srs}


def ranks2stats_nas(r, use_sparse=False):
    """
    Convert ranks into decreasing order statistics and symmetric rank statistics
    skipping NA values
    
    Parameters:
    -----------
    r : array-like
        Input ranks
    use_sparse : bool
        Whether to use sparse computation
        
    Returns:
    --------
    dict
        Dictionary with 'dos' and 'srs' keys
    """
    r = np.asarray(r)
    p = len(r)
    
    # Handle NA values
    na_mask = np.isnan(r)
    if np.all(na_mask):
        return {'dos': np.full(p, np.nan), 'srs': np.full(p, np.nan)}
    
    # Create copies to avoid modifying input
    dos = np.full(p, np.nan)  # Initialize with NaN
    srs = np.full(p, np.nan)  # Initialize with NaN
    
    n_nas = np.sum(na_mask)
    mask = (~na_mask) & (r == 0)
    valid_mask = ~na_mask
    
    # Only process non-NA values
    r_dense = np.zeros(p, dtype=int)
    r_dense[valid_mask] = r[valid_mask].astype(int)
    
    # Handle sparse ranks
    if np.any(mask):
        nzs = np.sum(mask)
        r_dense[valid_mask & ~mask] += nzs  # shift ranks of nonzero values
        r_dense[mask] = np.arange(1, nzs + 1)  # zeros get increasing ranks
    
    # Calculate decreasing order statistics for non-NA values
    dos[valid_mask] = p - n_nas - r_dense[valid_mask] + 1
    
    # Calculate symmetric rank statistics for non-NA values
    if np.any(mask) and use_sparse:
        r_copy = r.copy()
        r_copy[valid_mask & ~mask] += 1  # shift ranks of nonzero values by one
        r_copy[mask] = 1    # all zeros get the same first rank
        srs[valid_mask] = np.abs(np.nanmax(r_copy)/2 - r_copy[valid_mask])
    else:
        srs[valid_mask] = np.abs((p - n_nas)/2 - r_dense[valid_mask])
    
    return {'dos': dos, 'srs': srs}

def filter_and_map_gene_sets_s(gene_sets,min_size,max_size,filtered_data_matrix, verbose=False):
    
    mapped_gene_sets = map_gene_sets_to_features(gene_sets, filtered_data_matrix.index)
    
    filtered_mapped_gene_sets = filter_gene_sets(mapped_gene_sets,
                                                 min_size=min_size,
                                                 max_size=max_size)
    
    if not filtered_mapped_gene_sets:
        raise ValueError("The gene set list is empty! Filter may be too stringent.")
    
    if len(set(filtered_mapped_gene_sets.keys())) != len(filtered_mapped_gene_sets):
        raise ValueError("The gene set list contains duplicated gene set names.")
    
    if any(len(genes) == 1 for genes in filtered_mapped_gene_sets.values()):
        warnings.warn("Some gene sets have size one. Consider setting min_size > 1")
    
    return filtered_mapped_gene_sets

def filter_and_map_genes_and_gene_sets_s(expr_data, gene_sets, min_size=1,max_size=np.inf,remove_constant=True, remove_nz_constant=True, use_sparse=False, verbose=False):
    data_matrix = expr_data  # 假设数据已经是解包的 DataFrame
    orig_index = data_matrix.index
    orig_columns = data_matrix.columns
    if use_sparse:
        expr_data = sparse.csc_matrix(data_matrix)
        mask, filtered_indices = filter_genes(expr_data, remove_constant=remove_constant, remove_nz_constant=remove_constant)
        filtered_data_matrix = expr_data[mask]
    else:
        mask, filtered_indices = filter_genes(data_matrix, remove_constant=remove_constant, remove_nz_constant=remove_constant)
        filtered_data_matrix = data_matrix.loc[filtered_indices]
    # Filter genes
    if use_sparse:
        filtered_data_matrix = filtered_data_matrix.toarray()
    new_index = orig_index[mask]
    filtered_data_matrix = pd.DataFrame(filtered_data_matrix,
                          index=new_index,
                          columns=orig_columns)
    filtered_mapped_gene_sets = filter_and_map_gene_sets_s(gene_sets=gene_sets,min_size=min_size,max_size=max_size,
                                                         filtered_data_matrix=filtered_data_matrix,
                                                         verbose=verbose)
    
    return {
        'filtered_data_matrix': filtered_data_matrix,
        'filtered_mapped_gene_sets': filtered_mapped_gene_sets
    }

def convert_indices_to_genes(filtered_mapped_genesets, gene_names):
    """
    将基因集的索引转换回基因名
    
    Parameters:
    -----------
    filtered_mapped_genesets : dict
        基因集字典，键为基因集名称，值为基因索引(1-based)
    gene_names : list or pd.Index
        基因名列表
    
    Returns:
    --------
    dict
        转换后的基因集字典，键为基因集名称，值为基因名列表
    """
    converted_genesets = {}
    for gset_name, indices in filtered_mapped_genesets.items():
        # 将1-based索引转换为0-based并获取基因名
        gene_list = [gene_names[idx-1] for idx in indices]
        converted_genesets[gset_name] = gene_list
    return converted_genesets