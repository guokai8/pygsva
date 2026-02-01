
from .utils import *
from joblib import Parallel, delayed
from .config import sigma_factor, precompute_resolution,max_precompute,init_cdfs,get_config,update_config  
import copy
# Assuming config.py contains the GSVAConfig class
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
        
        
def get_gene_sets(obj):
    """
    获取对象中的基因集
    
    Parameters:
    -----------
    obj : dict
        GSVA参数对象或结果对象
    
    Returns:
    --------
    dict
        基因集字典
    """
    if isinstance(obj, dict):
        if 'gene_sets' in obj:
            return obj['gene_sets']
        # 对于结果对象
        if 'filtered_mapped_gene_sets' in obj:
            # 将索引转换回基因名
            if 'filtered_data_matrix' in obj:
                gene_names = obj['filtered_data_matrix'].index
                return indices_to_gene_names(obj['filtered_mapped_gene_sets'], gene_names)
    raise ValueError("The object does not contain information about gene sets.")
    
####
def get_gene_sets(gsva_param):
    """
    获取GSVA参数对象中的基因集
    
    Parameters:
    -----------
    gsva_param : dict
        GSVA参数对象，包含gene_sets键
    
    Returns:
    --------
    dict
        基因集字典
    """
    if not isinstance(gsva_param, dict):
        raise ValueError("Input must be a dictionary")
    
    return gsva_param.gene_sets

def set_gene_sets(gsva_param, new_gene_sets):
    """
    更新GSVA参数对象中的基因集
    
    Parameters:
    -----------
    gsva_param : dict
        GSVA参数对象，包含完整的参数设置
    new_gene_sets : dict
        新的基因集字典
        
    Returns:
    --------
    dict
        更新后的GSVA参数对象
    """
    if not isinstance(gsva_param, dict):
        raise ValueError("Input must be a dictionary")
        
    # 创建新的参数对象（深拷贝以避免修改原始对象）
    new_param = copy.deepcopy(gsva_param)
    new_param.gene_sets = new_gene_sets
    
    return new_param

def indices_to_gene_names(gene_sets_indices, gene_names):
    """    
    Parameters:
    -----------
    gene_sets_indices : dict
    gene_names : list or pd.Index
    
    Returns:
    --------
    dict
    """
    return {
        set_name: [gene_names[idx-1] for idx in indices]  # -1因为indices是1-based
        for set_name, indices in gene_sets_indices.items()
    }

 ## compute gene cdf
## compute gene cdf
def compute_gene_cdf(expr, sample_idxs, Gaussk=True, kernel=True,
                    sparse_output=False, any_na=False,
                    na_use="everything", verbose=True):
    """
    Compute gene CDF values from expression data.
    """
    # Validate na_use parameter
    valid_na_use = ["everything", "all.obs", "na.rm"]
    if na_use not in valid_na_use:
        raise ValueError(f"na_use must be one of {valid_na_use}")

    if isinstance(expr, pd.DataFrame):
        expr = expr.values
    
    # Get dimensions
    n_genes, n_test_samples = expr.shape
    n_density_samples = len(sample_idxs)

    # Check for NA values with all.obs
    if any_na and na_use == "all.obs":
        raise ValueError("Missing values present in the input expression data and 'use=\"all.obs\"'")

    # Initialize result
    gene_cdf = None

    if kernel:
        if isinstance(expr, sparse.csc_matrix):
        #if sparse.issparse(expr) and isinstance(expr, sparse.csc_matrix):
            expr_subset = expr[:, sample_idxs]
            expr_subset_csr = expr_subset.tocsr()
            if sparse_output:
                gene_cdf = kcdfvals_sparse_to_sparse(
                    expr_subset,
                    expr_subset_csr,
                    Gaussk,
                    verbose
                )
            else:
                gene_cdf = kcdfvals_sparse_to_dense(
                    expr_subset,
                    expr_subset_csr,
                    Gaussk,
                    verbose
                )
        elif isinstance(expr, np.ndarray):
            expr_subset = expr[:, sample_idxs]

            # Convert to column-major order and flatten
            expr_subset_t = np.ascontiguousarray(expr_subset.T).flatten(order='F')
            expr_t = np.ascontiguousarray(expr.T).flatten(order='F')

            # Map na_use to integer value
            na_use_map = {"everything": 1, "all.obs": 2, "na.rm": 3}
            na_use_int = na_use_map[na_use]

            # Call matrix density function
            A = matrix_density_vectorized(
                expr_subset_t,
                expr_t,
                n_density_samples,
                n_test_samples,
                n_genes,
                Gaussk,
                any_na,
                na_use_int,
                verbose
            )

            # Reshape and transpose to match R output
            gene_cdf = A.reshape((n_test_samples,n_genes), order='F').T
        else:
            raise ValueError(f"Matrix class {type(expr)} cannot be handled yet.")
    else:
        if sparse.issparse(expr) and isinstance(expr, sparse.csc_matrix):
            expr_subset = expr[:, sample_idxs]
            expr_subset_csr = expr_subset.tocsr()
            if sparse_output:
                gene_cdf = ecdf_sparse_to_sparse(expr_subset, expr_subset_csr, verbose)
            else:
                gene_cdf = ecdf_sparse_to_dense(expr_subset, expr_subset_csr, verbose)
        elif isinstance(expr, np.ndarray):
            expr_subset = expr[:, sample_idxs]
            if any_na:
                gene_cdf = ecdf_dense_to_dense_nas(expr_subset, verbose)
            else:
                gene_cdf = ecdf_dense_to_dense(expr_subset, verbose)
        else:
            raise ValueError(f"Matrix class {type(expr)} cannot be handled yet.")

    return gene_cdf

def compute_gene_cdf_bk(expr, sample_idxs, Gaussk=True, kernel=True,
                    sparse_output=False, any_na=False,
                    na_use="everything", verbose=True):
    """
    Compute gene CDF values from expression data.
    """
    # Validate na_use parameter
    valid_na_use = ["everything", "all.obs", "na.rm"]
    if na_use not in valid_na_use:
        raise ValueError(f"na_use must be one of {valid_na_use}")

    if isinstance(expr, pd.DataFrame):
        expr = expr.values
    
    # Get dimensions
    n_genes, n_test_samples = expr.shape
    n_density_samples = len(sample_idxs)

    # Check for NA values with all.obs
    if any_na and na_use == "all.obs":
        raise ValueError("Missing values present in the input expression data and 'use=\"all.obs\"'")

    # Initialize result
    gene_cdf = None

    if kernel:
        if isinstance(expr, sparse.csc_matrix):
        #if sparse.issparse(expr) and isinstance(expr, sparse.csc_matrix):
            expr_subset = expr[:, sample_idxs]
            expr_subset_csr = expr_subset.tocsr()
            if sparse_output:
                gene_cdf = kcdfvals_sparse_to_sparse(
                    expr_subset,
                    expr_subset_csr,
                    Gaussk,
                    verbose
                )
            else:
                gene_cdf = kcdfvals_sparse_to_dense(
                    expr_subset,
                    expr_subset_csr,
                    Gaussk,
                    verbose
                )
        elif isinstance(expr, np.ndarray):
            expr_subset = expr[:, sample_idxs]

            # Convert to column-major order and flatten
            expr_subset_t = np.ascontiguousarray(expr_subset.T).flatten(order='F')
            expr_t = np.ascontiguousarray(expr.T).flatten(order='F')

            # Map na_use to integer value
            na_use_map = {"everything": 1, "all.obs": 2, "na.rm": 3}
            na_use_int = na_use_map[na_use]

            # Call matrix density function
            A = matrix_density(
                expr_subset_t,
                expr_t,
                n_density_samples,
                n_test_samples,
                n_genes,
                Gaussk,
                any_na,
                na_use_int,
                verbose
            )

            # Reshape and transpose to match R output
            gene_cdf = A.reshape((n_test_samples,n_genes), order='F').T
        else:
            raise ValueError(f"Matrix class {type(expr)} cannot be handled yet.")
    else:
        if isinstance(expr, sparse.csc_matrix):
            expr_subset = expr[:, sample_idxs]
            expr_subset_csr = expr_subset.tocsr()
            if sparse_output:
                gene_cdf = ecdf_sparse_to_sparse(expr_subset, expr_subset_csr, verbose)
            else:
                gene_cdf = ecdf_sparse_to_dense(expr_subset, expr_subset_csr, verbose)
        elif isinstance(expr, np.ndarray):
            expr_subset = expr[:, sample_idxs]
            if any_na:
                gene_cdf = ecdf_dense_to_dense_nas(expr_subset, verbose)
            else:
                gene_cdf = ecdf_dense_to_dense(expr_subset, verbose)
        else:
            raise ValueError(f"Matrix class {type(expr)} cannot be handled yet.")

    return gene_cdf

       
 #### compute GSVA ranks      
def compute_gsva_ranks(expr, kcdf, kcdf_min_ssize,
                       sparse_output, any_na, na_use, verbose,use_sparse,
                       n_jobs=1):
    """
    Computes the GSVA ranks from expression data.

    Parameters:
    -----------
    expr : numpy.ndarray or scipy.sparse.csc_matrix
        Expression data matrix (genes x samples).
    kcdf : str
        Kernel to use ('Gaussian', 'Poisson', etc.).
    kcdf_min_ssize : int
        Minimum sample size for kernel estimation.
    sparse_output : bool
        Whether to return a sparse matrix.
    any_na : bool
        Whether the input contains NA values.
    na_use : str
        How to handle NA values: 'all.obs', 'everything', or 'na.rm'.
    verbose : bool
        Whether to print progress messages.
    use_sparse: bool
        Whether to use sparse mode or not
    n_jobs : int
        Number of parallel jobs to run.

    Returns:
    --------
    R : numpy.ndarray or scipy.sparse.csc_matrix
        Matrix of ranks.
    """
    kcdf_param = parse_kcdf_param(expr, kcdf, kcdf_min_ssize, sparse_output, verbose)
    if use_sparse:
        sparse_output=True
    kernel = kcdf_param['kernel']
    Gaussk = kcdf_param['Gaussk']
    Z = None
    expr = csc_matrix(expr)
    n_genes, n_samples = expr.shape

    # Decide whether to use parallelism
    use_parallel_ecdf = n_jobs > 1 and n_genes > 100 and n_samples > 100

    if use_parallel_ecdf:
        n_chunks = 10  # 10 chunks of (n_genes / 10) rows
        if verbose:
            print(f"Estimating row ECDFs with {n_jobs} cores")

        # Split the data into chunks
        chunk_size = n_genes // n_chunks + (n_genes % n_chunks > 0)
        gene_indices = [range(i, min(i + chunk_size, n_genes)) for i in range(0, n_genes, chunk_size)]

        if verbose:
            gene_indices = tqdm(gene_indices, desc="Estimating row ECDFs")
        else:
            gene_indices = gene_indices

        # Parallel computation of ECDFs
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_gene_cdf)(
                expr[idx, :],
                sample_idxs=range(n_samples),
                Gaussk=Gaussk,
                kernel=kernel,
                sparse_output=sparse_output,
                any_na=any_na,
                na_use=na_use,
                verbose=False
            ) for idx in gene_indices
        )
        Z = np.vstack(results)
    else:
        Z = compute_gene_cdf(expr, range(n_samples), Gaussk, kernel,
                             sparse_output, any_na=any_na, na_use=na_use, verbose=verbose)

    R = None

    if sparse.issparse(Z):  # If Z is a sparse matrix
        if verbose:
            print("Calculating GSVA column ranks")
        R = sparse_column_apply_and_replace(Z, rank_ties_last)
    else:
        n_genes_Z, n_samples_Z = Z.shape
        use_parallel_ranks = n_jobs > 1 and n_genes_Z > 10000 and n_samples_Z > 1000

        if use_parallel_ranks:
            n_chunks = 100  # 100 chunks of (n_samples / 100) columns
            if verbose:
                print("Calculating GSVA column ranks with parallel processing")

            # Split the data into chunks
            chunk_size = n_samples_Z // n_chunks + (n_samples_Z % n_chunks > 0)
            sample_indices = [range(i, min(i + chunk_size, n_samples_Z)) for i in range(0, n_samples_Z, chunk_size)]

            if verbose:
                sample_indices = tqdm(sample_indices, desc="Calculating GSVA column ranks")
            else:
                sample_indices = sample_indices

            # Parallel computation of ranks
            ranked_columns = Parallel(n_jobs=n_jobs)(
                delayed(colRanks)(
                    Z[:, idx],
                    ties_method='last',
                    preserve_shape=True
                ) for idx in sample_indices
            )
            R = np.hstack(ranked_columns)
        else:
            if verbose:
                print("Calculating GSVA column ranks")
            R = colRanks(Z, ties_method='last', preserve_shape=True)

    return R
    
def gsva_ranks(param, use_sparse=False, verbose=True,sparse_output=False, n_jobs=1):
    """
    Calculate GSVA ranks

    Parameters:
    -----------
    param : gsvaParam object
        GSVA parameters from gsvaParam
    verbose : bool
        Whether to print progress messages
    n_jobs : int
        Number of parallel jobs to run

    Returns:
    --------
    dict
        Dictionary containing GSVA ranks and parameters, with expr_data as DataFrame
    """
    import pandas as pd

    if verbose:
        print("Starting GSVA ranks calculation")

    # Get expression data matrix and keep track of index/columns
    data_matrix = param.expr_data
    orig_index = data_matrix.index
    orig_columns = data_matrix.columns
    if(use_sparse):
        expr_data=sparse.csc_matrix(data_matrix)
        mask, filtered_indices = filter_genes(expr_data, remove_constant=True, remove_nz_constant=True)
        filtered_matrix = expr_data[mask]
        sparse_output = True
    else:
        mask, filtered_indices = filter_genes(data_matrix, remove_constant=True, remove_nz_constant=True)
        filtered_matrix = data_matrix.loc[filtered_indices]
    # Filter genes
    new_index = orig_index[mask]
    if verbose and n_jobs > 1:
        print(f"Using parallel processing with {n_jobs} workers")
    
    # Compute GSVA ranks
    ranks = compute_gsva_ranks(
        expr=filtered_matrix,
        kcdf=param.kcdf,
        kcdf_min_ssize=param.kcdf_none_min_sample_size,
        sparse_output=sparse_output,
        any_na=param.any_na,
        na_use=param.na_use,
        verbose=verbose,
        use_sparse=use_sparse,
        n_jobs=n_jobs
    )
    if use_sparse:
        ranks = ranks.toarray()
    # Convert ranks to DataFrame
    ranks_df = pd.DataFrame(ranks, 
                          index=new_index,
                          columns=orig_columns)
    
    # Create ranks parameters with DataFrame
   # ranks_param = param.copy()
  #  ranks_param.update({
  #      'expr_data': ranks_df,
   #     'assay': 'gsvaranks'
  #  })
    ranks_param = copy.deepcopy(param)  # Deep copy to avoid modifying the original object
    ranks_param.expr_data = ranks_df   # Set the new attribute
    ranks_param.assay = 'gsvaranks'    # Set the new attribute
    
    if verbose:
        print("GSVA ranks calculation finished")
    
    return ranks_param

def gsva_rnd_walk(gsetidx: np.ndarray,
                  decordstat: np.ndarray,
                  symrnkstat: np.ndarray,
                  tau: float = 1.0,
                  return_walkstat: bool = False) -> Union[Tuple[float, float], Tuple[np.ndarray, float, float]]:
    """
    Perform random walk for GSVA calculation.
    Optimized version using vectorized numpy operations.

    Parameters:
    -----------
    gsetidx : np.ndarray
        Gene set indices (1-based)
    decordstat : np.ndarray
        Decreasing ordered statistics
    symrnkstat : np.ndarray
        Symmetric rank statistics
    tau : float
        Weighting factor
    return_walkstat : bool
        Whether to return full walk statistics

    Returns:
    --------
    walkstat : np.ndarray, optional
        Full walk statistics
    walkstatpos : float
        Maximum positive deviation
    walkstatneg : float
        Maximum negative deviation
    """
    n = len(decordstat)
    k = len(gsetidx)

    # Get ranks of genes in gene set (convert to 0-based indexing)
    gsetidx_0based = gsetidx - 1
    gsetrnk = decordstat[gsetidx_0based].astype(int)

    # Initialize step CDFs
    stepcdfingeneset = np.zeros(n)
    stepcdfoutgeneset = np.ones(n)

    # Vectorized: Set values for genes in gene set
    rank_indices = gsetrnk - 1  # Convert to 0-based indexing

    if tau == 1:
        stepcdfingeneset[rank_indices] = symrnkstat[gsetidx_0based]
    else:
        stepcdfingeneset[rank_indices] = np.power(symrnkstat[gsetidx_0based], tau)

    stepcdfoutgeneset[rank_indices] = 0

    # Compute cumulative sums
    stepcdfingeneset = np.cumsum(stepcdfingeneset)
    stepcdfoutgeneset = np.cumsum(stepcdfoutgeneset)

    walkstatpos = walkstatneg = np.nan
    walkstat = None

    if stepcdfingeneset[-1] > 0 and stepcdfoutgeneset[-1] > 0:
        # Calculate walking statistic
        wlkstat = (stepcdfingeneset / stepcdfingeneset[-1] -
                  stepcdfoutgeneset / stepcdfoutgeneset[-1])

        if return_walkstat:
            walkstat = wlkstat

        # Get maximum deviations
        walkstatpos = np.max(wlkstat)
        walkstatneg = np.min(wlkstat)

    if return_walkstat:
        return walkstat if walkstat is not None else np.full(n, np.nan), walkstatpos, walkstatneg
    return walkstatpos, walkstatneg

def gsva_rnd_walk_nas(gsetidx: np.ndarray,
                      decordstat: np.ndarray,
                      symrnkstat: np.ndarray,
                      tau: float = 1.0,
                      na_use: int = 3,
                      minsize: int = 5,
                      return_walkstat: bool = False) -> Tuple[Union[np.ndarray, None], float, float, bool]:
    """
    Perform random walk for GSVA calculation with NA handling.
    Optimized version using vectorized numpy operations.
    """
    n = len(decordstat)
    k = len(gsetidx)
    walkstat = None
    walkstatpos = walkstatneg = np.nan
    wna = False

    # Ensure indices are integers
    gsetidx = gsetidx.astype(int)

    # Remove NAs from gene set (convert to 0-based for indexing)
    valid_mask = ~np.isnan(decordstat[gsetidx - 1])
    gsetidx_wonas = gsetidx[valid_mask]
    k_notna = len(gsetidx_wonas)

    # Handle NAs based on na_use parameter
    if k_notna < k and na_use < 3:  # everything or all.obs
        if return_walkstat:
            return np.full(n, np.nan), np.nan, np.nan, False
        return None, np.nan, np.nan, False

    if k_notna >= minsize:
        # Get ranks of non-NA genes (convert to 0-based, ensure integer)
        gsetidx_0based = gsetidx_wonas - 1
        gsetrnk = decordstat[gsetidx_0based].astype(int)

        # Initialize step CDFs
        stepcdfingeneset = np.zeros(n)
        stepcdfoutgeneset = np.ones(n)

        # Vectorized: Set values for genes in gene set
        rank_indices = gsetrnk - 1  # Convert to 0-based indexing

        if tau == 1:
            stepcdfingeneset[rank_indices] = symrnkstat[gsetidx_0based]
        else:
            stepcdfingeneset[rank_indices] = np.power(symrnkstat[gsetidx_0based], tau)

        stepcdfoutgeneset[rank_indices] = 0

        # Compute cumulative sums
        stepcdfingeneset = np.cumsum(stepcdfingeneset)
        stepcdfoutgeneset = np.cumsum(stepcdfoutgeneset)

        if stepcdfingeneset[-1] > 0 and stepcdfoutgeneset[-1] > 0:
            # Calculate walking statistic
            wlkstat = (stepcdfingeneset / stepcdfingeneset[-1] -
                      stepcdfoutgeneset / stepcdfoutgeneset[-1])

            if return_walkstat:
                walkstat = wlkstat

            # Get maximum deviations
            walkstatpos = np.max(wlkstat)
            walkstatneg = np.min(wlkstat)
    else:
        wna = True

    if return_walkstat:
        return (walkstat if walkstat is not None else np.full(n, np.nan),
                walkstatpos, walkstatneg, wna)
    return None, walkstatpos, walkstatneg, wna

def gsva_score_genesets(genesetsidx: List[List[int]], 
                       decordstat: np.ndarray,
                       symrnkstat: np.ndarray,
                       maxdiff: bool = False,
                       absrnk: bool = False,
                       tau: float = 1.0,
                       anyna: bool = False,
                       nause: int = 3,
                       minsize: int = 2) -> Tuple[np.ndarray, str]:
    """
    Python implementation of GSVA's gsva_score_genesets_R function
    """
    m = len(genesetsidx)  # Number of gene sets
    n = len(decordstat)   # Number of genes
    
    # Initialize result array with NAs
    es = np.full(m, np.nan)
    
    # Track NA warning
    wna = 0
    abort = False
    
    # Process each gene set
    for i in range(m):
        gsetidx = np.array(genesetsidx[i])
        k = len(gsetidx)
        
        # Calculate random walk statistics
        if anyna:
            _, walkstatpos, walkstatneg, curr_wna = gsva_rnd_walk_nas(
                gsetidx=gsetidx,
                decordstat=decordstat,
                symrnkstat=symrnkstat,
                tau=tau,
                na_use=nause,
                minsize=minsize,
                return_walkstat=False
            )
            wna = wna or curr_wna
        else:
            walkstatpos, walkstatneg = gsva_rnd_walk(
                gsetidx=gsetidx,
                decordstat=decordstat,
                symrnkstat=symrnkstat,
                tau=tau,
                return_walkstat=False
            )
        
        # Calculate enrichment score
        if not anyna or (not np.isnan(walkstatpos) and not np.isnan(walkstatneg)):
            if maxdiff:
                es[i] = walkstatpos + walkstatneg
                if absrnk:
                    es[i] = walkstatpos - walkstatneg
            else:
                es[i] = walkstatpos if walkstatpos > abs(walkstatneg) else walkstatneg
        else:
            if anyna and (np.isnan(walkstatpos) or np.isnan(walkstatneg)) and nause == 2:
                abort = True
                break
    
    # Determine class attribute
    class_attr = None
    if anyna:
        if nause == 2 and abort:
            class_attr = "abort"
        elif nause == 3 and wna:
            class_attr = "wna"
    
    return es, class_attr


def process_column_scores(ranks, gene_sets_idx, max_diff, abs_ranking, tau,
                         use_sparse, any_na, na_use, min_size):
    """
    Process a single column for GSVA scores
    """
    # Calculate rank statistics
    if any_na:
        rnkstats = ranks2stats_nas(ranks, use_sparse)
    else:
        rnkstats = ranks2stats(ranks, use_sparse)
    
    n = len(ranks)
    dos = rnkstats['dos']
    srs = rnkstats['srs']
    
    # Ensure dos values don't exceed n
    dos = np.minimum(dos, n)
    
    # 转换gene_sets_idx为列表格式
    gene_sets_list = [np.array(indices) for indices in gene_sets_idx.values()]
    
    # Calculate scores
    scores, _ = gsva_score_genesets(
        gene_sets_list,
        decordstat=dos,
        symrnkstat=srs,
        maxdiff=max_diff,
        absrnk=abs_ranking,
        tau=tau,
        anyna=any_na,
        nause={"everything": 1, "all.obs": 2, "na.rm": 3}[na_use],
        minsize=min_size
    )
    
    return scores

def compute_gsva_scores(R, gene_sets_idx, tau, max_diff, abs_ranking,
                       use_sparse=False, any_na=False, na_use="everything", 
                       min_size=1, verbose=True, n_jobs=1):
    """
    Compute GSVA scores
    """
    from joblib import Parallel, delayed
    
    # Get dimensions
    if isinstance(R, pd.DataFrame):
        n = R.shape[1]
    else:
        n = R.shape[1]
        
    if not sparse.issparse(R):
        use_sparse = False
    
    # Track warnings
    wna = False
    
    if n > 10 and n_jobs > 1:
        if verbose:
            print(f"Calculating GSVA scores with {n_jobs} cores")
            
        if isinstance(R, pd.DataFrame):
            es = Parallel(n_jobs=n_jobs)(
                delayed(process_column_scores)(
                    R.iloc[:, j].values, gene_sets_idx, max_diff, abs_ranking, tau,
                    use_sparse, any_na, na_use, min_size
                )
                for j in range(n)
            )
        else:
            es = Parallel(n_jobs=n_jobs)(
                delayed(lambda j: process_column_scores(
                    R[:, j], gene_sets_idx, max_diff, abs_ranking, tau,
                    use_sparse, any_na, na_use, min_size
                ))
                for j in range(n)
            )
            
    else:
        if verbose:
            print("Calculating GSVA scores")
        es = []
        for j in range(n):
            if isinstance(R, pd.DataFrame):
                scores = process_column_scores(
                    R.iloc[:, j].values, gene_sets_idx, max_diff, abs_ranking, tau,
                    use_sparse, any_na, na_use, min_size
                )
            else:
                scores = process_column_scores(
                    R[:, j], gene_sets_idx, max_diff, abs_ranking, tau,
                    use_sparse, any_na, na_use, min_size
                )
            es.append(scores)
            if verbose and (j + 1) % 10 == 0:
                print(f"Processed {j + 1}/{n} samples")
    
    es = np.column_stack(es)
    
    if verbose and any_na and na_use == "na.rm" and wna:
        print(f"Warning: NA enrichment scores in gene sets with less than {min_size} genes "
              "after removing missing values")
    
    return es


def gsva_scores(param, verbose=True, use_sparse=False, n_jobs=1):
    """
    Calculate GSVA scores from ranks
    
    Parameters:
    -----------
    param : gsvaParam object
        A gsvaParam object containing GSVA parameters including:
        - expr_data: Expression data matrix
        - gene_sets: Gene sets
        - tau: Weighting factor
        - max_diff: Whether to use maximum difference
        - abs_ranking: Whether to use absolute ranking
        - use_sparse: Whether to use sparse computation
        - any_na: Whether there are NA values
        - na_use: How to handle NA values
        - min_size: Minimum size for gene sets
    verbose : bool
        Whether to print progress messages
    n_jobs : int
        Number of parallel jobs to run
        
    Returns:
    --------
    dict
        Dictionary containing:
        - expr_data: Matrix of GSVA enrichment scores
        - gene_sets: Gene sets used in calculations
    """
    if verbose:
        try:
            import pkg_resources
            version = pkg_resources.get_distribution("gsva").version
            print(f"GSVA version {version}")
        except:
            pass

    # Check input data
    expr_data = param.expr_data
    check_rownames(expr_data)
    
    # Filter genes and map gene sets
    filtered_result = filter_and_map_genes_and_gene_sets(
        param=param,
        remove_constant=True,
        remove_nz_constant=True,
        filtered_gene=False,
        verbose=verbose
    )
    filtered_data_matrix = filtered_result['filtered_data_matrix']
    filtered_mapped_genesets = filtered_result['filtered_mapped_gene_sets']
    
    if n_jobs > 1 and verbose:
        print(f"Using parallel processing with {n_jobs} workers")
    
    if verbose:
        print("Calculating GSVA scores")
    
    # Compute GSVA scores
    gsva_es = compute_gsva_scores(
        R=filtered_data_matrix,
        gene_sets_idx=filtered_mapped_genesets,
        tau=param.tau,
        max_diff=param.max_diff,
        abs_ranking=param.abs_ranking,
        use_sparse=use_sparse,
        any_na=param.any_na,
        na_use=param.na_use,
        min_size=param.min_size,
        verbose=verbose,
        n_jobs=n_jobs
    )
    
    # Add row and column names
    if not isinstance(gsva_es, pd.DataFrame):
        gsva_es = pd.DataFrame(
            gsva_es,
            index=list(filtered_mapped_genesets.keys()),
            columns=filtered_data_matrix.columns
        )
    
    # Convert gene set indices to names
    gene_sets = {
        set_name: [filtered_data_matrix.index[i-1] for i in indices]
        for set_name, indices in filtered_mapped_genesets.items()
    }
    
    if verbose:
        print("Calculations finished")
    
    return {
        'expr_data': gsva_es,
        'gene_sets': gene_sets,
        'filtered_mapped_gene_sets': filtered_mapped_genesets
    }

def gsva_enrichment(param, column=0, gene_set=0, use_sparse=False,plot="auto"):
    """
    计算并可视化GSVA富集分析
    """
    # 获取基因集
    gene_sets = param.gene_sets
    
    # 检查基因集参数
    if isinstance(gene_set, str):
        if gene_set not in gene_sets:
            raise ValueError(f"Gene set {gene_set} not found in parameters")
    elif isinstance(gene_set, (int, float)):
        if gene_set < 0 or gene_set >= len(gene_sets):
            raise ValueError(f"Gene set index must be between 0 and {len(gene_sets)-1}")
        gene_set = list(gene_sets.keys())[gene_set]
    else:
        raise ValueError("gene_set must be string or integer")
    
    # 只过滤和映射单个基因集
    single_set_param = copy.deepcopy(param)
    single_set_param.gene_sets = {gene_set: gene_sets[gene_set]}
    
    # 过滤和映射基因集
    filtered_result = filter_and_map_genes_and_gene_sets(
        param=single_set_param,
        remove_constant=True,
        remove_nz_constant=True,
        verbose=False
    )
    
    gene_set_idx = filtered_result['filtered_mapped_gene_sets'][gene_set]
    gene_set_idx = np.array(gene_set_idx)
    
    # 计算富集数据 - 使用过滤后的数据矩阵
    edata = gsva_enrichment_data(
        R=filtered_result['filtered_data_matrix'],
        column=column,
        gene_set_idx=gene_set_idx,
        tau=param.tau,
        max_diff=param.max_diff,
        abs_ranking=param.abs_ranking,
        use_sparse=use_sparse
    )
    
    if plot == "no" or (plot == "auto" and not hasattr(sys, 'ps1')):
        return edata
        
    return edata

def gsva_enrichment_data(R, column, gene_set_idx, tau=1, max_diff=True,
                        abs_ranking=False, use_sparse=False):
    """
    计算GSVA富集数据
    """
    if sparse.issparse(R):
        use_sparse = True
        
    # 计算排序统计
    rnkstats = ranks2stats(R.iloc[:, column], use_sparse)
    
    # 确保gene_set_idx是np.array并且是1-based
    gene_set_idx = np.array(gene_set_idx, dtype=int)
    if gene_set_idx.min() == 0:
        gene_set_idx += 1
        
    # 使用rnkstats['dos']而不是排序的indices
    walkstat_res = gsva_rnd_walk(
        gene_set_idx, 
        rnkstats['dos'],
        rnkstats['srs'],
        tau,
        True
    )
    
    walk_stat = walkstat_res[0] if isinstance(walkstat_res, tuple) else walkstat_res
    
    # R的顺序是按照rnkstats['dos']排序的
    edat = pd.DataFrame({
        'rank': np.arange(1, R.shape[0] + 1),  # 等价于 seq.int(nrow(R))
        'stat': walk_stat
    })

# 初始化 edat 的行索引（模拟 R 中默认的行名）
    edat.index = np.arange(1, len(edat) + 1)

# 将 rnkstats_dos 转换为 0 索引（因为Python的索引从0开始）
    rank_order = np.array(rnkstats['dos']) - 1

# 按照 rnkstats_dos 指定的位置替换行名为 R 的行名
    edat.index = edat.index.astype(object)  # 确保索引可被修改为字符串
    edat.index.values[rank_order] = R.index  # R.index 即 rownames(R)
    ordered_stats = walk_stat
    # 获取基因集的排序
    gsetrnk = [rnkstats['dos'][i-1] for i in gene_set_idx]
    
    # 在ordered_stats中计算最大和最小值
    maxPos = float(max(0, np.max(ordered_stats)))
    maxNeg = float(min(0, np.min(ordered_stats)))
    whichMaxPos = int(np.argmax(ordered_stats) + 1) if maxPos > 0 else np.nan
    whichMaxNeg = int(np.argmin(ordered_stats) + 1) if maxNeg < 0 else np.nan
    
    # 处理leading edge，使用rank_order
    lepos = []
    leneg = []
    for idx in gene_set_idx:
        dos_rank = rnkstats['dos'][idx-1]
        if not np.isnan(whichMaxPos) and dos_rank <= whichMaxPos:
            lepos.append(R.index[idx-1])
        if not np.isnan(whichMaxNeg) and dos_rank >= whichMaxNeg:
            leneg.append(R.index[idx-1])
    
    # 计算得分
    if max_diff:
        score = maxPos + (maxNeg if not abs_ranking else -maxNeg)
    else:
        score = maxPos if maxPos > abs(maxNeg) else maxNeg
    
    return {
        'stats': edat,
        'gsetrnk': gsetrnk,
        'maxPos': maxPos,
        'whichMaxPos': whichMaxPos,
        'maxNeg': maxNeg,
        'whichMaxNeg': whichMaxNeg,
        'leadingEdgePos': lepos,
        'leadingEdgeNeg': leneg,
        'score': score,
        'tau': tau,
        'maxDiff': max_diff,
        'absRanking': abs_ranking,
        'sparse': use_sparse
    }


