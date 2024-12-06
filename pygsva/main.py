from .gsvap import gsva_ranks, gsva_scores, gsva_enrichment
from .ssgsea import ssgsea, ssgsea_batched
from .zscore import zscore
from .plage import plage
from .param import gsvaParam, ssgseaParam, zscoreParam, plageParam
from .utils import *  # Assuming utility functions are in utils.py
from scipy import sparse 
def gsva(param, verbose=False, cpus=1, batch=False, chunk_size=1000, use_sparse=False):
    """
    Unified GSVA function supporting multiple algorithms and parallel computation.

    Parameters:
    -----------
    param : gsvaParam, plageParam, zscoreParam, or ssgseaParam
        Parameter object containing data and method-specific parameters.
    verbose : bool, default=False
        Whether to print detailed logs.
    cpus : int, default=1
        Number of CPUs to use for parallelization.
    batch : bool, optional
        Whether to use batch mode computation (for extremely large data).
        batch mode was True with GSVA methods
    chunk_size : int, default=1000
        Size of each chunk for batch processing.

    Returns:
    --------
    Results specific to the algorithm used (GSVA, PLAGE, Z-score, or ssGSEA).
    """
    if isinstance(param, gsvaParam):
        if verbose:
            print("Running GSVA algorithm")
        # Calculate ranks with n_chunks if parallel
        if cpus > 1:
            param.n_jobs = cpus
        use_sparse = use_sparse               
        ranks_param = gsva_ranks(
            param,
            use_sparse = param.use_sparse,
            verbose=verbose,
            n_jobs=param.n_jobs
        )
        # Calculate scores with n_chunks
        scores = gsva_scores(
            ranks_param,
            verbose=verbose,
            use_sparse = param.use_sparse,
            n_jobs=param.n_jobs
        )
        return scores['expr_data']

    elif isinstance(param, plageParam):
        if verbose:
            print("Running PLAGE algorithm")
        if cpus > 1:
            param.n_jobs = cpus
        return plage(param.expr_data, param.gene_sets,param.min_size,param.max_size, param.remove_constant, param.remove_nz_constant,param.n_jobs, param.use_sparse,param.verbose)

    elif isinstance(param, zscoreParam):
        if verbose:
            print("Running Z-score algorithm")
        if cpus > 1:
            param.n_jobs = cpus
        return zscore(param.expr_data, param.gene_sets, param.min_size, param.max_size, param.remove_constant, param.remove_nz_constant, param.n_jobs, param.use_sparse, param.verbose)

    elif isinstance(param, ssgseaParam):
        if verbose:
            print("Running ssGSEA algorithm")
        if cpus > 1:
            param.n_jobs = cpus
        if batch:
            return ssgsea_batched(expr_df=param.expr_data, gene_sets=param.gene_sets, alpha=param.alpha, normalization=param.normalization, check_na=param.check_na, any_na=param.any_na, na_use=param.na_use, 
                   min_size=param.min_size, max_size=param.max_size,remove_constant=param.remove_constant, remove_nz_constant=param.remove_nz_constant, n_jobs=param.n_jobs, verbose=param.verbose, chunk_size=chunk_size,use_sparse = param.use_sparse) 
        else:
            return ssgsea(param.expr_data, param.gene_sets, param.alpha, param.normalization, param.check_na, param.any_na, param.na_use, 
                   param.min_size, param.max_size,param.remove_constant, param.remove_nz_constant, param.n_jobs,param.use_sparse,param.verbose)

    else:
        raise ValueError("Unsupported parameter type provided to gsva function")