from .utils import validate_params, check_for_na_values
import numpy as np
class plageParam:
    def __init__(self, expr_data, gene_sets, min_size=1, max_size=float('inf'), remove_constant=True, remove_nz_constant=True, n_jobs=1, use_sparse=False,verbose=True):
        """
        Constructor for PLAGE parameter object.

        Args:
            expr_data (pd.DataFrame): Expression data with genes as rows and samples as columns.
            gene_sets (dict): Dictionary with pathway names as keys and lists of genes as values.
            min_size (int, default=1): Minimum size for gene sets.
            max_size (float, default=inf): Maximum size for gene sets.
            remove_constant (bool): Whether to remove genes with constant values throughout the samples
            remove_nz_constant (bool): Whether to remove genes with constant non-zero values throughout the samples
            n_jobs: Number of parallel jobs (1 for serial, -1 for all cores)
            use_sparse (bool, default=False): Whether to use sparse computation.
            verbose (bool): Whether to print detailed logs during computation.
        """
        # Validate parameters

        self.expr_data = expr_data
        self.gene_sets = gene_sets
        self.min_size = min_size
        self.max_size = max_size
        self.remove_constant = remove_constant
        self.remove_nz_constant = remove_nz_constant
        self.n_jobs = n_jobs
        self.use_sparse = use_sparse
        self.verbose = verbose

def create_plage_param(expr_data, gene_sets, min_size=1, max_size=float('inf'), remove_constant=True, remove_nz_constant=True, n_jobs=1,use_sparse=False, verbose=True):
    """
    Create a PLAGE parameter object.

    Args:
        expr_data (pd.DataFrame): Expression data.
        gene_sets (dict): Gene sets for pathways.
        min_size (int, default=1): Minimum size for gene sets.
        max_size (float, default=inf): Maximum size for gene sets.
        remove_constant (bool): Whether to remove genes with constant values throughout the samples
        remove_nz_constant (bool): Whether to remove genes with constant non-zero values throughout the samples
        n_jobs: Number of parallel jobs (1 for serial, -1 for all cores)
        use_sparse (bool, default=False): Whether to use sparse computation.
        verbose (bool): Whether to print detailed logs during computation.

    Returns:
        plageParam: Object containing parameters for PLAGE.
    """
    return plageParam(expr_data, gene_sets, min_size, max_size, remove_constant, remove_nz_constant, n_jobs, use_sparse,verbose)

class zscoreParam:
    def __init__(self, expr_data, gene_sets, min_size=1, max_size=float('inf'), remove_constant=True, remove_nz_constant=True, n_jobs=1, use_sparse=False, verbose=True):
        """
        Constructor for Z-score parameter object.

        Args:
            expr_data (pd.DataFrame):: Gene expression DataFrame
            gene_sets: Dict of pathway name to gene list
            min_size (int, default=1): Minimum size for gene sets.
            max_size (float, default=inf): Maximum size for gene sets.
            remove_constant (bool): Whether to remove genes with constant values throughout the samples
            remove_nz_constant (bool): Whether to remove genes with constant non-zero values throughout the samples
            n_jobs: Number of parallel jobs (1 for serial, -1 for all cores)
            use_sparse (bool, default=False): Whether to use sparse computation.
            verbose (bool): Whether to print detailed logs during computation.
        """
        # Validate parameters
        self.expr_data = expr_data
        self.gene_sets = gene_sets
        self.gene_sets = gene_sets
        self.min_size = min_size
        self.max_size = max_size
        self.remove_constant = remove_constant
        self.remove_nz_constant = remove_nz_constant
        self.n_jobs = n_jobs
        self.use_sparse = use_sparse
        self.verbose = verbose
        
def create_zscore_param(expr_data, gene_sets, min_size=1, max_size=float('inf'), remove_constant=True, remove_nz_constant=True, n_jobs=1, use_sparse=False, verbose=True):
    """
    Create a Z-score parameter object.

    Args:
        expr_data (pd.DataFrame): Expression data.
        gene_sets (dict): Gene sets for pathways.
        min_size (int, default=1): Minimum size for gene sets.
        max_size (float, default=inf): Maximum size for gene sets.
        remove_constant (bool): Whether to remove genes with constant values throughout the samples
        remove_nz_constant (bool): Whether to remove genes with constant non-zero values throughout the samples
        n_jobs: Number of parallel jobs (1 for serial, -1 for all cores)
        use_sparse (bool, default=False): Whether to use sparse computation.
        verbose (bool): Whether to print detailed logs during computation.
    Returns:
        zscoreParam: Object containing parameters for Z-score.
    """
    return zscoreParam(expr_data, gene_sets, min_size, max_size, remove_constant, remove_nz_constant, n_jobs, use_sparse, verbose)

class ssgseaParam:
    def __init__(self, expr_data, gene_sets, alpha=0.25, check_na=False, any_na = False, na_use = "everything", min_size=1, max_size=float('inf'), remove_constant=False, remove_nz_constant=False, normalization=True, n_jobs=1, use_sparse=False, verbose=True):
        """
        Constructor for ssGSEA parameter object.

        Args:
            expr_data (pd.DataFrame): Expression data with genes as rows and samples as columns.
            gene_sets (dict): Dictionary with pathway names as keys and lists of genes as values.
            alpha (float): Weighting exponent for ranking metric.
            check_na (str, default='auto'): How to check for NA values.
            any_na (bool): Whether to handle NA values
            na_use (str, default='everything'): How to handle NA values: ["everything", "all.obs", "na.rm"]
            min_size (int, default=1): Minimum size for gene sets.
            max_size (float, default=inf): Maximum size for gene sets.
            normalization (bool): Whether to normalize the enrichment scores.
            remove_constant (bool): Whether to remove genes with constant values throughout the samples
            remove_nz_constant (bool): Whether to remove genes with constant non-zero values throughout the samples
            n_jobs(int): Determine whether to use parallel processing, if -1 with all cpus
            use_sparse (bool, default=False): Whether to use sparse computation.
            verbose (bool): Whether to print detailed logs during computation.
        """
        # Validate parameters
        valid_na_use = ["everything", "all.obs", "na.rm"]
        if na_use not in valid_na_use:
            raise ValueError(f"Invalid value for na_use. Expected one of {valid_na_use}, got '{na_use}'.")
        na_param = check_for_na_values(expr_data, check_na, na_use)
        self.expr_data = expr_data
        self.gene_sets = gene_sets
        self.alpha = alpha
        self.min_size = min_size
        self.max_size = max_size
        self.check_na = check_na
        self.any_na = na_param['any_na']
        self.na_use = na_use
        self.normalization = normalization
        self.remove_constant = remove_constant
        self.remove_nz_constant = remove_nz_constant
        self.n_jobs = n_jobs
        self.use_sparse = use_sparse
        self.verbose = verbose

def create_ssgsea_param(expr_data, gene_sets,  alpha=0.25, check_na=False, any_na = False, na_use = "everything", min_size=1, max_size=float('inf'), normalization=True, remove_constant=False, remove_nz_constant=False, n_jobs=1, use_sparse=False,verbose=True):
    """
    Create an ssGSEA parameter object.

    Args:
            expr_data (pd.DataFrame): Expression data with genes as rows and samples as columns.
            gene_sets (dict): Dictionary with pathway names as keys and lists of genes as values.
            alpha (float): Weighting exponent for ranking metric.
            check_na (str, default='auto'): How to check for NA values.
            any_na (bool): Whether to handle NA values
            na_use (str, default='everything'): How to handle NA values: ["everything", "all.obs", "na.rm"]
            min_size (int, default=1): Minimum size for gene sets.
            max_size (float, default=inf): Maximum size for gene sets.
            normalization (bool): Whether to normalize the enrichment scores.
            remove_constant (bool): Whether to remove genes with constant values throughout the samples
            remove_nz_constant (bool): Whether to remove genes with constant non-zero values throughout the samples
            n_jobs(int): Determine whether to use parallel processing, if -1 with all cpus
            use_sparse (bool, default=False): Whether to use sparse computation.
            verbose (bool): Whether to print detailed logs during computation.

    Returns:
        ssgseaParam: Object containing parameters for ssGSEA.
    """
    return ssgseaParam(expr_data, gene_sets, alpha, normalization, check_na, any_na, na_use, min_size, max_size, normalization, remove_constant, remove_nz_constant, n_jobs,use_sparse,verbose)

class gsvaParam:
    def __init__(self, expr_data, gene_sets, assay=None, annotation=None,
                 min_size=1, max_size=float('inf'),
                 kcdf='auto', kcdf_none_min_sample_size=200,
                 tau=1, max_diff=True, abs_ranking=False,
                 use_sparse=False, check_na='auto',n_jobs=1,
                 na_use='everything'):
        """
        Constructor for GSVA parameter object.

        Args:
            expr_data (numpy.ndarray or scipy.sparse.spmatrix): Expression data matrix.
            gene_sets (dict): Dictionary of gene sets.
            assay (str, optional): Assay name.
            annotation (str, optional): Annotation information.
            min_size (int, default=1): Minimum size for gene sets.
            max_size (float, default=inf): Maximum size for gene sets.
            kcdf (str, default='auto'): Kernel CDF method ('auto', 'Gaussian', 'Poisson', 'none').
            kcdf_none_min_sample_size (int, default=200): Minimum sample size for kernel estimation.
            tau (float, default=1): Tau parameter.
            max_diff (bool, default=True): Whether to use maximum difference.
            abs_ranking (bool, default=False): Whether to use absolute ranking.
            use_sparse (bool, default=True): Whether to use sparse computation.
            check_na (str, default='auto'): How to check for NA values.
            n_jobs(int): Determine whether to use parallel processing, if -1 with all cpus
            na_use (str, default='everything'): How to handle NA values: ["everything", "all.obs", "na.rm"]
        """
        # Validate parameters
        valid_na_use = ["everything", "all.obs", "na.rm"]
        if na_use not in valid_na_use:
            raise ValueError(f"Invalid value for na_use. Expected one of {valid_na_use}, got '{na_use}'.")
        validate_params(expr_data, gene_sets, kcdf, check_na, na_use)
        na_param = check_for_na_values(expr_data, check_na, na_use)

        self.expr_data = expr_data
        self.gene_sets = gene_sets
        self.assay = assay
        self.annotation = annotation
        self.min_size = min_size
        self.max_size = max_size
        self.kcdf = kcdf
        self.kcdf_none_min_sample_size = kcdf_none_min_sample_size
        self.tau = tau
        self.max_diff = max_diff
        self.abs_ranking = abs_ranking
        self.use_sparse = use_sparse
        self.check_na = check_na
        self.did_check_na = na_param['did_check_na']
        self.any_na = na_param['any_na']
        self.n_jobs = n_jobs
        self.na_use = na_use

    def copy(self):
        """
        Create a deep copy of the GSVAParam object.

        Returns:
        --------
        GSVAParam
            A new instance with the same attributes.
        """
        import copy
        return copy.deepcopy(self)
        
def create_gsva_param(expr_data, gene_sets, assay=None, annotation=None,
                     min_size=1, max_size=float('inf'),
                     kcdf='auto', kcdf_none_min_sample_size=200,
                     tau=1, max_diff=True, abs_ranking=False,
                     use_sparse=False, check_na='auto',n_jobs=1,
                     na_use='everything'):
    """
    Create a GSVA parameter object.

    Args:
        expr_data (numpy.ndarray or scipy.sparse.spmatrix): Expression data matrix.
        gene_sets (dict): Dictionary of gene sets.
        assay (str, optional): Assay name.
        annotation (str, optional): Annotation information.
        min_size (int, default=1): Minimum size for gene sets.
        max_size (float, default=inf): Maximum size for gene sets.
        kcdf (str, default='auto'): Kernel CDF method ('auto', 'Gaussian', 'Poisson', 'none').
        kcdf_none_min_sample_size (int, default=200): Minimum sample size for kernel estimation.
        tau (float, default=1): Tau parameter.
        max_diff (bool, default=True): Whether to use maximum difference.
        abs_ranking (bool, default=False): Whether to use absolute ranking.
        use_sparse (bool, default=False): Whether to use sparse computation.
        check_na (str, default='auto'): How to check for NA values.
        na_use (str, default='everything'): How to handle NA values: ["everything", "all.obs", "na.rm"]

    Returns:
        gsvaParam: Object containing parameters for GSVA.
    """
    return gsvaParam(expr_data, gene_sets, assay, annotation, min_size, max_size,
                     kcdf, kcdf_none_min_sample_size, tau, max_diff,
                     abs_ranking, use_sparse, check_na, n_jobs, na_use)
