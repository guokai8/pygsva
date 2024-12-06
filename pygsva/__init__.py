# Import necessary components
from enum import Enum
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix, csc_matrix
from scipy.stats import norm, poisson
from scipy.stats import norm, poisson, rankdata  # 添加 rankdata 的导入
from scipy.stats import poisson
from scipy.stats import rankdata
from tqdm import tqdm
from typing import List, Tuple
from typing import List, Union, Tuple
import math
import multiprocessing
import numpy as np
import pandas as pd
import sys
import warnings

from .config import sigma_factor, precompute_resolution,max_precompute,init_cdfs,get_config,update_config  # Assuming config.py contains the GSVAConfig class
from .main import gsva  # Assuming your main entry point is defined here
from .gsvap import gsva_ranks, gsva_scores, gsva_enrichment
from .ssgsea import ssgsea, ssgsea_batched
from .zscore import zscore
from .plage import plage
from .param import gsvaParam, ssgseaParam, zscoreParam, plageParam
from .utils import *  # Assuming utility functions are in utils.py
from .data_loader import load_hsko_data, load_pbmc_data

# Initialize package configuration
__all__ = [
    "init_cdfs","sigma_factor", "precompute_resolution","max_precompute",
    "gsva", "ssgsea","ssgsea_batched","gsva_enrichment", "zscore", "plage",
    "gsvaParam", "ssgseaParam", "zscoreParam", "plageParam",'load_hsko_data', 'load_pbmc_data'
]

# Informative package-level docstring
"""
GSVA Python Package
--------------------
This package provides implementations of GSVA, ssGSEA, PLAGE, and Z-score methods
for gene set enrichment analysis. The configuration system allows users to customize
parameters like SIGMA_FACTOR, PRECOMPUTE_RESOLUTION, and MAX_PRECOMPUTE.

Usage:
    # Access configuration
    import my_gsva_package
    print(my_gsva_package.gsva_config.sigma_factor)

    # Update configuration
    my_gsva_package.set_gsva_config(sigma_factor=5.0)

    # Run GSVA
    results = my_gsva_package.gsva(params)
"""
# Initialize default configuration
precomputed_values=init_cdfs()
is_precomputed = True