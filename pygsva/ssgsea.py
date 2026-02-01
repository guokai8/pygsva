import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from enum import Enum
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
from joblib import Parallel, delayed
import gc
from typing import Dict, List, Optional
from dataclasses import dataclass
from scipy.stats import rankdata
from .utils import colRanks, filter_genes

def check_rownames(expr):
    if expr.index is None or expr.index.empty:
        raise ValueError("The input assay object doesn't have rownames")
    elif expr.index.has_duplicates:
        raise ValueError("The input assay object has duplicated rownames")

def fast_rnd_walk(g_set_idx, gene_ranking, j, Ra):
    """
    Python implementation of fastRndWalk function with proper NA and index handling
    
    Parameters:
    g_set_idx: array-like, indices of genes in the gene set (1-based from R)
    gene_ranking: array-like, ranked list of gene indices (0-based)
    j: int, column index for Ra matrix (0-based)
    Ra: array-like, expression matrix
    """
    n = len(gene_ranking)
    
    # Convert inputs to numpy arrays if they aren't already
    g_set_idx = np.array(g_set_idx).astype(int)
    gene_ranking = np.array(gene_ranking)
    
    # Remove NaN values from g_set_idx
   # mask = ~np.isnan(g_set_idx)
   # valid_idx = g_set_idx[mask].astype(int)
    
    k = len(g_set_idx)
    
    # Calculate step CDF in gene set
    indices = gene_ranking[g_set_idx - 1]  # Convert to 0-based indexing
    weights = Ra[indices, j]
    positions = n - g_set_idx + 1  # Keep 1-based for positions
    
    step_cdf_in_gene_set = np.sum(weights * positions) / np.sum(weights)
    
    # Calculate step CDF out of gene set
    step_cdf_out_gene_set = (n * (n + 1) / 2 - np.sum(positions)) / (n - k)
    
    # Calculate walk statistic
    walk_stat = step_cdf_in_gene_set - step_cdf_out_gene_set
    
    return walk_stat


from typing import Union

def fast_rnd_walkRm(g_set_idx: np.ndarray, 
                     gene_ranking: np.ndarray, 
                     j: int, 
                     Ra: np.ndarray,
                     any_na: bool = False,
                     na_use: str = "everything",
                     min_size: int = 1) -> float:
    """
    Python implementation of fastRndWalkNArm function with proper index handling
    
    Parameters:
    g_set_idx (np.ndarray): Indices of genes in the gene set (1-based from R)
    gene_ranking (np.ndarray): Ranked list of gene indices (0-based)
    j (int): Column index for Ra matrix (0-based)
    Ra (np.ndarray): Matrix of expression values
    any_na (bool): Whether to handle NA values
    na_use (str): How to handle NA values ("everything" or "na.rm")
    min_size (int): Minimum size requirement for gene set
    
    Returns:
    float: Walk statistic or np.nan if conditions not met
    """
    n = len(gene_ranking)
    
    # Handle NA values if requested
    if any_na and na_use == "na.rm":
        # Filter out NaN values
        g_set_idx = g_set_idx[~np.isnan(g_set_idx)]
    
    k = len(g_set_idx)
    walk_stat = np.nan
    
    if k >= min_size:
        # Important: g_set_idx is 1-based from R, so subtract 1 for Python indexing
        indices = gene_ranking[g_set_idx.astype(int) - 1]
        
        # Calculate Ra values for the gene set
        # gene_ranking[indices] gives positions, but we need to convert to 0-based
        Ra_values = Ra[indices, j]
        
        # Calculate positions (keep 1-based for consistency with R)
        positions = n - g_set_idx + 1
        
        # Calculate step CDF in gene set
        step_cdf_in_gene_set = np.sum(Ra_values * positions) / np.sum(Ra_values)
        
        # Calculate step CDF out of gene set
        step_cdf_out_gene_set = (n * (n + 1) / 2 - np.sum(positions)) / (n - k)
        
        # Calculate walk statistic
        walk_stat = step_cdf_in_gene_set - step_cdf_out_gene_set
    
    return walk_stat    


    
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

    
def order_value(x, decreasing=True):
    """
    Optimized implementation of R's order function behavior.
    Returns indices that would sort the array.
    """
    if decreasing:
        # For decreasing order, negate values and use stable sort
        return np.argsort(-x, kind='stable')
    else:
        return np.argsort(x, kind='stable') 
    
def ssgsea(expr_df, gene_sets, alpha=0.25, normalization=True, check_na=False, any_na = False, na_use = "everything", min_size=1, max_size=np.inf,remove_constant=True, remove_nz_constant=True, n_jobs=1,use_sparse=False, verbose=True):
    """
    Args:
    expr_df (pd.DataFrame): Expression data with genes as rows and samples as columns.
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


    expr_df_values = filtered_result['filtered_data_matrix']
    genesets = filtered_result['filtered_mapped_gene_sets']
    n_genes, n_samples = expr_df_values.shape

    R = colRanks(expr_df_values.values, ties_method='average')
    R = R.astype(int)
    Ra = np.abs(R) ** alpha

    # Pre-convert gene sets to numpy arrays for faster access
    genesets_arrays = {name: np.array(genes) for name, genes in genesets.items()}
    geneset_names = list(genesets.keys())

    def process_sample(j):
        if any_na and na_use == "na.rm":
            gene_ranking = order_value(R[:, j], decreasing=True) + 1
            valid_mask = ~np.isnan(R[:, j])
            gene_ranking = gene_ranking[valid_mask]

            # Vectorized gene set rank index computation
            gene_ranking_set = set(gene_ranking)
            geneSetsRankIdx = {}
            for name, genes in genesets_arrays.items():
                # Use set intersection for faster lookup
                mask = np.isin(gene_ranking, genes)
                geneSetsRankIdx[name] = np.where(mask)[0] + 1

            es_sample = []
            for name in geneset_names:
                genes = geneSetsRankIdx[name]
                es_value = fast_rnd_walkRm(
                    g_set_idx=genes,
                    gene_ranking=gene_ranking-1,
                    j=j,
                    Ra=Ra,
                    any_na=any_na,
                    na_use=na_use,
                    min_size=min_size
                )
                es_sample.append(es_value)
        else:
            gene_ranking = order_value(R[:, j], decreasing=True) + 1

            # Vectorized gene set rank index computation
            geneSetsRankIdx = {}
            for name, genes in genesets_arrays.items():
                mask = np.isin(gene_ranking, genes)
                geneSetsRankIdx[name] = np.where(mask)[0] + 1

            es_sample = []
            for name in geneset_names:
                genes = geneSetsRankIdx[name]
                es_value = fast_rnd_walk(
                    g_set_idx=genes,
                    gene_ranking=gene_ranking-1,
                    j=j,
                    Ra=Ra
                )
                es_sample.append(es_value)
        return es_sample

    if n_samples > 10 and n_jobs != 1:
        n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        if verbose:
            print(f"Processing samples in parallel using {n_jobs} cores")

        with Parallel(n_jobs=n_jobs) as parallel:
            iterator = tqdm(range(n_samples), desc="Calculating ssGSEA scores") if verbose else range(n_samples)
            es = parallel(delayed(process_sample)(j) for j in iterator)
    else:
        iterator = tqdm(range(n_samples), desc="Calculating ssGSEA scores") if verbose else range(n_samples)
        es = [process_sample(j) for j in iterator]

    es = np.column_stack(es)

    if normalization:
        if verbose:
            print("Normalizing ssGSEA scores")
        if any_na:
            min_val = np.nanmin(es)
            max_val = np.nanmax(es)
        else:
            min_val = np.min(es)
            max_val = np.max(es)

        if np.isnan(min_val) or np.isnan(max_val) or not np.isfinite(min_val) or not np.isfinite(max_val):
            raise ValueError("Cannot calculate normalizing factor for the enrichment scores")

        score_range = max_val - min_val
        if score_range != 0:
            es = es[:, :n_samples] / score_range

    es = es.reshape(1, -1) if len(gene_sets) == 1 else es
    return pd.DataFrame(es, index=list(genesets.keys()), columns=expr_df.columns)
    


def ssgsea_batched(expr_df, gene_sets, alpha=0.25, normalization=True, check_na=False, any_na=False,
                   na_use="everything", min_size=1, max_size=np.inf, remove_constant=True,
                   remove_nz_constant=True, n_jobs=1, use_sparse=False, chunk_size=1000, verbose=True):
    """
    Args:
    expr_df (pd.DataFrame): Expression data with genes as rows and samples as columns.
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
    chunk_size(int): chunk size to use
    verbose (bool): Whether to print detailed logs during computation.
    """
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

    expr_df_values = filtered_result['filtered_data_matrix']
    genesets = filtered_result['filtered_mapped_gene_sets']
    n_genes, n_samples = expr_df_values.shape

    R = colRanks(expr_df_values.values, ties_method='average')
    R = R.astype(int)
    Ra = np.abs(R) ** alpha

    # Pre-convert gene sets to numpy arrays for faster access
    genesets_arrays = {name: np.array(genes) for name, genes in genesets.items()}
    geneset_names = list(genesets.keys())

    def process_chunk(start_idx, end_idx):
        chunk_results = []
        for j in range(start_idx, end_idx):
            if any_na and na_use == "na.rm":
                gene_ranking = order_value(R[:, j], decreasing=True) + 1
                gene_ranking = gene_ranking[~np.isnan(R[:, j])]
            else:
                gene_ranking = order_value(R[:, j], decreasing=True) + 1

            # Vectorized gene set rank index computation
            geneSetsRankIdx = {}
            for name, genes in genesets_arrays.items():
                mask = np.isin(gene_ranking, genes)
                geneSetsRankIdx[name] = np.where(mask)[0] + 1

            es_sample = []
            for name in geneset_names:
                genes = geneSetsRankIdx[name]
                if any_na and na_use == "na.rm":
                    es_value = fast_rnd_walkRm(genes, gene_ranking-1, j, Ra, any_na, na_use, min_size)
                else:
                    es_value = fast_rnd_walk(genes, gene_ranking-1, j, Ra)
                es_sample.append(es_value)
            chunk_results.append(es_sample)
        return chunk_results

    # Split into chunks for parallel processing
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    chunk_ranges = [(i * chunk_size, min((i + 1) * chunk_size, n_samples))
                   for i in range(n_chunks)]

    if n_jobs != 1:
        with Parallel(n_jobs=n_jobs) as parallel:
            if verbose:
                print(f"Processing {n_chunks} chunks using {n_jobs} cores")
            all_results = parallel(delayed(process_chunk)(start, end)
                                 for start, end in tqdm(chunk_ranges) if verbose)
    else:
        iterator = tqdm(chunk_ranges, desc="Processing chunks") if verbose else chunk_ranges
        all_results = [process_chunk(start, end) for start, end in iterator]

    # Merge results
    es = np.column_stack([result for chunk in all_results for result in chunk])

    if normalization:
        if verbose:
            print("Normalizing scores")
        min_val = np.nanmin(es) if any_na else np.min(es)
        max_val = np.nanmax(es) if any_na else np.max(es)
        if np.isnan(min_val) or np.isnan(max_val) or not np.isfinite(min_val) or not np.isfinite(max_val):
            raise ValueError("Cannot calculate normalizing factor")
        score_range = max_val - min_val
        if score_range != 0:
            es = es[:, :n_samples] / score_range

    es = es.reshape(1, -1) if len(gene_sets) == 1 else es
    return pd.DataFrame(es, index=list(genesets.keys()), columns=expr_df.columns)

