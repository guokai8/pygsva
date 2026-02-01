# pygsva: Gene Set Variation Analysis in Python

[![Python application](https://github.com/guokai8/pygsva/actions/workflows/python-app.yml/badge.svg)](https://github.com/guokai8/pygsva/actions/workflows/python-app.yml)
[![PyPI version](https://badge.fury.io/py/pygsva.svg)](https://badge.fury.io/py/pygsva)

Gene Set Variation Analysis (GSVA) is a powerful gene set enrichment method designed for single-sample analysis. It enables pathway-centric analyses of molecular data by shifting the functional unit of analysis from individual genes to gene sets. This approach is particularly useful for bulk microarray, RNA-seq, and other molecular profiling data types, providing a pathway-level view of biological activity.

## Overview

GSVA transforms an input gene-by-sample expression data matrix into a gene-set-by-sample expression data matrix, representing pathway activities. This transformed data can then be utilized with classical analytical methods such as:

- **Differential Expression**
- **Classification**
- **Survival Analysis**
- **Clustering**
- **Correlation Analysis**

Additionally, GSVA enables pathway comparisons with other molecular data types, such as microRNA expression, binding data, copy-number variation (CNV), or single nucleotide polymorphisms (SNPs).

## Performance

Version 0.2.0 includes significant performance optimizations with vectorized NumPy operations:

| Method | Small (1K×100) | Medium (5K×200) | Large (10K×500) |
|--------|----------------|-----------------|-----------------|
| GSVA (Gaussian) | 0.5s | 4.2s | 31.6s |
| GSVA (none) | 0.2s | 1.5s | 10.9s |
| ssGSEA | 0.2s | 1.1s | 7.1s |
| PLAGE | 0.4s | 2.1s | 7.2s |
| Z-score | 0.1s | 0.5s | 1.1s |

For very large datasets (19,000 genes × 1,500 samples), GSVA with `kcdf='Gaussian'` completes in ~4 minutes (previously 52 minutes).

## Methods

The `pygsva` package provides Python implementations of four single-sample gene set enrichment methods:

### 1. **PLAGE** (Pathway Level Analysis of Gene Expression)
   - **Reference**: Tomfohr, Lu, and Kepler (2005)
   - Standardizes expression profiles over the samples.
   - Performs Singular Value Decomposition (SVD) on each gene set.
   - The coefficients of the first right-singular vector are returned as pathway activity estimates.

### 2. **Z-Score Method**
   - **Reference**: Lee et al. (2008)
   - Standardizes expression profiles over the samples.
   - Combines standardized values for each gene in a gene set.

### 3. **ssGSEA** (Single-Sample Gene Set Enrichment Analysis)
   - **Reference**: Barbie et al. (2009)
   - Calculates enrichment scores as the normalized difference in empirical cumulative distribution functions (CDFs) of gene expression ranks inside and outside the gene set.
   - By default, the pathway scores are normalized by dividing them by the range of calculated values.

### 4. **GSVA** (Default Method)
   - **Reference**: Hänzelmann, Castelo, and Guinney (2013)
   - A non-parametric method using empirical CDFs of gene expression ranks inside and outside the gene set.
   - Calculates an expression-level statistic to bring gene expression profiles with varying dynamic ranges to a common scale.

## Installation

```bash
# Install from PyPI
pip install pygsva

# Or install from source
git clone https://github.com/guokai8/pygsva
cd pygsva
pip install .
```

## Usage

```python
from pygsva import *

# Load example data
hsko = load_hsko_data()
pbmc = load_pbmc_data()
gene_sets = {key: group.iloc[:, 0].tolist() for key, group in hsko.groupby(hsko.iloc[:, 2])}

# GSVA (default method)
param = gsvaParam(pbmc, gene_sets=gene_sets, kcdf="Gaussian", n_jobs=4)
result = gsva(param)

# For faster computation on large datasets, use kcdf="none"
param_fast = gsvaParam(pbmc, gene_sets=gene_sets, kcdf="none", n_jobs=4)
result_fast = gsva(param_fast)

# ssGSEA
result_ssgsea = ssgsea(pbmc, gene_sets, n_jobs=4)

# PLAGE
param_plage = plageParam(pbmc, gene_sets=gene_sets, min_size=2)
result_plage = gsva(param_plage)

# Z-score
param_zscore = zscoreParam(pbmc, gene_sets)
result_zscore = gsva(param_zscore)
```

## Key Parameters

- **kcdf**: Kernel for cumulative distribution function estimation
  - `"Gaussian"`: Standard GSVA with Gaussian kernel (default)
  - `"Poisson"`: For count data (RNA-seq)
  - `"none"`: Empirical CDF only (fastest, recommended for large datasets)
- **n_jobs**: Number of CPU cores for parallel processing (-1 for all cores)
- **min_size/max_size**: Filter gene sets by size
- **use_sparse**: Use sparse matrix operations for memory efficiency

## References

If you use any of the methods in this package, please cite the corresponding articles:

1. Tomfohr, Lu, and Kepler (2005) - Pathway Level Analysis of Gene Expression (PLAGE)
2. Lee et al. (2008) - Z-Score Method
3. Barbie et al. (2009) - Single Sample Gene Set Enrichment Analysis (ssGSEA)
4. Hänzelmann, Castelo, and Guinney (2013) - Gene Set Variation Analysis (GSVA)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions please contact guokai8@gmail.com or https://github.com/guokai8/pygsva/issues
