# Changelog

## [0.2.0] - 2026-02-01

### Performance Improvements

This release includes comprehensive performance optimizations across all methods, achieving 2-10x speedups through vectorized NumPy operations.

#### Benchmark Results

| Method | Small (1K×100) | Medium (5K×200) | Large (10K×500) |
|--------|----------------|-----------------|-----------------|
| GSVA (Gaussian) | 0.5s | 4.2s | 31.6s |
| GSVA (none) | 0.2s | 1.5s | 10.9s |
| ssGSEA | 0.2s | 1.1s | 7.1s |
| PLAGE | 0.4s | 2.1s | 7.2s |
| Z-score | 0.1s | 0.5s | 1.1s |

#### Optimized Functions

**pygsva/utils.py:**
- `ecdf_dense_to_dense()`: Replaced nested loops with `np.unique(return_inverse=True)` for O(n log n) complexity
- `ecdf_dense_to_dense_nas()`: Vectorized NA handling
- `ecdf_sparse_to_sparse()`: Vectorized counting and ECDF mapping
- `ecdf_sparse_to_dense()`: Vectorized operations
- `rank_ties_last()`: Replaced while loops with `np.lexsort()` for vectorized tie-breaking
- `colRanks()`: Optimized dispatch for different tie methods
- `filter_genes()`: Replaced list comprehension with `np.minimum.reduceat()` and `np.maximum.reduceat()`

**pygsva/gsvap.py:**
- `gsva_rnd_walk()`: Vectorized gene set indexing operations
- `gsva_rnd_walk_nas()`: Vectorized with NA handling
- `precomputed_cdf_vectorized()`: Vectorized CDF lookup using NumPy array indexing
- `matrix_density_vectorized()`: Replaced nested loops with vectorized CDF computation

**pygsva/ssgsea.py:**
- `order_value()`: Simplified to `np.argsort()` with stable sorting
- `ssgsea()`: Pre-converted gene sets to numpy arrays, consistent iteration order
- `ssgsea_batched()`: Same optimizations for batch processing

**pygsva/plage.py & pygsva/zscore.py:**
- `sparse_column_standardize()`: Vectorized with `np.add.reduceat()` for column statistics

### Validation

All optimizations produce **identical results** to the previous implementation (validated to machine precision).

### Backward Compatibility

No API changes. Existing code works without modification.

---

## [0.1.7] - 2026-01-15

### Performance Improvements

#### GSVA Algorithm Optimization (13x speedup)

**Summary**: Optimized GSVA's Gaussian kernel CDF computation by replacing nested Python loops with vectorized NumPy operations.

**Performance Impact**:
- Small datasets (1,000 × 100): 5.8x faster
- Medium datasets (5,000 × 500): 24.7x faster
- Large datasets (19,000 × 1,500): 13.1x faster

**Changes**:
- `pygsva/utils.py`:
  - `precomputed_cdf_vectorized()` (lines 41-55): Vectorized CDF lookup using NumPy array indexing
  - `matrix_density_vectorized()` (lines 622-644): Replaced nested loops with vectorized CDF computation

**Validation**: All optimizations produce identical results to previous implementation (validated to machine precision).

**Backward Compatibility**: No API changes. Existing code works without modification.

**Example**: For a dataset with 19,000 genes × 1,500 samples:
- Before: 3,122 seconds (52 minutes)
- After: 238 seconds (4 minutes)
- Alternative with `kcdf='none'`: 73 seconds (1.2 minutes)
