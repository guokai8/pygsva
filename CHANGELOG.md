# Changelog

## [Unreleased] - 2026-01-15

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
