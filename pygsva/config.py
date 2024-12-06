# config.py
import numpy as np
from scipy.stats import norm

# Default configuration values
sigma_factor = 4.0
precompute_resolution = 10000
max_precompute = 10.0
precomputed_values = None

def init_cdfs():
    """Initialize precomputed CDF values."""
    global precomputed_values
    precomputed = []
    for i in range(precompute_resolution + 1):
        x = max_precompute * (i / precompute_resolution)
        precomputed.append(norm.cdf(x, loc=0, scale=1))
    precomputed_values = precomputed
    return precomputed_values

def get_config():
    """Get current configuration values."""
    return {
        'sigma_factor': sigma_factor,
        'precompute_resolution': precompute_resolution,
        'max_precompute': max_precompute
    }

def update_config(new_sigma_factor=None, new_precompute_resolution=None, new_max_precompute=None):
    """
    Update configuration values and reinitialize precomputed values if needed.
    
    Parameters:
    -----------
    new_sigma_factor : float, optional
        Sigma factor for Gaussian kernels.
    new_precompute_resolution : int, optional
        Resolution for precomputing CDF values.
    new_max_precompute : float, optional
        Maximum value for precomputing CDF values.
    """
    global sigma_factor, precompute_resolution, max_precompute
    
    changed = False
    if new_sigma_factor is not None:
        sigma_factor = float(new_sigma_factor)
        changed = True
    if new_precompute_resolution is not None:
        precompute_resolution = int(new_precompute_resolution)
        changed = True
    if new_max_precompute is not None:
        max_precompute = float(new_max_precompute)
        changed = True
        
    if changed:
        init_cdfs()

# Initialize the precomputed values when the module is imported
init_cdfs()