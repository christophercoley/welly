"""
Utility functions for the petrophysics module.

:copyright: 2024 Agile Scientific
:license: Apache 2.0
"""
import numpy as np
from typing import Union, Optional, Tuple
import warnings


# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, list, float]


def to_numpy(data: ArrayLike) -> np.ndarray:
    """
    Convert input to numpy array, handling Curve objects.
    
    Args:
        data: Input data (Curve, ndarray, list, or scalar)
        
    Returns:
        numpy array
    """
    # Handle welly Curve objects
    if hasattr(data, 'values'):
        return np.asarray(data.values, dtype=float)
    elif hasattr(data, 'df'):
        return np.asarray(data.df.values, dtype=float).flatten()
    return np.asarray(data, dtype=float)


def get_index(data) -> Optional[np.ndarray]:
    """
    Extract index from Curve object if available.
    
    Args:
        data: Input data (Curve or array)
        
    Returns:
        Index array or None
    """
    if hasattr(data, 'index'):
        return np.asarray(data.index)
    elif hasattr(data, 'df'):
        return np.asarray(data.df.index)
    return None


def get_curve_metadata(data) -> dict:
    """
    Extract metadata from Curve object if available.
    
    Args:
        data: Input data (Curve or array)
        
    Returns:
        Dictionary of metadata (excluding mnemonic to avoid conflicts)
    """
    metadata = {}
    if hasattr(data, 'index_units'):
        metadata['index_units'] = data.index_units
    # Note: we don't include mnemonic here to avoid conflicts
    # when creating new curves with different mnemonics
    return metadata


def validate_range(data: np.ndarray, 
                   min_val: Optional[float] = None,
                   max_val: Optional[float] = None,
                   name: str = 'data',
                   clip: bool = False,
                   warn: bool = True) -> np.ndarray:
    """
    Validate that data falls within expected range.
    
    Args:
        data: Input array
        min_val: Minimum expected value
        max_val: Maximum expected value
        name: Name for warning messages
        clip: If True, clip values to range instead of warning
        warn: If True, emit warnings for out-of-range values
        
    Returns:
        Validated (and optionally clipped) array
    """
    result = data.copy()
    
    if min_val is not None:
        below = np.nansum(data < min_val)
        if below > 0:
            if clip:
                result = np.maximum(result, min_val)
            elif warn:
                warnings.warn(
                    f"{name}: {below} values below minimum ({min_val})",
                    stacklevel=3
                )
    
    if max_val is not None:
        above = np.nansum(data > max_val)
        if above > 0:
            if clip:
                result = np.minimum(result, max_val)
            elif warn:
                warnings.warn(
                    f"{name}: {above} values above maximum ({max_val})",
                    stacklevel=3
                )
    
    return result


def validate_positive(data: np.ndarray, 
                      name: str = 'data',
                      allow_zero: bool = True) -> np.ndarray:
    """
    Validate that data is positive.
    
    Args:
        data: Input array
        name: Name for error messages
        allow_zero: If True, allow zero values
        
    Returns:
        Input array (raises if invalid)
    """
    if allow_zero:
        if np.any(data < 0):
            raise ValueError(f"{name} must be non-negative")
    else:
        if np.any(data <= 0):
            raise ValueError(f"{name} must be positive")
    return data


def safe_divide(numerator: np.ndarray, 
                denominator: np.ndarray,
                fill_value: float = np.nan) -> np.ndarray:
    """
    Safe division handling zeros and infinities.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        fill_value: Value to use where division is undefined
        
    Returns:
        Result array with fill_value where division failed
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
        result[~np.isfinite(result)] = fill_value
    return result


def safe_sqrt(data: ArrayLike, fill_value: float = np.nan) -> np.ndarray:
    """
    Safe square root handling negative values.
    
    Args:
        data: Input array or scalar
        fill_value: Value to use for negative inputs
        
    Returns:
        Square root with fill_value for negative inputs
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 0:
        # Scalar case
        if data >= 0:
            return np.sqrt(data)
        return fill_value
    result = np.full_like(data, fill_value, dtype=float)
    valid = data >= 0
    result[valid] = np.sqrt(data[valid])
    return result


def safe_power(base: np.ndarray, 
               exponent: float,
               fill_value: float = np.nan) -> np.ndarray:
    """
    Safe power operation handling edge cases.
    
    Args:
        base: Base array
        exponent: Exponent value
        fill_value: Value to use where operation fails
        
    Returns:
        Result array
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.power(base, exponent)
        result[~np.isfinite(result)] = fill_value
    return result


def safe_log(data: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """
    Safe natural logarithm handling non-positive values.
    
    Args:
        data: Input array
        fill_value: Value to use for non-positive inputs
        
    Returns:
        Log with fill_value for invalid inputs
    """
    result = np.full_like(data, fill_value, dtype=float)
    valid = data > 0
    result[valid] = np.log(data[valid])
    return result


def interpolate_nans(data: np.ndarray, 
                     index: Optional[np.ndarray] = None,
                     method: str = 'linear') -> np.ndarray:
    """
    Interpolate NaN values in array.
    
    Args:
        data: Input array with NaNs
        index: Optional index for interpolation
        method: Interpolation method ('linear', 'nearest')
        
    Returns:
        Array with NaNs interpolated
    """
    result = data.copy()
    nans = np.isnan(result)
    
    if not np.any(nans):
        return result
    
    if np.all(nans):
        return result
    
    if index is None:
        index = np.arange(len(data))
    
    if method == 'linear':
        result[nans] = np.interp(
            index[nans], 
            index[~nans], 
            result[~nans]
        )
    elif method == 'nearest':
        from scipy.interpolate import interp1d
        f = interp1d(index[~nans], result[~nans], 
                     kind='nearest', fill_value='extrapolate')
        result[nans] = f(index[nans])
    
    return result


def compute_statistics(data: np.ndarray) -> dict:
    """
    Compute summary statistics for petrophysical data.
    
    Args:
        data: Input array
        
    Returns:
        Dictionary of statistics
    """
    valid = data[~np.isnan(data)]
    
    if len(valid) == 0:
        return {
            'count': 0,
            'valid': 0,
            'nan_count': len(data),
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'p10': np.nan,
            'p50': np.nan,
            'p90': np.nan,
        }
    
    return {
        'count': len(data),
        'valid': len(valid),
        'nan_count': len(data) - len(valid),
        'mean': np.mean(valid),
        'std': np.std(valid),
        'min': np.min(valid),
        'max': np.max(valid),
        'p10': np.percentile(valid, 10),
        'p50': np.percentile(valid, 50),
        'p90': np.percentile(valid, 90),
    }


def apply_zone_mask(data: np.ndarray,
                    zones: np.ndarray,
                    target_zones: Union[list, int, str]) -> np.ndarray:
    """
    Apply zone-based masking to data.
    
    Args:
        data: Input array
        zones: Zone indicator array (same length as data)
        target_zones: Zone(s) to keep (others become NaN)
        
    Returns:
        Masked array with NaN outside target zones
    """
    if not isinstance(target_zones, (list, tuple)):
        target_zones = [target_zones]
    
    result = np.full_like(data, np.nan, dtype=float)
    mask = np.isin(zones, target_zones)
    result[mask] = data[mask]
    
    return result


def depth_shift(data: np.ndarray,
                index: np.ndarray,
                shift: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply depth shift to data.
    
    Args:
        data: Input array
        index: Depth index
        shift: Shift amount (positive = deeper)
        
    Returns:
        Tuple of (shifted_data, shifted_index)
    """
    return data, index + shift
