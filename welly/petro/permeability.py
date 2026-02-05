"""
Permeability estimation methods.

This module provides empirical correlations for estimating permeability
from well log data. All methods are empirical and should be calibrated
to core data when available.

:copyright: 2024 Agile Scientific
:license: Apache 2.0

References:
    Timur, A. (1968). An investigation of permeability, porosity, and 
        residual water saturation relationships for sandstone reservoirs.
        The Log Analyst, 9(4), 8-17.
    
    Coates, G.R. & Dumanoir, J.L. (1974). A new approach to improved 
        log-derived permeability. The Log Analyst, 15(1), 17-31.
    
    Tixier, M.P. (1949). Evaluation of permeability from electric-log 
        resistivity gradients. Oil & Gas Journal, 48(6), 113-122.
    
    Morris, R.L. & Biggs, W.P. (1967). Using log-derived values of water 
        saturation and porosity. SPWLA 8th Annual Logging Symposium.
    
    Wyllie, M.R.J. & Rose, W.D. (1950). Some theoretical considerations 
        related to the quantitative evaluation of the physical 
        characteristics of reservoir rock from electrical log data.
        Trans. AIME, 189, 105-118.
"""
import numpy as np
from typing import Union, Optional
import warnings

from .utils import (
    to_numpy, get_index, get_curve_metadata,
    validate_range, safe_divide, safe_power, ArrayLike
)


def _make_curve(data: np.ndarray, 
                index: Optional[np.ndarray],
                mnemonic: str,
                units: str = 'mD',
                description: str = '',
                **metadata) -> 'Curve':
    """Create a Curve object from computed data."""
    from welly import Curve
    
    return Curve(
        data=data,
        index=index,
        mnemonic=mnemonic,
        units=units,
        description=description,
        **{k: v for k, v in metadata.items() if v is not None}
    )


def perm_timur(phi: ArrayLike,
               swirr: ArrayLike,
               a: float = 8581,
               b: float = 4.4,
               c: float = 2.0,
               return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate permeability using Timur's equation.
    
    Timur's equation relates permeability to porosity and irreducible
    water saturation:
    
    k = a * φ^b / Swirr^c
    
    Default coefficients (a=8581, b=4.4, c=2.0) are for sandstones.
    
    Args:
        phi: Porosity (v/v)
        swirr: Irreducible water saturation (v/v)
        a: Coefficient (default 8581 for sandstones)
        b: Porosity exponent (default 4.4)
        c: Swirr exponent (default 2.0)
        return_curve: If True, return Curve object
        
    Returns:
        Permeability (mD)
        
    Example:
        >>> # Using Sw as proxy for Swirr (valid in transition zone)
        >>> k = perm_timur(phi, sw)
        
        >>> # With calibrated coefficients
        >>> k = perm_timur(phi, swirr, a=10000, b=4.5, c=2.2)
        
    Reference:
        Timur, A. (1968).
        
    Note:
        - Swirr should be irreducible water saturation
        - In practice, Sw from logs is often used as proxy
        - Calibrate coefficients to core data when available
    """
    phi_arr = to_numpy(phi)
    swirr_arr = to_numpy(swirr)
    index = get_index(phi)
    metadata = get_curve_metadata(phi)
    
    # Validate inputs
    validate_range(phi_arr, min_val=0.001, max_val=0.5, name='PHI')
    validate_range(swirr_arr, min_val=0.01, max_val=1.0, name='SWIRR')
    
    # Timur equation
    k = a * safe_power(phi_arr, b) / safe_power(swirr_arr, c)
    
    # Cap at reasonable values
    k = np.clip(k, 0.001, 100000)
    
    if return_curve:
        return _make_curve(
            k, index,
            mnemonic='KTIM',
            units='mD',
            description='Permeability (Timur)',
            **metadata
        )
    return k


def perm_coates(phi: ArrayLike,
                swirr: ArrayLike,
                c: float = 100,
                w: float = 4.0,
                return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate permeability using Coates equation.
    
    The Coates equation (also known as Coates-Dumanoir) uses the
    ratio of free fluid to bound fluid:
    
    k = (c * φ^2 * ((1-Swirr)/Swirr))^w
    
    Or equivalently:
    k = c^w * φ^(2w) * ((1-Swirr)/Swirr)^w
    
    Args:
        phi: Porosity (v/v)
        swirr: Irreducible water saturation (v/v)
        c: Coefficient (default 100)
        w: Exponent (default 4.0, giving k = (100*φ²*FFI/BVI)^4)
        return_curve: If True, return Curve object
        
    Returns:
        Permeability (mD)
        
    Example:
        >>> k = perm_coates(phi, swirr)
        
    Reference:
        Coates, G.R. & Dumanoir, J.L. (1974).
        
    Note:
        - FFI = Free Fluid Index = φ * (1 - Swirr)
        - BVI = Bulk Volume Irreducible = φ * Swirr
        - Often used with NMR-derived FFI and BVI
    """
    phi_arr = to_numpy(phi)
    swirr_arr = to_numpy(swirr)
    index = get_index(phi)
    metadata = get_curve_metadata(phi)
    
    # Validate inputs
    validate_range(phi_arr, min_val=0.001, max_val=0.5, name='PHI')
    validate_range(swirr_arr, min_val=0.01, max_val=0.99, name='SWIRR')
    
    # Coates equation
    # k = (c * φ² * (1-Swirr)/Swirr)^w
    ratio = safe_divide(1 - swirr_arr, swirr_arr)
    k = safe_power(c * phi_arr**2 * ratio, w)
    
    # Cap at reasonable values
    k = np.clip(k, 0.001, 100000)
    
    if return_curve:
        return _make_curve(
            k, index,
            mnemonic='KCOATES',
            units='mD',
            description='Permeability (Coates)',
            **metadata
        )
    return k


def perm_tixier(phi: ArrayLike,
                swirr: ArrayLike,
                return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate permeability using Tixier's equation.
    
    Tixier's equation is one of the earliest log-derived permeability
    correlations:
    
    k = 250 * φ³ / Swirr²  (for oil)
    k = 79 * φ³ / Swirr²   (for gas)
    
    This implementation uses the oil equation by default.
    
    Args:
        phi: Porosity (v/v)
        swirr: Irreducible water saturation (v/v)
        return_curve: If True, return Curve object
        
    Returns:
        Permeability (mD)
        
    Reference:
        Tixier, M.P. (1949).
    """
    phi_arr = to_numpy(phi)
    swirr_arr = to_numpy(swirr)
    index = get_index(phi)
    metadata = get_curve_metadata(phi)
    
    # Tixier equation (oil)
    k = 250 * safe_power(phi_arr, 3) / safe_power(swirr_arr, 2)
    k = np.clip(k, 0.001, 100000)
    
    if return_curve:
        return _make_curve(
            k, index,
            mnemonic='KTIX',
            units='mD',
            description='Permeability (Tixier)',
            **metadata
        )
    return k


def perm_morris_biggs(phi: ArrayLike,
                      swirr: ArrayLike,
                      fluid: str = 'oil',
                      return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate permeability using Morris-Biggs equation.
    
    Morris-Biggs provides separate equations for oil and gas:
    
    Oil:  k = 62500 * φ^6 / Swirr^2
    Gas:  k = 6241 * φ^6 / Swirr^2
    
    Args:
        phi: Porosity (v/v)
        swirr: Irreducible water saturation (v/v)
        fluid: 'oil' or 'gas'
        return_curve: If True, return Curve object
        
    Returns:
        Permeability (mD)
        
    Reference:
        Morris, R.L. & Biggs, W.P. (1967).
    """
    phi_arr = to_numpy(phi)
    swirr_arr = to_numpy(swirr)
    index = get_index(phi)
    metadata = get_curve_metadata(phi)
    
    if fluid.lower() == 'oil':
        coef = 62500
    elif fluid.lower() == 'gas':
        coef = 6241
    else:
        raise ValueError("fluid must be 'oil' or 'gas'")
    
    k = coef * safe_power(phi_arr, 6) / safe_power(swirr_arr, 2)
    k = np.clip(k, 0.001, 100000)
    
    if return_curve:
        return _make_curve(
            k, index,
            mnemonic='KMB',
            units='mD',
            description=f'Permeability (Morris-Biggs, {fluid})',
            **metadata
        )
    return k


def perm_wyllie_rose(phi: ArrayLike,
                     swirr: ArrayLike,
                     return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate permeability using Wyllie-Rose equation.
    
    k = 250 * φ³ / Swirr
    
    Args:
        phi: Porosity (v/v)
        swirr: Irreducible water saturation (v/v)
        return_curve: If True, return Curve object
        
    Returns:
        Permeability (mD)
        
    Reference:
        Wyllie, M.R.J. & Rose, W.D. (1950).
    """
    phi_arr = to_numpy(phi)
    swirr_arr = to_numpy(swirr)
    index = get_index(phi)
    metadata = get_curve_metadata(phi)
    
    k = 250 * safe_power(phi_arr, 3) / swirr_arr
    k = np.clip(k, 0.001, 100000)
    
    if return_curve:
        return _make_curve(
            k, index,
            mnemonic='KWR',
            units='mD',
            description='Permeability (Wyllie-Rose)',
            **metadata
        )
    return k


def perm_kozeny_carman(phi: ArrayLike,
                       grain_size: float = 0.1,
                       tortuosity: float = 2.5,
                       specific_surface: Optional[float] = None,
                       return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate permeability using Kozeny-Carman equation.
    
    The Kozeny-Carman equation is a theoretical relationship based
    on flow through packed spheres:
    
    k = (d² * φ³) / (180 * τ² * (1-φ)²)
    
    Or using specific surface area:
    k = φ³ / (5 * τ² * S² * (1-φ)²)
    
    Args:
        phi: Porosity (v/v)
        grain_size: Mean grain diameter (mm), used if specific_surface not given
        tortuosity: Tortuosity factor (typically 2-3)
        specific_surface: Specific surface area (1/mm), optional
        return_curve: If True, return Curve object
        
    Returns:
        Permeability (mD)
        
    Note:
        - Theoretical equation, often underestimates real permeability
        - Best for well-sorted, unconsolidated sands
        - Grain size can be estimated from grain size logs or assumed
    """
    phi_arr = to_numpy(phi)
    index = get_index(phi)
    metadata = get_curve_metadata(phi)
    
    if specific_surface is not None:
        # Using specific surface area
        # k = φ³ / (5 * τ² * S² * (1-φ)²)
        # Convert to mD (multiply by 1e6 for mm² to µm², then by 1013.25 for Darcy to mD)
        k = safe_power(phi_arr, 3) / (5 * tortuosity**2 * specific_surface**2 * safe_power(1 - phi_arr, 2))
        k = k * 1e6 * 1013.25  # Convert to mD
    else:
        # Using grain size
        # k = (d² * φ³) / (180 * τ² * (1-φ)²)
        # d in mm, convert to mD
        d_um = grain_size * 1000  # mm to µm
        k = (d_um**2 * safe_power(phi_arr, 3)) / (180 * tortuosity**2 * safe_power(1 - phi_arr, 2))
        k = k / 1000  # Approximate conversion factor
    
    k = np.clip(k, 0.001, 100000)
    
    if return_curve:
        return _make_curve(
            k, index,
            mnemonic='KKC',
            units='mD',
            description='Permeability (Kozeny-Carman)',
            **metadata
        )
    return k
