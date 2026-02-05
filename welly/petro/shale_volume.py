"""
Shale volume (Vshale) calculation methods.

This module provides multiple methods for estimating shale/clay volume
from well log data. Each method has different assumptions and applicability.

:copyright: 2024 Agile Scientific
:license: Apache 2.0

References:
    Larionov, V.V. (1969). Borehole Radiometry. Moscow, Nedra.
    
    Steiber, R.G. (1973). Optimization of shale volumes in open hole logs.
        Journal of Petroleum Technology, 31, 147-162.
    
    Clavier, C., Hoyle, W. & Meunier, D. (1971). Quantitative interpretation 
        of thermal neutron decay time logs. Journal of Petroleum Technology,
        23(6), 743-755.
    
    Dresser Atlas (1979). Log Interpretation Charts. Houston, TX.
"""
import numpy as np
from typing import Union, Optional
import warnings

from .utils import (
    to_numpy, get_index, get_curve_metadata,
    validate_range, safe_divide, ArrayLike
)


def _make_curve(data: np.ndarray, 
                index: Optional[np.ndarray],
                mnemonic: str,
                units: str = 'v/v',
                description: str = '',
                **metadata) -> 'Curve':
    """
    Create a Curve object from computed data.
    
    Args:
        data: Computed values
        index: Depth index
        mnemonic: Curve mnemonic
        units: Units string
        description: Curve description
        **metadata: Additional metadata
        
    Returns:
        Curve object
    """
    from welly import Curve
    
    return Curve(
        data=data,
        index=index,
        mnemonic=mnemonic,
        units=units,
        description=description,
        **{k: v for k, v in metadata.items() if v is not None}
    )


def _gamma_ray_index(gr: np.ndarray, 
                     gr_clean: float, 
                     gr_shale: float) -> np.ndarray:
    """
    Calculate gamma ray index (IGR).
    
    IGR = (GR - GR_clean) / (GR_shale - GR_clean)
    
    Args:
        gr: Gamma ray log values
        gr_clean: Clean sand gamma ray value
        gr_shale: Shale gamma ray value
        
    Returns:
        Gamma ray index (0-1)
    """
    if gr_shale <= gr_clean:
        raise ValueError("gr_shale must be greater than gr_clean")
    
    igr = safe_divide(gr - gr_clean, gr_shale - gr_clean)
    return np.clip(igr, 0, 1)


def vshale_linear(gr: ArrayLike,
                  gr_clean: float,
                  gr_shale: float,
                  return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate shale volume using linear gamma ray method.
    
    This is the simplest method and typically overestimates Vshale.
    Best used as an upper bound or in formations with low radioactive
    mineral content.
    
    Vsh = (GR - GR_clean) / (GR_shale - GR_clean)
    
    Args:
        gr: Gamma ray log (API units) - Curve object or array
        gr_clean: Gamma ray value in clean sand (API)
        gr_shale: Gamma ray value in pure shale (API)
        return_curve: If True, return Curve object; else return array
        
    Returns:
        Shale volume (v/v, 0-1) as Curve or array
        
    Example:
        >>> vsh = vshale_linear(well.data['GR'], gr_clean=20, gr_shale=120)
        >>> well.data['VSH_LIN'] = vsh
        
    Note:
        Linear method typically overestimates Vshale in consolidated
        formations. Consider using Larionov or Steiber for better estimates.
    """
    gr_arr = to_numpy(gr)
    index = get_index(gr)
    metadata = get_curve_metadata(gr)
    
    # Validate inputs
    validate_range(gr_arr, min_val=0, max_val=300, name='GR')
    
    vsh = _gamma_ray_index(gr_arr, gr_clean, gr_shale)
    
    if return_curve:
        return _make_curve(
            vsh, index, 
            mnemonic='VSH_LIN',
            units='v/v',
            description='Shale volume (linear GR method)',
            **metadata
        )
    return vsh


def vshale_larionov(gr: ArrayLike,
                    gr_clean: float,
                    gr_shale: float,
                    tertiary: bool = True,
                    return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate shale volume using Larionov's non-linear method.
    
    This method accounts for the non-linear relationship between
    gamma ray response and shale content, providing more realistic
    estimates than the linear method.
    
    For Tertiary (younger, unconsolidated) rocks:
        Vsh = 0.083 * (2^(3.7 * IGR) - 1)
    
    For older (consolidated) rocks:
        Vsh = 0.33 * (2^(2 * IGR) - 1)
    
    Args:
        gr: Gamma ray log (API units) - Curve object or array
        gr_clean: Gamma ray value in clean sand (API)
        gr_shale: Gamma ray value in pure shale (API)
        tertiary: If True, use Tertiary equation; else use older rocks equation
        return_curve: If True, return Curve object; else return array
        
    Returns:
        Shale volume (v/v, 0-1) as Curve or array
        
    Example:
        >>> # For Gulf Coast Tertiary sands
        >>> vsh = vshale_larionov(well.data['GR'], 20, 120, tertiary=True)
        
        >>> # For Paleozoic consolidated sands
        >>> vsh = vshale_larionov(well.data['GR'], 20, 120, tertiary=False)
        
    Reference:
        Larionov, V.V. (1969). Borehole Radiometry. Moscow, Nedra.
    """
    gr_arr = to_numpy(gr)
    index = get_index(gr)
    metadata = get_curve_metadata(gr)
    
    validate_range(gr_arr, min_val=0, max_val=300, name='GR')
    
    igr = _gamma_ray_index(gr_arr, gr_clean, gr_shale)
    
    if tertiary:
        vsh = 0.083 * (np.power(2, 3.7 * igr) - 1)
        method = 'Larionov Tertiary'
    else:
        vsh = 0.33 * (np.power(2, 2.0 * igr) - 1)
        method = 'Larionov Older'
    
    vsh = np.clip(vsh, 0, 1)
    
    if return_curve:
        return _make_curve(
            vsh, index,
            mnemonic='VSH_LAR',
            units='v/v',
            description=f'Shale volume ({method})',
            **metadata
        )
    return vsh


def vshale_larionov_older(gr: ArrayLike,
                          gr_clean: float,
                          gr_shale: float,
                          return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate shale volume using Larionov's equation for older rocks.
    
    Convenience function equivalent to vshale_larionov(..., tertiary=False).
    
    Args:
        gr: Gamma ray log (API units)
        gr_clean: Gamma ray value in clean sand (API)
        gr_shale: Gamma ray value in pure shale (API)
        return_curve: If True, return Curve object
        
    Returns:
        Shale volume (v/v, 0-1)
    """
    return vshale_larionov(gr, gr_clean, gr_shale, 
                           tertiary=False, return_curve=return_curve)


def vshale_steiber(gr: ArrayLike,
                   gr_clean: float,
                   gr_shale: float,
                   return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate shale volume using Steiber's method.
    
    Steiber's method provides estimates between linear and Larionov,
    and is widely used in the industry.
    
    Vsh = IGR / (3 - 2 * IGR)
    
    Args:
        gr: Gamma ray log (API units)
        gr_clean: Gamma ray value in clean sand (API)
        gr_shale: Gamma ray value in pure shale (API)
        return_curve: If True, return Curve object
        
    Returns:
        Shale volume (v/v, 0-1)
        
    Reference:
        Steiber, R.G. (1973). Optimization of shale volumes in open hole logs.
    """
    gr_arr = to_numpy(gr)
    index = get_index(gr)
    metadata = get_curve_metadata(gr)
    
    validate_range(gr_arr, min_val=0, max_val=300, name='GR')
    
    igr = _gamma_ray_index(gr_arr, gr_clean, gr_shale)
    
    vsh = safe_divide(igr, 3 - 2 * igr)
    vsh = np.clip(vsh, 0, 1)
    
    if return_curve:
        return _make_curve(
            vsh, index,
            mnemonic='VSH_STB',
            units='v/v',
            description='Shale volume (Steiber method)',
            **metadata
        )
    return vsh


def vshale_clavier(gr: ArrayLike,
                   gr_clean: float,
                   gr_shale: float,
                   return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate shale volume using Clavier's method.
    
    Clavier's method is similar to Steiber but with different coefficients.
    
    Vsh = 1.7 - sqrt(3.38 - (IGR + 0.7)^2)
    
    Args:
        gr: Gamma ray log (API units)
        gr_clean: Gamma ray value in clean sand (API)
        gr_shale: Gamma ray value in pure shale (API)
        return_curve: If True, return Curve object
        
    Returns:
        Shale volume (v/v, 0-1)
        
    Reference:
        Clavier, C., Hoyle, W. & Meunier, D. (1971).
    """
    gr_arr = to_numpy(gr)
    index = get_index(gr)
    metadata = get_curve_metadata(gr)
    
    validate_range(gr_arr, min_val=0, max_val=300, name='GR')
    
    igr = _gamma_ray_index(gr_arr, gr_clean, gr_shale)
    
    # Clavier equation
    term = 3.38 - np.power(igr + 0.7, 2)
    term = np.maximum(term, 0)  # Avoid sqrt of negative
    vsh = 1.7 - np.sqrt(term)
    vsh = np.clip(vsh, 0, 1)
    
    if return_curve:
        return _make_curve(
            vsh, index,
            mnemonic='VSH_CLV',
            units='v/v',
            description='Shale volume (Clavier method)',
            **metadata
        )
    return vsh


def vshale_from_sp(sp: ArrayLike,
                   sp_clean: float,
                   sp_shale: float,
                   return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate shale volume from spontaneous potential (SP) log.
    
    The SP method is useful when GR is affected by radioactive minerals
    (e.g., potassium feldspars, uranium-bearing sands).
    
    Vsh = (SP - SP_clean) / (SP_shale - SP_clean)
    
    Args:
        sp: SP log (mV)
        sp_clean: SP value in clean sand (mV)
        sp_shale: SP value in shale (mV, typically near 0)
        return_curve: If True, return Curve object
        
    Returns:
        Shale volume (v/v, 0-1)
        
    Note:
        SP response can be affected by:
        - Formation water salinity variations
        - Thin beds
        - Hydrocarbon effects
        - Invasion effects
    """
    sp_arr = to_numpy(sp)
    index = get_index(sp)
    metadata = get_curve_metadata(sp)
    
    if sp_shale == sp_clean:
        raise ValueError("sp_shale and sp_clean cannot be equal")
    
    vsh = safe_divide(sp_arr - sp_clean, sp_shale - sp_clean)
    vsh = np.clip(vsh, 0, 1)
    
    if return_curve:
        return _make_curve(
            vsh, index,
            mnemonic='VSH_SP',
            units='v/v',
            description='Shale volume (SP method)',
            **metadata
        )
    return vsh


def vshale_from_neutron_density(nphi: ArrayLike,
                                rhob: ArrayLike,
                                nphi_clean: float = 0.0,
                                nphi_shale: float = 0.35,
                                rhob_clean: float = 2.65,
                                rhob_shale: float = 2.45,
                                return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate shale volume from neutron-density crossplot.
    
    This method uses the separation between neutron and density
    porosity to estimate shale content. It's particularly useful
    in gas-bearing zones where GR may be unreliable.
    
    The method finds the intersection point on the neutron-density
    crossplot between the clean line and shale point.
    
    Args:
        nphi: Neutron porosity log (v/v)
        rhob: Bulk density log (g/cc)
        nphi_clean: Neutron porosity of clean matrix (v/v)
        nphi_shale: Neutron porosity of shale (v/v)
        rhob_clean: Bulk density of clean matrix (g/cc)
        rhob_shale: Bulk density of shale (g/cc)
        return_curve: If True, return Curve object
        
    Returns:
        Shale volume (v/v, 0-1)
        
    Note:
        This method assumes a two-component system (clean + shale).
        Results may be unreliable in:
        - Gas zones (neutron reads low)
        - Heavy mineral zones
        - Coal seams
    """
    nphi_arr = to_numpy(nphi)
    rhob_arr = to_numpy(rhob)
    index = get_index(nphi)
    metadata = get_curve_metadata(nphi)
    
    # Validate inputs
    validate_range(nphi_arr, min_val=-0.15, max_val=0.6, name='NPHI')
    validate_range(rhob_arr, min_val=1.5, max_val=3.0, name='RHOB')
    
    # Calculate Vshale from neutron
    vsh_n = safe_divide(nphi_arr - nphi_clean, nphi_shale - nphi_clean)
    
    # Calculate Vshale from density
    vsh_d = safe_divide(rhob_arr - rhob_clean, rhob_shale - rhob_clean)
    
    # Take minimum (most optimistic) - common industry practice
    # Could also use average or geometric mean
    vsh = np.minimum(vsh_n, vsh_d)
    vsh = np.clip(vsh, 0, 1)
    
    if return_curve:
        return _make_curve(
            vsh, index,
            mnemonic='VSH_ND',
            units='v/v',
            description='Shale volume (Neutron-Density method)',
            **metadata
        )
    return vsh
