"""
Porosity calculation methods.

This module provides methods for calculating porosity from various
well log measurements including density, neutron, and sonic logs.

:copyright: 2024 Agile Scientific
:license: Apache 2.0

References:
    Wyllie, M.R.J., Gregory, A.R. & Gardner, G.H.F. (1958). An experimental 
        investigation of factors affecting elastic wave velocities in 
        porous media. Geophysics, 23(3), 459-493.
    
    Raymer, L.L., Hunt, E.R. & Gardner, J.S. (1980). An improved sonic 
        transit time-to-porosity transform. SPWLA 21st Annual Logging 
        Symposium.
    
    Gaymard, R. & Poupon, A. (1968). Response of neutron and formation 
        density logs in hydrocarbon bearing formations. The Log Analyst, 
        9(5), 3-12.
"""
import numpy as np
from typing import Union, Optional
import warnings

from .utils import (
    to_numpy, get_index, get_curve_metadata,
    validate_range, safe_divide, safe_sqrt, ArrayLike
)


def _make_curve(data: np.ndarray, 
                index: Optional[np.ndarray],
                mnemonic: str,
                units: str = 'v/v',
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


def porosity_density(rhob: ArrayLike,
                     rho_matrix: float = 2.65,
                     rho_fluid: float = 1.0,
                     return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate porosity from bulk density log.
    
    The density porosity equation assumes a two-component system
    of matrix and fluid:
    
    φ_D = (ρ_matrix - ρ_bulk) / (ρ_matrix - ρ_fluid)
    
    Args:
        rhob: Bulk density log (g/cc) - Curve object or array
        rho_matrix: Matrix density (g/cc). Common values:
            - Sandstone (quartz): 2.65
            - Limestone (calcite): 2.71
            - Dolomite: 2.87
        rho_fluid: Fluid density (g/cc). Common values:
            - Fresh water: 1.0
            - Salt water: 1.0-1.2
            - Oil: 0.7-0.9
            - Gas: 0.1-0.3
        return_curve: If True, return Curve object; else return array
        
    Returns:
        Density porosity (v/v) as Curve or array
        
    Example:
        >>> # Sandstone with fresh water
        >>> phi_d = porosity_density(well.data['RHOB'], 
        ...                          rho_matrix=2.65, 
        ...                          rho_fluid=1.0)
        
        >>> # Limestone with brine
        >>> phi_d = porosity_density(well.data['RHOB'],
        ...                          rho_matrix=2.71,
        ...                          rho_fluid=1.1)
        
    Note:
        - In gas zones, use gas-corrected fluid density
        - In shaly sands, density porosity includes clay-bound water
        - Heavy minerals (pyrite, siderite) will cause low porosity
    """
    rhob_arr = to_numpy(rhob)
    index = get_index(rhob)
    metadata = get_curve_metadata(rhob)
    
    # Validate inputs
    validate_range(rhob_arr, min_val=1.5, max_val=3.0, name='RHOB')
    
    if rho_matrix <= rho_fluid:
        raise ValueError("rho_matrix must be greater than rho_fluid")
    
    phi = safe_divide(rho_matrix - rhob_arr, rho_matrix - rho_fluid)
    
    # Clip to physical bounds but warn about issues
    out_of_range = np.sum((phi < -0.05) | (phi > 0.5))
    if out_of_range > 0:
        warnings.warn(
            f"Density porosity: {out_of_range} values outside typical range. "
            "Check matrix/fluid parameters or data quality.",
            stacklevel=2
        )
    
    phi = np.clip(phi, 0, 1)
    
    if return_curve:
        return _make_curve(
            phi, index,
            mnemonic='PHID',
            units='v/v',
            description=f'Density porosity (ρma={rho_matrix}, ρfl={rho_fluid})',
            **metadata
        )
    return phi


def porosity_neutron(nphi: ArrayLike,
                     nphi_matrix: float = 0.0,
                     lithology: str = 'limestone',
                     return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate corrected neutron porosity.
    
    Neutron logs are typically calibrated to limestone. This function
    applies matrix corrections for other lithologies.
    
    Args:
        nphi: Neutron porosity log (v/v) - typically limestone-calibrated
        nphi_matrix: Matrix neutron response (v/v). Common values:
            - Sandstone: -0.02 to 0.0
            - Limestone: 0.0 (reference)
            - Dolomite: 0.02
        lithology: Lithology name for description
        return_curve: If True, return Curve object
        
    Returns:
        Corrected neutron porosity (v/v)
        
    Example:
        >>> # Correct limestone-calibrated neutron for sandstone
        >>> phi_n = porosity_neutron(well.data['NPHI'], 
        ...                          nphi_matrix=-0.02,
        ...                          lithology='sandstone')
        
    Note:
        - Gas effect causes neutron to read low (excavation effect)
        - Shale causes neutron to read high (hydrogen in clay)
        - Neutron porosity includes all hydrogen (free + bound water)
    """
    nphi_arr = to_numpy(nphi)
    index = get_index(nphi)
    metadata = get_curve_metadata(nphi)
    
    # Validate inputs
    validate_range(nphi_arr, min_val=-0.15, max_val=0.6, name='NPHI')
    
    # Apply matrix correction
    phi = nphi_arr - nphi_matrix
    phi = np.clip(phi, 0, 1)
    
    if return_curve:
        return _make_curve(
            phi, index,
            mnemonic='PHIN',
            units='v/v',
            description=f'Neutron porosity ({lithology} corrected)',
            **metadata
        )
    return phi


def porosity_sonic_wyllie(dt: ArrayLike,
                          dt_matrix: float = 55.5,
                          dt_fluid: float = 189.0,
                          compaction_factor: float = 1.0,
                          return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate porosity from sonic log using Wyllie time-average equation.
    
    The Wyllie equation assumes the sonic travel time is a weighted
    average of matrix and fluid travel times:
    
    φ = (Δt - Δt_matrix) / (Δt_fluid - Δt_matrix) / Cp
    
    Args:
        dt: Sonic travel time log (µs/ft)
        dt_matrix: Matrix travel time (µs/ft). Common values:
            - Sandstone: 55.5-51.0
            - Limestone: 47.5
            - Dolomite: 43.5
        dt_fluid: Fluid travel time (µs/ft). Common values:
            - Water: 189
            - Oil: 230
            - Gas: 600+
        compaction_factor: Compaction correction factor (Cp).
            - Cp = 1.0 for compacted formations
            - Cp = Δt_shale / 100 for unconsolidated (Cp > 1)
        return_curve: If True, return Curve object
        
    Returns:
        Sonic porosity (v/v)
        
    Example:
        >>> # Compacted sandstone
        >>> phi_s = porosity_sonic_wyllie(well.data['DT'],
        ...                               dt_matrix=55.5,
        ...                               dt_fluid=189.0)
        
        >>> # Unconsolidated sand with Cp correction
        >>> phi_s = porosity_sonic_wyllie(well.data['DT'],
        ...                               dt_matrix=55.5,
        ...                               dt_fluid=189.0,
        ...                               compaction_factor=1.3)
        
    Reference:
        Wyllie, M.R.J., Gregory, A.R. & Gardner, G.H.F. (1958).
        
    Note:
        - Wyllie equation works best in compacted, water-filled formations
        - Overestimates porosity in unconsolidated formations
        - Gas effect causes sonic to read high (slow)
        - Consider Raymer-Hunt-Gardner for unconsolidated formations
    """
    dt_arr = to_numpy(dt)
    index = get_index(dt)
    metadata = get_curve_metadata(dt)
    
    # Validate inputs
    validate_range(dt_arr, min_val=40, max_val=200, name='DT')
    
    if dt_fluid <= dt_matrix:
        raise ValueError("dt_fluid must be greater than dt_matrix")
    
    if compaction_factor <= 0:
        raise ValueError("compaction_factor must be positive")
    
    phi = safe_divide(dt_arr - dt_matrix, dt_fluid - dt_matrix)
    phi = phi / compaction_factor
    phi = np.clip(phi, 0, 1)
    
    if return_curve:
        return _make_curve(
            phi, index,
            mnemonic='PHIS_WYL',
            units='v/v',
            description=f'Sonic porosity (Wyllie, Δtma={dt_matrix})',
            **metadata
        )
    return phi


def porosity_sonic_raymer(dt: ArrayLike,
                          dt_matrix: float = 55.5,
                          return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate porosity from sonic log using Raymer-Hunt-Gardner equation.
    
    The Raymer equation is an empirical relationship that works better
    than Wyllie in unconsolidated formations:
    
    For φ < 0.37:
        φ = (5/8) * (Δt - Δt_matrix) / Δt
    
    For φ ≥ 0.37:
        φ = (Δt - Δt_matrix) / (Δt + Δt_matrix)
    
    Args:
        dt: Sonic travel time log (µs/ft)
        dt_matrix: Matrix travel time (µs/ft)
        return_curve: If True, return Curve object
        
    Returns:
        Sonic porosity (v/v)
        
    Reference:
        Raymer, L.L., Hunt, E.R. & Gardner, J.S. (1980).
        
    Note:
        - Better than Wyllie for unconsolidated formations
        - Does not require compaction correction
        - Still affected by gas and shale
    """
    dt_arr = to_numpy(dt)
    index = get_index(dt)
    metadata = get_curve_metadata(dt)
    
    # Validate inputs
    validate_range(dt_arr, min_val=40, max_val=200, name='DT')
    
    # Raymer equation (simplified form)
    # Using the approximation: φ ≈ 0.625 * (Δt - Δtma) / Δt
    phi = 0.625 * safe_divide(dt_arr - dt_matrix, dt_arr)
    
    # For high porosity (>37%), use alternate form
    high_phi_mask = phi > 0.37
    if np.any(high_phi_mask):
        phi_high = safe_divide(dt_arr - dt_matrix, dt_arr + dt_matrix)
        phi[high_phi_mask] = phi_high[high_phi_mask]
    
    phi = np.clip(phi, 0, 1)
    
    if return_curve:
        return _make_curve(
            phi, index,
            mnemonic='PHIS_RHG',
            units='v/v',
            description=f'Sonic porosity (Raymer-Hunt-Gardner, Δtma={dt_matrix})',
            **metadata
        )
    return phi


def porosity_neutron_density(nphi: ArrayLike,
                             rhob: ArrayLike,
                             rho_matrix: float = 2.65,
                             rho_fluid: float = 1.0,
                             nphi_matrix: float = 0.0,
                             method: str = 'rms',
                             return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate porosity from neutron-density combination.
    
    Combining neutron and density provides a more robust porosity
    estimate that partially compensates for gas and shale effects.
    
    Methods:
        'average': φ = (φ_N + φ_D) / 2
        'rms': φ = sqrt((φ_N² + φ_D²) / 2)  [recommended]
        'minimum': φ = min(φ_N, φ_D)
        'gaymard': φ = sqrt((φ_N² + φ_D²) / 2) with gas correction
    
    Args:
        nphi: Neutron porosity log (v/v)
        rhob: Bulk density log (g/cc)
        rho_matrix: Matrix density (g/cc)
        rho_fluid: Fluid density (g/cc)
        nphi_matrix: Matrix neutron response (v/v)
        method: Combination method ('average', 'rms', 'minimum', 'gaymard')
        return_curve: If True, return Curve object
        
    Returns:
        Combined porosity (v/v)
        
    Example:
        >>> phi_nd = porosity_neutron_density(
        ...     well.data['NPHI'],
        ...     well.data['RHOB'],
        ...     rho_matrix=2.65,
        ...     method='rms'
        ... )
        
    Reference:
        Gaymard, R. & Poupon, A. (1968).
    """
    nphi_arr = to_numpy(nphi)
    rhob_arr = to_numpy(rhob)
    index = get_index(nphi)
    metadata = get_curve_metadata(nphi)
    
    # Calculate individual porosities
    phi_n = nphi_arr - nphi_matrix
    phi_d = safe_divide(rho_matrix - rhob_arr, rho_matrix - rho_fluid)
    
    # Combine based on method
    if method == 'average':
        phi = (phi_n + phi_d) / 2
    elif method == 'rms':
        phi = safe_sqrt((phi_n**2 + phi_d**2) / 2)
    elif method == 'minimum':
        phi = np.minimum(phi_n, phi_d)
    elif method == 'gaymard':
        # Gaymard method with gas correction
        # When φN < φD (gas effect), use geometric mean
        # Otherwise use RMS
        gas_flag = phi_n < phi_d
        phi = safe_sqrt((phi_n**2 + phi_d**2) / 2)
        phi_gas = safe_sqrt(phi_n * phi_d)
        phi[gas_flag] = phi_gas[gas_flag]
    else:
        raise ValueError(f"Unknown method: {method}. "
                        "Use 'average', 'rms', 'minimum', or 'gaymard'")
    
    phi = np.clip(phi, 0, 1)
    
    if return_curve:
        return _make_curve(
            phi, index,
            mnemonic='PHIND',
            units='v/v',
            description=f'Neutron-Density porosity ({method})',
            **metadata
        )
    return phi


def porosity_effective(phi_total: ArrayLike,
                       vshale: ArrayLike,
                       phi_shale: float = 0.0,
                       return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate effective porosity from total porosity and shale volume.
    
    Effective porosity excludes clay-bound water and represents
    the pore space available for moveable fluids.
    
    φ_e = φ_t - V_sh * φ_sh
    
    Args:
        phi_total: Total porosity (v/v)
        vshale: Shale volume (v/v)
        phi_shale: Porosity of shale (v/v), typically 0 for dry clay model
            or 0.1-0.3 for wet clay model
        return_curve: If True, return Curve object
        
    Returns:
        Effective porosity (v/v)
        
    Example:
        >>> # Dry clay model (φsh = 0)
        >>> phi_e = porosity_effective(phi_total, vshale, phi_shale=0)
        
        >>> # Wet clay model
        >>> phi_e = porosity_effective(phi_total, vshale, phi_shale=0.15)
        
    Note:
        - Dry clay model: assumes clay-bound water is part of matrix
        - Wet clay model: accounts for clay porosity explicitly
        - Effective porosity is used in Archie-type saturation equations
    """
    phi_t = to_numpy(phi_total)
    vsh = to_numpy(vshale)
    index = get_index(phi_total)
    metadata = get_curve_metadata(phi_total)
    
    phi_e = phi_t - vsh * phi_shale
    phi_e = np.clip(phi_e, 0, 1)
    
    if return_curve:
        return _make_curve(
            phi_e, index,
            mnemonic='PHIE',
            units='v/v',
            description='Effective porosity',
            **metadata
        )
    return phi_e


def porosity_total_from_density(rhob: ArrayLike,
                                rho_matrix: float = 2.65,
                                rho_fluid: float = 1.0,
                                rho_hydrocarbon: Optional[float] = None,
                                sw: Optional[ArrayLike] = None,
                                return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate total porosity from density with optional hydrocarbon correction.
    
    When hydrocarbons are present, the effective fluid density is:
    ρ_fluid_eff = Sw * ρ_water + (1 - Sw) * ρ_hydrocarbon
    
    Args:
        rhob: Bulk density log (g/cc)
        rho_matrix: Matrix density (g/cc)
        rho_fluid: Water density (g/cc)
        rho_hydrocarbon: Hydrocarbon density (g/cc), optional
        sw: Water saturation (v/v), required if rho_hydrocarbon provided
        return_curve: If True, return Curve object
        
    Returns:
        Total porosity (v/v)
        
    Example:
        >>> # Simple case (water-filled)
        >>> phi_t = porosity_total_from_density(rhob, rho_matrix=2.65)
        
        >>> # With gas correction
        >>> phi_t = porosity_total_from_density(
        ...     rhob, 
        ...     rho_matrix=2.65,
        ...     rho_fluid=1.05,
        ...     rho_hydrocarbon=0.2,
        ...     sw=sw_curve
        ... )
    """
    rhob_arr = to_numpy(rhob)
    index = get_index(rhob)
    metadata = get_curve_metadata(rhob)
    
    # Calculate effective fluid density if hydrocarbon present
    if rho_hydrocarbon is not None:
        if sw is None:
            raise ValueError("sw required when rho_hydrocarbon is provided")
        sw_arr = to_numpy(sw)
        rho_fluid_eff = sw_arr * rho_fluid + (1 - sw_arr) * rho_hydrocarbon
    else:
        rho_fluid_eff = rho_fluid
    
    phi = safe_divide(rho_matrix - rhob_arr, rho_matrix - rho_fluid_eff)
    phi = np.clip(phi, 0, 1)
    
    if return_curve:
        return _make_curve(
            phi, index,
            mnemonic='PHIT',
            units='v/v',
            description='Total porosity from density',
            **metadata
        )
    return phi
