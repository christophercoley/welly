"""
Water saturation calculation methods.

This module provides multiple methods for calculating water saturation
from resistivity logs, ranging from simple Archie to complex shaly sand
models.

:copyright: 2024 Agile Scientific
:license: Apache 2.0

References:
    Archie, G.E. (1942). The electrical resistivity log as an aid in 
        determining some reservoir characteristics. Trans. AIME, 146, 54-62.
    
    Simandoux, P. (1963). Dielectric measurements on porous media: 
        Application to the measurement of water saturations. 
        Revue de l'Institut Français du Pétrole, 18, 193-215.
    
    Poupon, A. & Leveaux, J. (1971). Evaluation of water saturation in 
        shaly formations. SPWLA 12th Annual Logging Symposium.
    
    Fertl, W.H. & Hammack, G.W. (1971). A comparative look at water 
        saturation computations in shaly pay sands. SPWLA 12th Annual 
        Logging Symposium.
    
    Waxman, M.H. & Smits, L.J.M. (1968). Electrical conductivities in 
        oil-bearing shaly sands. SPE Journal, 8(2), 107-122.
    
    Clavier, C., Coates, G. & Dumanoir, J. (1984). Theoretical and 
        experimental bases for the dual-water model for interpretation 
        of shaly sands. SPE Journal, 24(2), 153-168.
"""
import numpy as np
from typing import Union, Optional
import warnings

from .utils import (
    to_numpy, get_index, get_curve_metadata,
    validate_range, safe_divide, safe_sqrt, safe_power, ArrayLike
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


def archie(rt: ArrayLike,
           phi: ArrayLike,
           rw: float,
           a: float = 1.0,
           m: float = 2.0,
           n: float = 2.0,
           return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate water saturation using Archie's equation.
    
    Archie's equation is the fundamental relationship between
    resistivity and water saturation in clean (non-shaly) formations:
    
    Sw = (a * Rw / (φ^m * Rt))^(1/n)
    
    Where:
        Sw = Water saturation
        a = Tortuosity factor (typically 0.62-1.0)
        Rw = Formation water resistivity
        φ = Porosity
        m = Cementation exponent (typically 1.8-2.2)
        Rt = True formation resistivity
        n = Saturation exponent (typically 1.8-2.2)
    
    Args:
        rt: True resistivity log (ohm.m)
        phi: Porosity (v/v)
        rw: Formation water resistivity (ohm.m)
        a: Tortuosity factor. Common values:
            - 1.0: Carbonates
            - 0.81: Consolidated sandstones (Humble formula)
            - 0.62: Unconsolidated sands (Humble formula)
        m: Cementation exponent. Common values:
            - 2.0: Default
            - 1.8-2.0: Sandstones
            - 2.0-2.2: Carbonates
        n: Saturation exponent (typically 2.0)
        return_curve: If True, return Curve object
        
    Returns:
        Water saturation (v/v, 0-1)
        
    Example:
        >>> sw = archie(
        ...     rt=well.data['RT'],
        ...     phi=well.data['PHIE'],
        ...     rw=0.05,
        ...     a=0.81, m=2.0, n=2.0
        ... )
        
    Note:
        - Only valid for clean (non-shaly) formations
        - Overestimates Sw in shaly sands
        - Use shaly sand models (Simandoux, Indonesia) for Vsh > 0.1
    """
    rt_arr = to_numpy(rt)
    phi_arr = to_numpy(phi)
    index = get_index(rt)
    metadata = get_curve_metadata(rt)
    
    # Validate inputs
    validate_range(rt_arr, min_val=0.1, max_val=10000, name='RT')
    validate_range(phi_arr, min_val=0.001, max_val=0.5, name='PHI')
    
    if rw <= 0:
        raise ValueError("rw must be positive")
    
    # Calculate formation factor
    F = a / safe_power(phi_arr, m)
    
    # Calculate Sw
    sw = safe_power(F * rw / rt_arr, 1/n)
    sw = np.clip(sw, 0, 1)
    
    if return_curve:
        return _make_curve(
            sw, index,
            mnemonic='SW_ARCH',
            units='v/v',
            description=f'Water saturation (Archie, a={a}, m={m}, n={n})',
            **metadata
        )
    return sw


def simandoux(rt: ArrayLike,
              phi: ArrayLike,
              vshale: ArrayLike,
              rw: float,
              rsh: float,
              a: float = 1.0,
              m: float = 2.0,
              n: float = 2.0,
              return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate water saturation using Simandoux equation.
    
    The Simandoux equation is a shaly sand model that accounts for
    the additional conductivity from clay minerals:
    
    1/Rt = (φ^m * Sw^n) / (a * Rw) + (Vsh * Sw) / Rsh
    
    Solving for Sw (quadratic):
    Sw = [C/2] * [-B + sqrt(B² + 4/C)]
    
    Where:
        C = a * Rw / (φ^m * Rt)
        B = Vsh / Rsh
    
    Args:
        rt: True resistivity log (ohm.m)
        phi: Effective porosity (v/v)
        vshale: Shale volume (v/v)
        rw: Formation water resistivity (ohm.m)
        rsh: Shale resistivity (ohm.m)
        a: Tortuosity factor
        m: Cementation exponent
        n: Saturation exponent (must be 2 for this formulation)
        return_curve: If True, return Curve object
        
    Returns:
        Water saturation (v/v, 0-1)
        
    Example:
        >>> sw = simandoux(
        ...     rt=well.data['RT'],
        ...     phi=well.data['PHIE'],
        ...     vshale=well.data['VSH'],
        ...     rw=0.05,
        ...     rsh=5.0
        ... )
        
    Reference:
        Simandoux, P. (1963).
        
    Note:
        - Valid for dispersed shale distribution
        - Assumes n=2 (quadratic solution)
        - More accurate than Archie for Vsh < 0.3
    """
    rt_arr = to_numpy(rt)
    phi_arr = to_numpy(phi)
    vsh_arr = to_numpy(vshale)
    index = get_index(rt)
    metadata = get_curve_metadata(rt)
    
    # Validate inputs
    validate_range(rt_arr, min_val=0.1, max_val=10000, name='RT')
    validate_range(phi_arr, min_val=0.001, max_val=0.5, name='PHI')
    validate_range(vsh_arr, min_val=0, max_val=1, name='VSH')
    
    if n != 2:
        warnings.warn(
            "Simandoux equation assumes n=2. Using n=2 regardless of input.",
            stacklevel=2
        )
    
    # Simandoux coefficients
    C = a * rw / (safe_power(phi_arr, m) * rt_arr)
    B = vsh_arr / rsh
    
    # Quadratic solution for Sw
    discriminant = B**2 + 4/C
    discriminant = np.maximum(discriminant, 0)  # Ensure non-negative
    
    sw = (C / 2) * (-B + np.sqrt(discriminant))
    sw = np.clip(sw, 0, 1)
    
    if return_curve:
        return _make_curve(
            sw, index,
            mnemonic='SW_SIM',
            units='v/v',
            description='Water saturation (Simandoux)',
            **metadata
        )
    return sw


def indonesia(rt: ArrayLike,
              phi: ArrayLike,
              vshale: ArrayLike,
              rw: float,
              rsh: float,
              a: float = 1.0,
              m: float = 2.0,
              n: float = 2.0,
              return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate water saturation using Indonesian (Poupon-Leveaux) equation.
    
    The Indonesian equation was developed for shaly sands in Indonesia
    and is widely used for its simplicity and effectiveness:
    
    1/sqrt(Rt) = sqrt(φ^m / (a*Rw)) * Sw^(n/2) + Vsh^(1-Vsh/2) / sqrt(Rsh) * Sw
    
    This is solved iteratively or using the simplified form.
    
    Args:
        rt: True resistivity log (ohm.m)
        phi: Effective porosity (v/v)
        vshale: Shale volume (v/v)
        rw: Formation water resistivity (ohm.m)
        rsh: Shale resistivity (ohm.m)
        a: Tortuosity factor
        m: Cementation exponent
        n: Saturation exponent
        return_curve: If True, return Curve object
        
    Returns:
        Water saturation (v/v, 0-1)
        
    Example:
        >>> sw = indonesia(
        ...     rt=well.data['RT'],
        ...     phi=well.data['PHIE'],
        ...     vshale=well.data['VSH'],
        ...     rw=0.05,
        ...     rsh=5.0
        ... )
        
    Reference:
        Poupon, A. & Leveaux, J. (1971).
        
    Note:
        - Works well for moderate to high shale content
        - More robust than Simandoux for high Vsh
        - Widely used in SE Asia and other shaly sand environments
    """
    rt_arr = to_numpy(rt)
    phi_arr = to_numpy(phi)
    vsh_arr = to_numpy(vshale)
    index = get_index(rt)
    metadata = get_curve_metadata(rt)
    
    # Validate inputs
    validate_range(rt_arr, min_val=0.1, max_val=10000, name='RT')
    validate_range(phi_arr, min_val=0.001, max_val=0.5, name='PHI')
    validate_range(vsh_arr, min_val=0, max_val=1, name='VSH')
    
    # Indonesian equation terms
    # Term A: clean sand contribution
    term_a = safe_sqrt(safe_power(phi_arr, m) / (a * rw))
    
    # Term B: shale contribution  
    # Vsh^(1 - Vsh/2) is the shale exponent
    vsh_exp = 1 - vsh_arr / 2
    term_b = safe_power(vsh_arr, vsh_exp) / safe_sqrt(rsh)
    
    # Combined term
    combined = term_a + term_b
    
    # Solve for Sw (assuming n=2 for simplicity)
    # 1/sqrt(Rt) = combined * Sw
    sw = safe_divide(1, combined * safe_sqrt(rt_arr))
    
    # For n != 2, iterate or use approximation
    if n != 2:
        # Iterative refinement
        for _ in range(5):
            sw_prev = sw.copy()
            sw = safe_power(
                safe_divide(1, (term_a * safe_power(sw_prev, n/2 - 1) + term_b) * safe_sqrt(rt_arr)),
                2/n
            )
            sw = np.clip(sw, 0, 1)
    
    sw = np.clip(sw, 0, 1)
    
    if return_curve:
        return _make_curve(
            sw, index,
            mnemonic='SW_IND',
            units='v/v',
            description='Water saturation (Indonesia/Poupon-Leveaux)',
            **metadata
        )
    return sw


def fertl(rt: ArrayLike,
          phi: ArrayLike,
          vshale: ArrayLike,
          rw: float,
          alpha: float = 0.25,
          a: float = 1.0,
          m: float = 2.0,
          n: float = 2.0,
          return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate water saturation using Fertl equation.
    
    The Fertl equation is a simplified shaly sand model:
    
    Sw = [(a * Rw) / (φ^m * Rt * (1 - α*Vsh))]^(1/n)
    
    Where α is an empirical shale correction factor.
    
    Args:
        rt: True resistivity log (ohm.m)
        phi: Effective porosity (v/v)
        vshale: Shale volume (v/v)
        rw: Formation water resistivity (ohm.m)
        alpha: Shale correction factor (typically 0.25-0.35)
        a: Tortuosity factor
        m: Cementation exponent
        n: Saturation exponent
        return_curve: If True, return Curve object
        
    Returns:
        Water saturation (v/v, 0-1)
        
    Reference:
        Fertl, W.H. & Hammack, G.W. (1971).
    """
    rt_arr = to_numpy(rt)
    phi_arr = to_numpy(phi)
    vsh_arr = to_numpy(vshale)
    index = get_index(rt)
    metadata = get_curve_metadata(rt)
    
    # Fertl correction factor
    correction = 1 - alpha * vsh_arr
    correction = np.maximum(correction, 0.1)  # Prevent division issues
    
    # Modified Archie with Fertl correction
    F = a / safe_power(phi_arr, m)
    sw = safe_power(F * rw / (rt_arr * correction), 1/n)
    sw = np.clip(sw, 0, 1)
    
    if return_curve:
        return _make_curve(
            sw, index,
            mnemonic='SW_FERTL',
            units='v/v',
            description=f'Water saturation (Fertl, α={alpha})',
            **metadata
        )
    return sw


def waxman_smits(rt: ArrayLike,
                 phi: ArrayLike,
                 qv: ArrayLike,
                 rw: float,
                 temp: float,
                 a: float = 1.0,
                 m_star: float = 2.0,
                 n_star: float = 2.0,
                 return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate water saturation using Waxman-Smits equation.
    
    The Waxman-Smits model accounts for clay conductivity through
    the cation exchange capacity (CEC) of the clay minerals:
    
    Ct = (1/F*) * (Cw + B*Qv/Sw) * Sw^n*
    
    Where:
        Ct = Total conductivity (1/Rt)
        F* = Formation factor (a/φ^m*)
        Cw = Water conductivity (1/Rw)
        B = Equivalent conductance of clay exchange cations
        Qv = Cation concentration per unit pore volume (meq/ml)
        Sw = Water saturation
        n* = Saturation exponent
    
    Args:
        rt: True resistivity log (ohm.m)
        phi: Total porosity (v/v)
        qv: Cation concentration (meq/ml) - can be computed from CEC
        rw: Formation water resistivity (ohm.m)
        temp: Formation temperature (°F)
        a: Tortuosity factor
        m_star: Cementation exponent (Waxman-Smits)
        n_star: Saturation exponent (Waxman-Smits)
        return_curve: If True, return Curve object
        
    Returns:
        Water saturation (v/v, 0-1)
        
    Reference:
        Waxman, M.H. & Smits, L.J.M. (1968).
        
    Note:
        - Requires Qv which can be estimated from CEC and porosity
        - More physically rigorous than empirical shaly sand models
        - B is temperature dependent
    """
    rt_arr = to_numpy(rt)
    phi_arr = to_numpy(phi)
    qv_arr = to_numpy(qv)
    index = get_index(rt)
    metadata = get_curve_metadata(rt)
    
    # Calculate B (equivalent conductance) - temperature dependent
    # B = -1.28 + 0.225*T - 0.0004059*T^2 (T in °C)
    temp_c = (temp - 32) * 5/9
    B = -1.28 + 0.225 * temp_c - 0.0004059 * temp_c**2
    B = max(B, 0.1)  # Ensure positive
    
    # Water conductivity
    Cw = 1 / rw
    
    # Formation factor
    F_star = a / safe_power(phi_arr, m_star)
    
    # Iterative solution for Sw
    sw = np.ones_like(rt_arr) * 0.5  # Initial guess
    
    for _ in range(20):
        sw_prev = sw.copy()
        
        # Waxman-Smits equation rearranged
        # Ct = (Cw + B*Qv/Sw) * Sw^n* / F*
        # Rt = F* / ((Cw + B*Qv/Sw) * Sw^n*)
        
        term = Cw + B * qv_arr / sw
        sw = safe_power(F_star / (rt_arr * term), 1/n_star)
        sw = np.clip(sw, 0.01, 1)
        
        # Check convergence
        if np.max(np.abs(sw - sw_prev)) < 0.001:
            break
    
    sw = np.clip(sw, 0, 1)
    
    if return_curve:
        return _make_curve(
            sw, index,
            mnemonic='SW_WS',
            units='v/v',
            description='Water saturation (Waxman-Smits)',
            **metadata
        )
    return sw


def dual_water(rt: ArrayLike,
               phi_t: ArrayLike,
               phi_e: ArrayLike,
               rw: float,
               rwb: float,
               a: float = 1.0,
               m: float = 2.0,
               n: float = 2.0,
               return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate water saturation using Dual-Water model.
    
    The Dual-Water model treats clay-bound water and free water
    as two distinct phases with different resistivities:
    
    1/Rt = (φt^m / a) * [Swt^n / Rw + (Swb^n - Swt^n) / Rwb]
    
    Where:
        Swt = Total water saturation
        Swb = Bound water saturation = (φt - φe) / φt
        Rw = Free water resistivity
        Rwb = Bound water resistivity
    
    Args:
        rt: True resistivity log (ohm.m)
        phi_t: Total porosity (v/v)
        phi_e: Effective porosity (v/v)
        rw: Free water resistivity (ohm.m)
        rwb: Bound water resistivity (ohm.m), typically 0.3-0.5 ohm.m
        a: Tortuosity factor
        m: Cementation exponent
        n: Saturation exponent
        return_curve: If True, return Curve object
        
    Returns:
        Total water saturation (v/v, 0-1)
        
    Reference:
        Clavier, C., Coates, G. & Dumanoir, J. (1984).
        
    Note:
        - Requires both total and effective porosity
        - Rwb is typically 0.3-0.5 ohm.m at reservoir temperature
        - Returns total water saturation (includes bound water)
    """
    rt_arr = to_numpy(rt)
    phi_t_arr = to_numpy(phi_t)
    phi_e_arr = to_numpy(phi_e)
    index = get_index(rt)
    metadata = get_curve_metadata(rt)
    
    # Bound water saturation
    swb = safe_divide(phi_t_arr - phi_e_arr, phi_t_arr)
    swb = np.clip(swb, 0, 1)
    
    # Formation factor
    F = a / safe_power(phi_t_arr, m)
    
    # Iterative solution for Swt
    swt = np.ones_like(rt_arr) * 0.5  # Initial guess
    
    for _ in range(20):
        swt_prev = swt.copy()
        
        # Dual-water equation
        # 1/Rt = (1/F) * [Swt^n/Rw + (Swb^n - Swt^n)/Rwb]
        # Rearranged for Swt
        
        term1 = safe_power(swb, n) / rwb
        term2 = 1 / rw - 1 / rwb
        
        # Solve: Swt^n * term2 + term1 = F / Rt
        target = F / rt_arr - term1
        swt = safe_power(target / term2, 1/n)
        swt = np.clip(swt, swb, 1)  # Swt >= Swb
        
        if np.max(np.abs(swt - swt_prev)) < 0.001:
            break
    
    swt = np.clip(swt, 0, 1)
    
    if return_curve:
        return _make_curve(
            swt, index,
            mnemonic='SWT_DW',
            units='v/v',
            description='Total water saturation (Dual-Water)',
            **metadata
        )
    return swt


def buckles_number(phi: ArrayLike,
                   sw: ArrayLike,
                   return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Calculate Buckles number (bulk volume water).
    
    Buckles number is the product of porosity and water saturation:
    
    BVW = φ * Sw
    
    In the transition zone above the free water level, BVW is
    approximately constant (Buckles' relationship). This can be
    used to identify the transition zone and estimate irreducible
    water saturation.
    
    Args:
        phi: Porosity (v/v)
        sw: Water saturation (v/v)
        return_curve: If True, return Curve object
        
    Returns:
        Buckles number / BVW (v/v)
        
    Example:
        >>> bvw = buckles_number(phi, sw)
        >>> # In transition zone, BVW ≈ constant ≈ BVW_irreducible
        >>> # If BVW < BVW_irr, zone is at irreducible saturation
        
    Note:
        - BVW_irreducible typically 0.02-0.05 for sandstones
        - Constant BVW indicates capillary-controlled saturation
        - BVW > BVW_irr indicates mobile water
    """
    phi_arr = to_numpy(phi)
    sw_arr = to_numpy(sw)
    index = get_index(phi)
    metadata = get_curve_metadata(phi)
    
    bvw = phi_arr * sw_arr
    
    if return_curve:
        return _make_curve(
            bvw, index,
            mnemonic='BVW',
            units='v/v',
            description='Bulk volume water (Buckles number)',
            **metadata
        )
    return bvw


def bulk_volume_water(phi: ArrayLike,
                      sw: ArrayLike,
                      return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Alias for buckles_number().
    
    See buckles_number() for documentation.
    """
    return buckles_number(phi, sw, return_curve=return_curve)
