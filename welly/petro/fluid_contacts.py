"""
Fluid contact detection and analysis.

This module provides methods for detecting fluid contacts (OWC, GOC, FWL)
from well log data using various techniques.

:copyright: 2024 Agile Scientific
:license: Apache 2.0
"""
import numpy as np
from typing import Union, Optional, Tuple, List, Dict
import warnings

from .utils import (
    to_numpy, get_index, ArrayLike, safe_divide
)


def detect_owc(sw: ArrayLike,
               index: ArrayLike,
               sw_threshold: float = 0.5,
               method: str = 'threshold',
               window: int = 5,
               min_depth: Optional[float] = None,
               max_depth: Optional[float] = None) -> Optional[float]:
    """
    Detect Oil-Water Contact (OWC) from water saturation log.
    
    Methods:
        'threshold': Find depth where Sw first exceeds threshold
        'gradient': Find depth of maximum Sw gradient
        'inflection': Find inflection point in Sw curve
    
    Args:
        sw: Water saturation log (v/v)
        index: Depth index
        sw_threshold: Sw threshold for 'threshold' method (default 0.5)
        method: Detection method ('threshold', 'gradient', 'inflection')
        window: Smoothing window for gradient methods
        min_depth: Minimum depth to search
        max_depth: Maximum depth to search
        
    Returns:
        OWC depth or None if not detected
        
    Example:
        >>> owc = detect_owc(sw, depth, sw_threshold=0.5)
        >>> if owc:
        ...     print(f"OWC at {owc:.1f} ft")
        
    Note:
        - Results should be validated against other data (pressure, DST)
        - Multiple contacts may exist in stacked reservoirs
        - Transition zone effects can complicate detection
    """
    sw_arr = to_numpy(sw)
    idx_arr = to_numpy(index)
    
    # Apply depth limits
    if min_depth is not None:
        mask = idx_arr >= min_depth
        sw_arr = sw_arr[mask]
        idx_arr = idx_arr[mask]
    if max_depth is not None:
        mask = idx_arr <= max_depth
        sw_arr = sw_arr[mask]
        idx_arr = idx_arr[mask]
    
    if len(sw_arr) < window:
        return None
    
    # Remove NaNs for analysis
    valid = ~np.isnan(sw_arr)
    if np.sum(valid) < window:
        return None
    
    if method == 'threshold':
        # Find first depth where Sw exceeds threshold
        above_threshold = sw_arr > sw_threshold
        if not np.any(above_threshold):
            return None
        
        # Find first occurrence
        first_idx = np.argmax(above_threshold)
        return float(idx_arr[first_idx])
    
    elif method == 'gradient':
        # Find maximum gradient in Sw
        # Smooth first
        sw_smooth = np.convolve(sw_arr, np.ones(window)/window, mode='same')
        gradient = np.gradient(sw_smooth, idx_arr)
        
        # Find maximum positive gradient (Sw increasing with depth)
        max_grad_idx = np.argmax(gradient)
        return float(idx_arr[max_grad_idx])
    
    elif method == 'inflection':
        # Find inflection point (second derivative = 0)
        sw_smooth = np.convolve(sw_arr, np.ones(window)/window, mode='same')
        d1 = np.gradient(sw_smooth, idx_arr)
        d2 = np.gradient(d1, idx_arr)
        
        # Find zero crossings of second derivative
        sign_changes = np.where(np.diff(np.sign(d2)))[0]
        
        if len(sign_changes) == 0:
            return None
        
        # Return deepest inflection point (likely OWC)
        return float(idx_arr[sign_changes[-1]])
    
    else:
        raise ValueError(f"Unknown method: {method}")


def detect_goc(rhob: ArrayLike,
               nphi: ArrayLike,
               index: ArrayLike,
               method: str = 'crossover',
               window: int = 5,
               min_depth: Optional[float] = None,
               max_depth: Optional[float] = None) -> Optional[float]:
    """
    Detect Gas-Oil Contact (GOC) from density-neutron logs.
    
    Gas causes:
        - Density to decrease (gas is light)
        - Neutron to decrease (low hydrogen index)
    
    Methods:
        'crossover': Find depth where density porosity > neutron porosity
        'separation': Find depth of maximum N-D separation
    
    Args:
        rhob: Bulk density log (g/cc)
        nphi: Neutron porosity log (v/v)
        index: Depth index
        method: Detection method ('crossover', 'separation')
        window: Smoothing window
        min_depth: Minimum depth to search
        max_depth: Maximum depth to search
        
    Returns:
        GOC depth or None if not detected
        
    Example:
        >>> goc = detect_goc(rhob, nphi, depth)
        >>> if goc:
        ...     print(f"GOC at {goc:.1f} ft")
        
    Note:
        - Gas effect is most pronounced in high-porosity zones
        - Shale can mask gas effect
        - Validate with resistivity and other indicators
    """
    rhob_arr = to_numpy(rhob)
    nphi_arr = to_numpy(nphi)
    idx_arr = to_numpy(index)
    
    # Apply depth limits
    if min_depth is not None:
        mask = idx_arr >= min_depth
        rhob_arr = rhob_arr[mask]
        nphi_arr = nphi_arr[mask]
        idx_arr = idx_arr[mask]
    if max_depth is not None:
        mask = idx_arr <= max_depth
        rhob_arr = rhob_arr[mask]
        nphi_arr = nphi_arr[mask]
        idx_arr = idx_arr[mask]
    
    if len(rhob_arr) < window:
        return None
    
    # Calculate density porosity (assuming sandstone matrix)
    rho_matrix = 2.65
    rho_fluid = 1.0
    phi_d = (rho_matrix - rhob_arr) / (rho_matrix - rho_fluid)
    
    # N-D separation (positive = gas effect)
    separation = phi_d - nphi_arr
    
    if method == 'crossover':
        # Find where density porosity exceeds neutron (gas effect)
        gas_flag = separation > 0.02  # Small threshold to avoid noise
        
        if not np.any(gas_flag):
            return None
        
        # Find shallowest gas (top of gas cap)
        first_gas = np.argmax(gas_flag)
        return float(idx_arr[first_gas])
    
    elif method == 'separation':
        # Find maximum separation
        sep_smooth = np.convolve(separation, np.ones(window)/window, mode='same')
        max_sep_idx = np.argmax(sep_smooth)
        
        if sep_smooth[max_sep_idx] < 0.02:
            return None
        
        return float(idx_arr[max_sep_idx])
    
    else:
        raise ValueError(f"Unknown method: {method}")


def detect_fwl(sw: ArrayLike,
               index: ArrayLike,
               phi: Optional[ArrayLike] = None,
               bvw_irreducible: float = 0.035,
               method: str = 'buckles',
               window: int = 5) -> Optional[float]:
    """
    Detect Free Water Level (FWL) from saturation data.
    
    The FWL is the depth where capillary pressure equals zero,
    below which the formation is 100% water saturated.
    
    Methods:
        'buckles': Use Buckles relationship (BVW = constant in transition zone)
        'sw_unity': Find depth where Sw approaches 1.0
    
    Args:
        sw: Water saturation log (v/v)
        index: Depth index
        phi: Porosity log (required for 'buckles' method)
        bvw_irreducible: Irreducible BVW for Buckles method
        method: Detection method ('buckles', 'sw_unity')
        window: Smoothing window
        
    Returns:
        FWL depth or None if not detected
        
    Example:
        >>> fwl = detect_fwl(sw, depth, phi, bvw_irreducible=0.04)
        >>> if fwl:
        ...     print(f"FWL at {fwl:.1f} ft")
        
    Note:
        - FWL is typically deeper than OWC
        - OWC = FWL - (capillary pressure offset)
        - Buckles method assumes transition zone behavior
    """
    sw_arr = to_numpy(sw)
    idx_arr = to_numpy(index)
    
    if method == 'buckles':
        if phi is None:
            raise ValueError("phi required for 'buckles' method")
        
        phi_arr = to_numpy(phi)
        
        # Calculate BVW
        bvw = phi_arr * sw_arr
        
        # In transition zone, BVW ≈ constant
        # Below FWL, BVW = φ (since Sw = 1)
        # Find where BVW starts increasing significantly
        
        bvw_smooth = np.convolve(bvw, np.ones(window)/window, mode='same')
        bvw_gradient = np.gradient(bvw_smooth, idx_arr)
        
        # FWL is where BVW gradient becomes large (leaving transition zone)
        threshold = 0.001  # BVW/ft
        high_gradient = bvw_gradient > threshold
        
        if not np.any(high_gradient):
            return None
        
        fwl_idx = np.argmax(high_gradient)
        return float(idx_arr[fwl_idx])
    
    elif method == 'sw_unity':
        # Find where Sw approaches 1.0
        sw_smooth = np.convolve(sw_arr, np.ones(window)/window, mode='same')
        
        # Find first depth where Sw > 0.95
        water_zone = sw_smooth > 0.95
        
        if not np.any(water_zone):
            return None
        
        fwl_idx = np.argmax(water_zone)
        return float(idx_arr[fwl_idx])
    
    else:
        raise ValueError(f"Unknown method: {method}")


def gradient_intersection(pressure: ArrayLike,
                          depth: ArrayLike,
                          fluid1_gradient: float,
                          fluid2_gradient: float,
                          fluid1_depth_range: Tuple[float, float],
                          fluid2_depth_range: Tuple[float, float]) -> Optional[float]:
    """
    Calculate fluid contact from pressure gradient intersection.
    
    This is the most reliable method for determining fluid contacts
    when pressure data (MDT, RFT, DST) is available.
    
    Args:
        pressure: Pressure measurements (psi)
        depth: Depth of measurements (ft TVD)
        fluid1_gradient: Expected gradient of upper fluid (psi/ft)
            - Oil: ~0.35 psi/ft
            - Gas: ~0.05-0.15 psi/ft
        fluid2_gradient: Expected gradient of lower fluid (psi/ft)
            - Water: ~0.43-0.47 psi/ft
        fluid1_depth_range: (top, base) depth range for fluid 1 fit
        fluid2_depth_range: (top, base) depth range for fluid 2 fit
        
    Returns:
        Contact depth or None if lines don't intersect
        
    Example:
        >>> # OWC from MDT data
        >>> owc = gradient_intersection(
        ...     pressure, tvd,
        ...     fluid1_gradient=0.35,  # oil
        ...     fluid2_gradient=0.45,  # water
        ...     fluid1_depth_range=(8000, 8200),
        ...     fluid2_depth_range=(8300, 8500)
        ... )
        
    Note:
        - Use TVD for depth
        - Gradients depend on fluid properties and temperature
        - Multiple measurements in each zone improve accuracy
    """
    p_arr = to_numpy(pressure)
    d_arr = to_numpy(depth)
    
    # Fit line to fluid 1 data
    mask1 = (d_arr >= fluid1_depth_range[0]) & (d_arr <= fluid1_depth_range[1])
    if np.sum(mask1) < 2:
        warnings.warn("Insufficient data points in fluid 1 range", stacklevel=2)
        return None
    
    # Linear fit: P = m*D + b
    coef1 = np.polyfit(d_arr[mask1], p_arr[mask1], 1)
    
    # Fit line to fluid 2 data
    mask2 = (d_arr >= fluid2_depth_range[0]) & (d_arr <= fluid2_depth_range[1])
    if np.sum(mask2) < 2:
        warnings.warn("Insufficient data points in fluid 2 range", stacklevel=2)
        return None
    
    coef2 = np.polyfit(d_arr[mask2], p_arr[mask2], 1)
    
    # Find intersection
    # m1*D + b1 = m2*D + b2
    # D = (b2 - b1) / (m1 - m2)
    
    m1, b1 = coef1
    m2, b2 = coef2
    
    if abs(m1 - m2) < 1e-6:
        warnings.warn("Gradients are parallel, no intersection", stacklevel=2)
        return None
    
    contact_depth = (b2 - b1) / (m1 - m2)
    
    # Validate that intersection is between the two zones
    min_depth = min(fluid1_depth_range[0], fluid2_depth_range[0])
    max_depth = max(fluid1_depth_range[1], fluid2_depth_range[1])
    
    if contact_depth < min_depth or contact_depth > max_depth:
        warnings.warn(
            f"Intersection at {contact_depth:.1f} is outside data range",
            stacklevel=2
        )
    
    return float(contact_depth)


def analyze_transition_zone(sw: ArrayLike,
                            phi: ArrayLike,
                            index: ArrayLike,
                            owc: float,
                            fwl: Optional[float] = None) -> Dict:
    """
    Analyze transition zone characteristics.
    
    Args:
        sw: Water saturation log (v/v)
        phi: Porosity log (v/v)
        index: Depth index
        owc: Oil-water contact depth
        fwl: Free water level depth (optional)
        
    Returns:
        Dictionary with transition zone analysis:
            - transition_height: Height of transition zone
            - avg_sw_transition: Average Sw in transition
            - bvw_irreducible: Estimated irreducible BVW
            - j_function_params: J-function parameters if calculable
    """
    sw_arr = to_numpy(sw)
    phi_arr = to_numpy(phi)
    idx_arr = to_numpy(index)
    
    # Estimate FWL if not provided
    if fwl is None:
        fwl = detect_fwl(sw_arr, idx_arr, phi_arr)
        if fwl is None:
            fwl = owc + 50  # Default assumption
    
    # Transition zone mask
    tz_mask = (idx_arr >= owc) & (idx_arr <= fwl)
    
    if not np.any(tz_mask):
        return {
            'transition_height': 0,
            'avg_sw_transition': np.nan,
            'bvw_irreducible': np.nan,
        }
    
    # Calculate BVW in transition zone
    bvw_tz = phi_arr[tz_mask] * sw_arr[tz_mask]
    
    # Irreducible BVW is approximately the minimum BVW
    bvw_irr = np.nanpercentile(bvw_tz, 10)
    
    return {
        'transition_height': fwl - owc,
        'avg_sw_transition': np.nanmean(sw_arr[tz_mask]),
        'bvw_irreducible': bvw_irr,
        'owc': owc,
        'fwl': fwl,
    }
