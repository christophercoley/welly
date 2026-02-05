"""
Net pay determination and analysis.

This module provides functions for determining net pay intervals
based on petrophysical cutoffs and computing pay statistics.

:copyright: 2024 Agile Scientific
:license: Apache 2.0
"""
from typing import Union, Optional, Dict, List, Tuple

import numpy as np

from .utils import to_numpy, get_index, get_curve_metadata, ArrayLike


def _make_curve(data: np.ndarray,
                index: Optional[np.ndarray],
                mnemonic: str,
                units: str = '',
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


def net_pay_flag(phi: ArrayLike,
                 sw: ArrayLike,
                 vshale: Optional[ArrayLike] = None,
                 phi_cutoff: float = 0.08,
                 sw_cutoff: float = 0.5,
                 vsh_cutoff: float = 0.4,
                 perm: Optional[ArrayLike] = None,
                 perm_cutoff: float = 0.1,
                 return_curve: bool = True) -> Union[np.ndarray, 'Curve']:
    """
    Generate net pay flag based on petrophysical cutoffs.

    A sample is flagged as pay (1) if it meets ALL of the following:
        - Porosity >= phi_cutoff
        - Water saturation <= sw_cutoff
        - Shale volume <= vsh_cutoff (if provided)
        - Permeability >= perm_cutoff (if provided)

    Args:
        phi: Porosity (v/v)
        sw: Water saturation (v/v)
        vshale: Shale volume (v/v), optional
        phi_cutoff: Minimum porosity for pay (default 0.08 = 8%)
        sw_cutoff: Maximum water saturation for pay (default 0.5 = 50%)
        vsh_cutoff: Maximum shale volume for pay (default 0.4 = 40%)
        perm: Permeability (mD), optional
        perm_cutoff: Minimum permeability for pay (default 0.1 mD)
        return_curve: If True, return Curve object

    Returns:
        Pay flag (1=pay, 0=non-pay) as Curve or array

    Example:
        >>> # Basic pay flag
        >>> pay = net_pay_flag(phi, sw, vshale,
        ...                    phi_cutoff=0.10,
        ...                    sw_cutoff=0.50,
        ...                    vsh_cutoff=0.35)

        >>> # With permeability cutoff
        >>> pay = net_pay_flag(phi, sw, vshale,
        ...                    phi_cutoff=0.08,
        ...                    sw_cutoff=0.60,
        ...                    vsh_cutoff=0.40,
        ...                    perm=k,
        ...                    perm_cutoff=1.0)

    Note:
        - Cutoffs should be calibrated to production data
        - Different cutoffs may apply to different zones
        - Consider using zone-specific cutoffs for complex reservoirs
    """
    phi_arr = to_numpy(phi)
    sw_arr = to_numpy(sw)
    index = get_index(phi)
    metadata = get_curve_metadata(phi)

    # Initialize pay flag
    pay = np.ones_like(phi_arr, dtype=float)

    # Apply porosity cutoff
    pay[phi_arr < phi_cutoff] = 0
    pay[np.isnan(phi_arr)] = np.nan

    # Apply water saturation cutoff
    pay[sw_arr > sw_cutoff] = 0
    pay[np.isnan(sw_arr)] = np.nan

    # Apply shale volume cutoff if provided
    if vshale is not None:
        vsh_arr = to_numpy(vshale)
        pay[vsh_arr > vsh_cutoff] = 0
        pay[np.isnan(vsh_arr)] = np.nan

    # Apply permeability cutoff if provided
    if perm is not None:
        perm_arr = to_numpy(perm)
        pay[perm_arr < perm_cutoff] = 0
        pay[np.isnan(perm_arr)] = np.nan

    if return_curve:
        desc = f'Net pay flag (φ≥{phi_cutoff}, Sw≤{sw_cutoff}'
        if vshale is not None:
            desc += f', Vsh≤{vsh_cutoff}'
        if perm is not None:
            desc += f', k≥{perm_cutoff}'
        desc += ')'

        return _make_curve(
            pay, index,
            mnemonic='PAY_FLAG',
            units='flag',
            description=desc,
            **metadata
        )
    return pay


def net_to_gross(pay_flag: ArrayLike,
                 top: Optional[float] = None,
                 base: Optional[float] = None,
                 index: Optional[ArrayLike] = None) -> float:
    """
    Calculate net-to-gross ratio for an interval.

    N/G = Net Pay Thickness / Gross Thickness

    Args:
        pay_flag: Pay flag array (1=pay, 0=non-pay)
        top: Top depth of interval (optional)
        base: Base depth of interval (optional)
        index: Depth index (required if top/base specified)

    Returns:
        Net-to-gross ratio (0-1)

    Example:
        >>> ntg = net_to_gross(pay_flag)
        >>> print(f"N/G = {ntg:.1%}")

        >>> # For specific interval
        >>> ntg = net_to_gross(pay_flag, top=3000, base=3100,
        ...                    index=depth)
    """
    pay_arr = to_numpy(pay_flag)

    # Handle interval selection
    if top is not None and base is not None:
        if index is None:
            raise ValueError("index required when top/base specified")
        idx_arr = to_numpy(index)
        mask = (idx_arr >= top) & (idx_arr <= base)
        pay_arr = pay_arr[mask]

    # Remove NaNs
    valid = pay_arr[~np.isnan(pay_arr)]

    if len(valid) == 0:
        return np.nan

    return np.sum(valid) / len(valid)


def pay_summary(phi: ArrayLike,
                sw: ArrayLike,
                pay_flag: ArrayLike,
                perm: Optional[ArrayLike] = None,
                index: Optional[ArrayLike] = None,
                top: Optional[float] = None,
                base: Optional[float] = None) -> Dict:
    """
    Generate comprehensive pay summary statistics.

    Args:
        phi: Porosity (v/v)
        sw: Water saturation (v/v)
        pay_flag: Pay flag (1=pay, 0=non-pay)
        perm: Permeability (mD), optional
        index: Depth index, optional
        top: Top depth of interval, optional
        base: Base depth of interval, optional

    Returns:
        Dictionary containing:
            - gross_thickness: Total interval thickness
            - net_thickness: Net pay thickness
            - net_to_gross: N/G ratio
            - avg_phi_pay: Average porosity in pay
            - avg_sw_pay: Average Sw in pay
            - avg_perm_pay: Average permeability in pay (if provided)
            - hydrocarbon_pore_volume: φ * (1-Sw) * h integrated
            - pay_intervals: List of (top, base) tuples for pay zones

    Example:
        >>> summary = pay_summary(phi, sw, pay_flag, perm=k, index=depth)
        >>> print(f"Net Pay: {summary['net_thickness']:.1f} ft")
        >>> print(f"N/G: {summary['net_to_gross']:.1%}")
        >>> print(f"Avg φ in pay: {summary['avg_phi_pay']:.1%}")
    """
    phi_arr = to_numpy(phi)
    sw_arr = to_numpy(sw)
    pay_arr = to_numpy(pay_flag)
    perm_arr = to_numpy(perm) if perm is not None else None

    # Get index
    if index is not None:
        idx_arr = to_numpy(index)
    elif hasattr(phi, 'index'):
        idx_arr = to_numpy(phi.index)
    else:
        idx_arr = np.arange(len(phi_arr))

    # Handle interval selection
    if top is not None and base is not None:
        mask = (idx_arr >= top) & (idx_arr <= base)
        phi_arr = phi_arr[mask]
        sw_arr = sw_arr[mask]
        pay_arr = pay_arr[mask]
        idx_arr = idx_arr[mask]
        if perm_arr is not None:
            perm_arr = perm_arr[mask]

    # Calculate step (assume uniform sampling)
    step = np.median(np.diff(idx_arr)) if len(idx_arr) > 1 else 1.0

    # Gross thickness
    gross = len(idx_arr) * step

    # Net thickness
    pay_samples = np.nansum(pay_arr)
    net = pay_samples * step

    # N/G
    ntg = net / gross if gross > 0 else 0

    # Pay zone mask
    pay_mask = pay_arr == 1

    # Average properties in pay
    if np.any(pay_mask):
        avg_phi = np.nanmean(phi_arr[pay_mask])
        avg_sw = np.nanmean(sw_arr[pay_mask])
    else:
        avg_phi = np.nan
        avg_sw = np.nan

    # Permeability in pay
    avg_perm = np.nan
    geom_perm = np.nan
    if perm_arr is not None and np.any(pay_mask):
        avg_perm = np.nanmean(perm_arr[pay_mask])
        # Geometric mean often more appropriate for permeability
        valid_perm = perm_arr[pay_mask & (perm_arr > 0)]
        if len(valid_perm) > 0:
            geom_perm = np.exp(np.nanmean(np.log(valid_perm)))

    # Hydrocarbon pore volume (φ * (1-Sw) * h)
    hpv = np.nansum(phi_arr[pay_mask] * (1 - sw_arr[pay_mask])) * step

    # Identify pay intervals
    pay_intervals = _find_pay_intervals(pay_arr, idx_arr)

    return {
        'gross_thickness': gross,
        'net_thickness': net,
        'net_to_gross': ntg,
        'avg_phi_pay': avg_phi,
        'avg_sw_pay': avg_sw,
        'avg_perm_pay': avg_perm,
        'geom_perm_pay': geom_perm,
        'hydrocarbon_pore_volume': hpv,
        'pay_intervals': pay_intervals,
        'n_pay_zones': len(pay_intervals),
    }


def _find_pay_intervals(pay_flag: np.ndarray,
                        index: np.ndarray) -> List[Tuple[float, float]]:
    """
    Find continuous pay intervals.

    Args:
        pay_flag: Pay flag array
        index: Depth index

    Returns:
        List of (top, base) tuples for each pay zone
    """
    intervals = []

    # Handle NaNs
    pay = np.nan_to_num(pay_flag, nan=0)

    # Find transitions
    diff = np.diff(np.concatenate([[0], pay, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for start, end in zip(starts, ends):
        if end > start:
            top = index[start]
            base = index[min(end - 1, len(index) - 1)]
            intervals.append((float(top), float(base)))

    return intervals


def reservoir_summary(phi: ArrayLike,
                      sw: ArrayLike,
                      pay_flag: ArrayLike,
                      perm: Optional[ArrayLike] = None,
                      index: Optional[ArrayLike] = None,
                      area: float = 1.0,
                      bo: float = 1.2,
                      bg: float = 0.005,
                      fluid: str = 'oil') -> Dict:
    """
    Generate reservoir volumetrics summary.

    Calculates STOIIP/GIIP and related volumetrics.

    Args:
        phi: Porosity (v/v)
        sw: Water saturation (v/v)
        pay_flag: Pay flag (1=pay, 0=non-pay)
        perm: Permeability (mD), optional
        index: Depth index
        area: Drainage area (acres)
        bo: Oil formation volume factor (rb/stb)
        bg: Gas formation volume factor (rcf/scf)
        fluid: 'oil' or 'gas'

    Returns:
        Dictionary with volumetrics including STOIIP or GIIP

    Example:
        >>> vol = reservoir_summary(phi, sw, pay,
        ...                         area=640,  # 1 section
        ...                         bo=1.25)
        >>> print(f"STOIIP: {vol['stoiip']/1e6:.1f} MMSTB")
    """
    # Get basic pay summary
    summary = pay_summary(phi, sw, pay_flag, perm, index)

    phi_arr = to_numpy(phi)
    sw_arr = to_numpy(sw)
    pay_arr = to_numpy(pay_flag)

    pay_mask = pay_arr == 1

    # Calculate hydrocarbon pore volume
    # HPV = φ * (1 - Sw) * h
    if hasattr(phi, 'index'):
        idx = to_numpy(phi.index)
        step = np.median(np.diff(idx)) if len(idx) > 1 else 1.0
    else:
        step = 1.0

    hpv_ft = np.nansum(phi_arr[pay_mask] * (1 - sw_arr[pay_mask])) * step

    # Convert to acre-feet
    hpv_acre_ft = hpv_ft * area

    # Calculate STOIIP or GIIP
    # STOIIP = 7758 * A * h * φ * (1-Sw) / Bo  [STB]
    # GIIP = 43560 * A * h * φ * (1-Sw) / Bg  [SCF]

    if fluid.lower() == 'oil':
        stoiip = 7758 * hpv_acre_ft / bo
        summary['stoiip'] = stoiip
        summary['stoiip_mmstb'] = stoiip / 1e6
    else:
        giip = 43560 * hpv_acre_ft / bg
        summary['giip'] = giip
        summary['giip_bcf'] = giip / 1e9

    summary['hpv_acre_ft'] = hpv_acre_ft
    summary['area_acres'] = area

    return summary
