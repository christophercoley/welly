"""
Petrophysical Interpreter for Well objects.

This module provides a high-level interface for running petrophysical
interpretations on Well objects, integrating the petro calculation
functions with welly's Well and Curve classes.

:copyright: 2024 Agile Scientific
:license: Apache 2.0
"""
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
import json
import copy

import numpy as np

from .parameters import PetrophysicalParameters
from . import (
    vshale_linear, vshale_larionov, vshale_steiber, vshale_clavier,
    porosity_density, porosity_neutron, porosity_sonic_wyllie,
    porosity_sonic_raymer, porosity_neutron_density, porosity_effective,
    archie, simandoux, indonesia, fertl, waxman_smits, dual_water,
    buckles_number,
    perm_timur, perm_coates, perm_tixier,
    net_pay_flag, net_to_gross,
    detect_owc,
)


# Default alias dictionary for common curve mnemonics
DEFAULT_ALIAS = {
    'GR': ['GR', 'GRD', 'GRR', 'GRGC', 'SGR', 'CGR', 'GR_EDTC', 'GAMMA'],
    'SP': ['SP', 'SSP'],
    'CALI': ['CALI', 'CAL', 'HCAL', 'BS', 'CALIPER'],
    'RHOB': ['RHOB', 'RHOZ', 'DEN', 'DENS', 'ZDEN', 'BDCOR', 'DENSITY'],
    'NPHI': ['NPHI', 'NPOR', 'TNPH', 'NEU', 'NPHI_LS', 'NPHI_SS', 'NPHI_DOL',
             'NPHI_LIM', 'NPHI_SAN', 'NEUTRON', 'PHIN'],
    'DT': ['DT', 'DTC', 'AC', 'DTCO', 'DT4P', 'DTLN', 'SONIC'],
    'DTS': ['DTS', 'DTSM'],
    'RT': ['RT', 'ILD', 'RILD', 'RD', 'RLLD', 'LLD', 'RT_HRLT', 'HDRS',
           'RESD', 'AT90', 'AHT90', 'RLA5'],
    'RXO': ['RXO', 'RXOZ', 'RSFL', 'RFOC', 'RXO_HRLT', 'HMRS', 'MSFL',
            'SFL', 'RLA1'],
    'PEF': ['PEF', 'PE', 'PEFZ'],
    'DRHO': ['DRHO', 'DCOR'],
}

# Define what curves are required for each calculation
CALCULATION_REQUIREMENTS = {
    'vshale_gr': {'required': ['GR'], 'optional': []},
    'vshale_sp': {'required': ['SP'], 'optional': []},
    'vshale_nd': {'required': ['NPHI', 'RHOB'], 'optional': []},
    'porosity_density': {'required': ['RHOB'], 'optional': []},
    'porosity_neutron': {'required': ['NPHI'], 'optional': []},
    'porosity_sonic': {'required': ['DT'], 'optional': []},
    'porosity_neutron_density': {'required': ['NPHI', 'RHOB'], 'optional': []},
    'sw_archie': {'required': ['RT'], 'optional': []},
    'sw_shaly': {'required': ['RT'], 'optional': ['VSH']},
    'permeability': {'required': [], 'optional': []},  # Uses computed curves
    'net_pay': {'required': [], 'optional': ['VSH']},  # Uses computed curves
}


class PetroInterpreter:
    """
    High-level petrophysical interpreter for Well objects.

    The PetroInterpreter provides a convenient interface for running
    petrophysical calculations on well log data. It handles:

    - Curve lookup with alias support (GR, GRGC, GRD all map to gamma ray)
    - Parameter management via PetrophysicalParameters
    - Automatic storage of results back to the Well
    - Input validation and missing curve handling
    - Standard interpretation workflows

    Args:
        well: A welly.Well object containing log curves
        params: PetrophysicalParameters for the interpretation.
            If None, uses default sandstone parameters.
        alias: Dictionary to extend/override default aliases.
            Merged with DEFAULT_ALIAS (not replaced).
        alias_file: Path to JSON file containing alias definitions.

    Example:
        >>> from welly import Well
        >>> from welly.petro import PetroInterpreter, PetrophysicalParameters
        >>>
        >>> # Load well and create interpreter
        >>> well = Well.from_las('my_well.las')
        >>> interp = well.petro()
        >>>
        >>> # Check what calculations are possible
        >>> status = interp.check_inputs()
        >>> print(status)
        >>>
        >>> # Run interpretation
        >>> interp.run_standard_interpretation()

    Attributes:
        well: The Well object being interpreted
        params: PetrophysicalParameters used for calculations
        alias: Alias dictionary for curve lookup
        results: Dictionary tracking which curves have been computed
    """

    def __init__(self,
                 well: 'Well',
                 params: Optional[PetrophysicalParameters] = None,
                 alias: Optional[Dict[str, List[str]]] = None,
                 alias_file: Optional[Union[str, Path]] = None):
        self.well = well
        self.params = params or PetrophysicalParameters.default_sandstone()
        self.results = {}  # Track computed curves

        # Initialize aliases from defaults
        self._alias = copy.deepcopy(DEFAULT_ALIAS)

        # Load from file if specified
        if alias_file:
            self.load_aliases(alias_file)

        # Merge with provided alias dict
        if alias:
            self.update_aliases(alias)

    # =========================================================================
    # Alias Management
    # =========================================================================

    @property
    def alias(self) -> Dict[str, List[str]]:
        """Get current alias dictionary."""
        return self._alias

    def update_aliases(self, alias_dict: Dict[str, List[str]],
                       replace: bool = False) -> None:
        """
        Update alias dictionary.

        Args:
            alias_dict: Dictionary mapping standard names to mnemonic lists
            replace: If True, replace entire alias dict. If False (default),
                     merge with existing aliases.

        Example:
            >>> interp.update_aliases({'GR': ['GAMMA_RAY', 'GR_CORR']})
            >>> # Now GR will also match GAMMA_RAY and GR_CORR
        """
        if replace:
            self._alias = copy.deepcopy(alias_dict)
        else:
            for key, values in alias_dict.items():
                if key in self._alias:
                    # Extend existing, avoiding duplicates, preserving order
                    existing = self._alias[key]
                    for v in values:
                        if v not in existing:
                            existing.append(v)
                else:
                    self._alias[key] = list(values)

    def add_alias(self, standard_name: str, mnemonics: List[str]) -> None:
        """
        Add mnemonics to an alias group.

        Args:
            standard_name: Standard curve name (e.g., 'GR', 'RHOB')
            mnemonics: List of mnemonics to add

        Example:
            >>> interp.add_alias('GR', ['GAMMA_RAY', 'GR_CORR'])
        """
        self.update_aliases({standard_name: mnemonics})

    def remove_alias(self, standard_name: str,
                     mnemonics: Optional[List[str]] = None) -> None:
        """
        Remove mnemonics from an alias group, or remove entire group.

        Args:
            standard_name: Standard curve name
            mnemonics: Specific mnemonics to remove. If None, removes entire group.
        """
        if standard_name not in self._alias:
            return

        if mnemonics is None:
            del self._alias[standard_name]
        else:
            self._alias[standard_name] = [
                m for m in self._alias[standard_name] if m not in mnemonics
            ]

    def load_aliases(self, filepath: Union[str, Path]) -> None:
        """
        Load aliases from a JSON file.

        File format:
        {
            "aliases": {
                "GR": ["GR", "GRGC", "GRD"],
                "RHOB": ["RHOB", "DEN", "DENS"]
            },
            "description": "Optional description"
        }

        Args:
            filepath: Path to JSON file
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Alias file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if 'aliases' in data:
            self.update_aliases(data['aliases'])
        else:
            # Assume entire file is alias dict
            self.update_aliases(data)

    def save_aliases(self, filepath: Union[str, Path],
                     description: str = '') -> None:
        """
        Save current aliases to a JSON file.

        Args:
            filepath: Path to save to
            description: Optional description to include
        """
        filepath = Path(filepath)
        data = {
            'aliases': self._alias,
            'description': description,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def get_alias_matches(self) -> Dict[str, Optional[str]]:
        """
        Show which curves in the well match each standard alias.

        Returns:
            Dictionary mapping standard names to matched curve names (or None)

        Example:
            >>> matches = interp.get_alias_matches()
            >>> print(matches)
            {'GR': 'GRGC', 'RHOB': 'RHOB', 'NPHI': None, ...}
        """
        matches = {}
        for std_name in self._alias.keys():
            curve = self._get_curve(std_name, required=False)
            if curve is not None:
                matches[std_name] = curve.mnemonic
            else:
                matches[std_name] = None
        return matches


    # =========================================================================
    # Input Checking and Curve Access
    # =========================================================================

    def _get_curve(self, mnemonic: str, required: bool = True):
        """
        Get a curve from the well using alias lookup.

        Args:
            mnemonic: Standard curve name (e.g., 'GR', 'RHOB') or actual mnemonic
            required: If True, raise error if curve not found

        Returns:
            Curve object or None
        """
        # First try direct lookup (handles computed curves like VSH, PHIE)
        if mnemonic in self.well.data:
            return self.well.data[mnemonic]

        # Try alias lookup
        aliases = self._alias.get(mnemonic, [mnemonic])
        for alias in aliases:
            if alias in self.well.data:
                return self.well.data[alias]

        if required:
            available = list(self.well.data.keys())
            raise MissingCurveError(
                mnemonic=mnemonic,
                tried=aliases,
                available=available
            )
        return None

    def _get_curve_name(self, mnemonic: str) -> Optional[str]:
        """Get the actual curve name in the well for a standard mnemonic."""
        curve = self._get_curve(mnemonic, required=False)
        return curve.mnemonic if curve is not None else None

    def _store_curve(self, name: str, curve: 'Curve'):
        """Store a computed curve in the well."""
        self.well.data[name] = curve
        self.results[name] = True

    def check_inputs(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Check what input curves are available and what calculations are possible.

        Returns a dictionary showing:
        - Which standard curves are found (and their actual names)
        - Which calculations can be performed
        - Which calculations are missing required inputs

        Args:
            verbose: If True, print a formatted summary

        Returns:
            Dictionary with input status

        Example:
            >>> status = interp.check_inputs(verbose=True)
            Available curves:
              GR -> GRGC ✓
              RHOB -> RHOB ✓
              NPHI -> (not found) ✗
              RT -> RT_HRLT ✓

            Possible calculations:
              vshale_gr: ✓ ready
              porosity_density: ✓ ready
              porosity_neutron_density: ✗ missing NPHI
        """
        result = {
            'curves_found': {},
            'curves_missing': [],
            'calculations_ready': [],
            'calculations_blocked': {},
        }

        # Check each standard curve
        for std_name in self._alias.keys():
            actual = self._get_curve_name(std_name)
            if actual:
                result['curves_found'][std_name] = actual
            else:
                result['curves_missing'].append(std_name)

        # Check each calculation
        for calc_name, reqs in CALCULATION_REQUIREMENTS.items():
            missing = []
            for req in reqs['required']:
                if req not in result['curves_found']:
                    missing.append(req)

            if not missing:
                result['calculations_ready'].append(calc_name)
            else:
                result['calculations_blocked'][calc_name] = missing

        if verbose:
            self._print_input_status(result)

        return result

    def _print_input_status(self, status: Dict) -> None:
        """Print formatted input status."""
        print("\nAvailable curves:")
        for std_name in sorted(self._alias.keys()):
            actual = status['curves_found'].get(std_name)
            if actual:
                print(f"  {std_name:8} -> {actual} ✓")
            else:
                print(f"  {std_name:8} -> (not found) ✗")

        print("\nPossible calculations:")
        for calc in sorted(CALCULATION_REQUIREMENTS.keys()):
            if calc in status['calculations_ready']:
                print(f"  {calc}: ✓ ready")
            else:
                missing = status['calculations_blocked'].get(calc, [])
                print(f"  {calc}: ✗ missing {', '.join(missing)}")

    def available_curves(self) -> List[str]:
        """List all curves available in the well."""
        return list(self.well.data.keys())

    def list_aliases(self) -> None:
        """Print all current aliases."""
        print("Current aliases:")
        for std_name, mnemonics in sorted(self._alias.items()):
            print(f"  {std_name}: {mnemonics}")

    # =========================================================================
    # Calculation Methods
    # =========================================================================

    def vshale(self,
               method: str = 'larionov',
               gr_clean: Optional[float] = None,
               gr_shale: Optional[float] = None,
               output: str = 'VSH') -> 'Curve':
        """
        Calculate shale volume from gamma ray.

        Args:
            method: 'linear', 'larionov', 'steiber', or 'clavier'
            gr_clean: Clean sand GR value. Uses params.clay.gr_clean if None.
            gr_shale: Shale GR value. Uses params.clay.gr_shale if None.
            output: Output curve name

        Returns:
            Curve object with Vshale values
        """
        gr = self._get_curve('GR')

        gr_clean = gr_clean or self.params.clay.gr_clean
        gr_shale = gr_shale or self.params.clay.gr_shale

        method_map = {
            'linear': vshale_linear,
            'larionov': vshale_larionov,
            'steiber': vshale_steiber,
            'clavier': vshale_clavier,
        }

        if method not in method_map:
            raise ValueError(f"Unknown method: {method}. Use one of {list(method_map.keys())}")

        vsh = method_map[method](gr, gr_clean, gr_shale)
        vsh.mnemonic = output

        self._store_curve(output, vsh)
        return vsh

    def porosity(self,
                 method: str = 'density',
                 output: str = None,
                 **kwargs) -> 'Curve':
        """
        Calculate porosity using specified method.

        Args:
            method: 'density', 'neutron', 'sonic_wyllie', 'sonic_raymer', 'neutron_density'
            output: Output curve name. Defaults based on method.
            **kwargs: Additional parameters passed to calculation function

        Returns:
            Curve object with porosity values
        """
        output_names = {
            'density': 'PHID',
            'neutron': 'PHIN',
            'sonic_wyllie': 'PHIS',
            'sonic_raymer': 'PHIS',
            'neutron_density': 'PHIT',
        }
        output = output or output_names.get(method, 'PHI')

        if method == 'density':
            rhob = self._get_curve('RHOB')
            rho_matrix = kwargs.get('rho_matrix', self.params.matrix.rho_matrix)
            rho_fluid = kwargs.get('rho_fluid', self.params.fluid.rho_fluid)
            phi = porosity_density(rhob, rho_matrix=rho_matrix, rho_fluid=rho_fluid)

        elif method == 'neutron':
            nphi = self._get_curve('NPHI')
            phi = porosity_neutron(nphi, **kwargs)

        elif method == 'sonic_wyllie':
            dt = self._get_curve('DT')
            dt_matrix = kwargs.get('dt_matrix', self.params.matrix.dt_matrix)
            dt_fluid = kwargs.get('dt_fluid', self.params.fluid.dt_fluid)
            phi = porosity_sonic_wyllie(dt, dt_matrix=dt_matrix, dt_fluid=dt_fluid)

        elif method == 'sonic_raymer':
            dt = self._get_curve('DT')
            dt_matrix = kwargs.get('dt_matrix', self.params.matrix.dt_matrix)
            phi = porosity_sonic_raymer(dt, dt_matrix=dt_matrix)

        elif method == 'neutron_density':
            nphi = self._get_curve('NPHI')
            rhob = self._get_curve('RHOB')
            rho_matrix = kwargs.get('rho_matrix', self.params.matrix.rho_matrix)
            rho_fluid = kwargs.get('rho_fluid', self.params.fluid.rho_fluid)
            phi = porosity_neutron_density(nphi, rhob, rho_matrix=rho_matrix,
                                           rho_fluid=rho_fluid, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        phi.mnemonic = output
        self._store_curve(output, phi)
        return phi

    def sw(self,
           method: str = 'archie',
           phi: Optional[Union[str, 'Curve', np.ndarray]] = None,
           vsh: Optional[Union[str, 'Curve', np.ndarray]] = None,
           output: str = 'SW',
           **kwargs) -> 'Curve':
        """
        Calculate water saturation.

        Args:
            method: 'archie', 'simandoux', 'indonesia', 'fertl', 'waxman_smits', 'dual_water'
            phi: Porosity curve name, Curve object, or array. Uses 'PHIE' or 'PHIT' if None.
            vsh: Vshale curve name, Curve object, or array. Required for shaly-sand methods.
            output: Output curve name
            **kwargs: Additional parameters

        Returns:
            Curve object with Sw values
        """
        rt = self._get_curve('RT')

        # Get porosity
        if phi is None:
            phi_curve = self._get_curve('PHIE', required=False)
            if phi_curve is None:
                phi_curve = self._get_curve('PHIT', required=False)
            if phi_curve is None:
                raise MissingCurveError(
                    mnemonic='PHIE/PHIT',
                    tried=['PHIE', 'PHIT'],
                    available=list(self.well.data.keys())
                )
            phi_arr = phi_curve.values
        elif isinstance(phi, str):
            phi_arr = self._get_curve(phi).values
        elif hasattr(phi, 'values'):
            phi_arr = phi.values
        else:
            phi_arr = np.asarray(phi)

        # Ensure valid porosity
        phi_arr = np.clip(phi_arr, 0.01, 1.0)

        # Get Vshale for shaly-sand methods
        vsh_arr = None
        if method in ['simandoux', 'indonesia', 'fertl', 'waxman_smits', 'dual_water']:
            if vsh is None:
                vsh_curve = self._get_curve('VSH', required=False)
                if vsh_curve is not None:
                    vsh_arr = vsh_curve.values
            elif isinstance(vsh, str):
                vsh_arr = self._get_curve(vsh).values
            elif hasattr(vsh, 'values'):
                vsh_arr = vsh.values
            else:
                vsh_arr = np.asarray(vsh)

        # Get parameters
        rw = kwargs.get('rw', self.params.fluid.rw)
        a = kwargs.get('a', self.params.a)
        m = kwargs.get('m', self.params.m)
        n = kwargs.get('n', self.params.n)

        if method == 'archie':
            sw_curve = archie(rt, phi_arr, rw=rw, a=a, m=m, n=n)

        elif method == 'simandoux':
            rsh = kwargs.get('rsh', self.params.clay.rsh if self.params.clay else 5.0)
            if vsh_arr is None:
                vsh_arr = np.zeros_like(phi_arr)
            sw_curve = simandoux(rt, phi_arr, vsh_arr, rw=rw, rsh=rsh, a=a, m=m, n=n)

        elif method == 'indonesia':
            rsh = kwargs.get('rsh', self.params.clay.rsh if self.params.clay else 5.0)
            if vsh_arr is None:
                vsh_arr = np.zeros_like(phi_arr)
            sw_curve = indonesia(rt, phi_arr, vsh_arr, rw=rw, rsh=rsh, a=a, m=m, n=n)

        elif method == 'fertl':
            alpha = kwargs.get('alpha', 0.25)
            if vsh_arr is None:
                vsh_arr = np.zeros_like(phi_arr)
            sw_curve = fertl(rt, phi_arr, vsh_arr, rw=rw, alpha=alpha, a=a, m=m, n=n)

        elif method == 'waxman_smits':
            qv = kwargs.get('qv', 0.0)
            temp = kwargs.get('temp', 150.0)  # Default formation temp in °F
            sw_curve = waxman_smits(rt, phi_arr, qv, rw=rw, temp=temp, a=a, m_star=m, n_star=n)

        elif method == 'dual_water':
            # Dual water needs both total and effective porosity
            phi_t = kwargs.get('phi_t', phi_arr)
            phi_e = kwargs.get('phi_e', phi_arr * 0.9)  # Approximate if not provided
            rwb = kwargs.get('rwb', 0.4)  # Typical bound water resistivity
            sw_curve = dual_water(rt, phi_t, phi_e, rw=rw, rwb=rwb, a=a, m=m, n=n)

        else:
            raise ValueError(f"Unknown method: {method}")

        sw_curve.mnemonic = output
        self._store_curve(output, sw_curve)
        return sw_curve

    def permeability(self,
                     method: str = 'timur',
                     phi: Optional[Union[str, 'Curve', np.ndarray]] = None,
                     swirr: Optional[Union[str, 'Curve', np.ndarray]] = None,
                     output: str = 'PERM',
                     **kwargs) -> 'Curve':
        """
        Calculate permeability.

        Args:
            method: 'timur', 'coates', 'tixier'
            phi: Porosity curve/array. Uses PHIE if None.
            swirr: Irreducible water saturation curve/array. Uses SW if None.
            output: Output curve name
            **kwargs: Additional parameters

        Returns:
            Curve object with permeability values (mD)
        """
        # Get porosity
        if phi is None:
            phi_curve = self._get_curve('PHIE', required=False)
            if phi_curve is None:
                phi_curve = self._get_curve('PHIT', required=False)
            if phi_curve is None:
                raise MissingCurveError('PHIE/PHIT', ['PHIE', 'PHIT'], list(self.well.data.keys()))
            phi_arr = phi_curve.values
        elif isinstance(phi, str):
            phi_arr = self._get_curve(phi).values
        elif hasattr(phi, 'values'):
            phi_arr = phi.values
        else:
            phi_arr = np.asarray(phi)

        # Get Swirr
        if swirr is None:
            sw_curve = self._get_curve('SW', required=False)
            if sw_curve is not None:
                swirr_arr = sw_curve.values
            else:
                swirr_arr = np.full_like(phi_arr, 0.2)  # Default assumption
        elif isinstance(swirr, str):
            swirr_arr = self._get_curve(swirr).values
        elif hasattr(swirr, 'values'):
            swirr_arr = swirr.values
        else:
            swirr_arr = np.asarray(swirr)

        # Ensure valid values
        phi_arr = np.clip(phi_arr, 0.01, 1.0)
        swirr_arr = np.clip(swirr_arr, 0.01, 1.0)

        method_map = {
            'timur': perm_timur,
            'coates': perm_coates,
            'tixier': perm_tixier,
        }

        if method not in method_map:
            raise ValueError(f"Unknown method: {method}. Use one of {list(method_map.keys())}")

        k = method_map[method](phi_arr, swirr_arr, **kwargs)
        k.mnemonic = output

        self._store_curve(output, k)
        return k

    def net_pay(self,
                phi: Optional[Union[str, 'Curve', np.ndarray]] = None,
                sw: Optional[Union[str, 'Curve', np.ndarray]] = None,
                vsh: Optional[Union[str, 'Curve', np.ndarray]] = None,
                phi_cutoff: float = 0.08,
                sw_cutoff: float = 0.50,
                vsh_cutoff: float = 0.40,
                output: str = 'PAY') -> 'Curve':
        """
        Calculate net pay flag.

        Args:
            phi: Porosity curve/array
            sw: Water saturation curve/array
            vsh: Shale volume curve/array
            phi_cutoff: Minimum porosity for pay
            sw_cutoff: Maximum Sw for pay
            vsh_cutoff: Maximum Vsh for pay
            output: Output curve name

        Returns:
            Curve object with pay flag (1=pay, 0=non-pay)
        """
        # Get curves
        if phi is None:
            phi_curve = self._get_curve('PHIE', required=False) or self._get_curve('PHIT', required=False)
            if phi_curve is None:
                raise MissingCurveError('PHIE/PHIT', ['PHIE', 'PHIT'], list(self.well.data.keys()))
            phi_arr = phi_curve.values
        elif isinstance(phi, str):
            phi_arr = self._get_curve(phi).values
        elif hasattr(phi, 'values'):
            phi_arr = phi.values
        else:
            phi_arr = np.asarray(phi)

        if sw is None:
            sw_curve = self._get_curve('SW', required=False)
            if sw_curve is None:
                raise MissingCurveError('SW', ['SW'], list(self.well.data.keys()))
            sw_arr = sw_curve.values
        elif isinstance(sw, str):
            sw_arr = self._get_curve(sw).values
        elif hasattr(sw, 'values'):
            sw_arr = sw.values
        else:
            sw_arr = np.asarray(sw)

        if vsh is None:
            vsh_curve = self._get_curve('VSH', required=False)
            vsh_arr = vsh_curve.values if vsh_curve is not None else np.zeros_like(phi_arr)
        elif isinstance(vsh, str):
            vsh_arr = self._get_curve(vsh).values
        elif hasattr(vsh, 'values'):
            vsh_arr = vsh.values
        else:
            vsh_arr = np.asarray(vsh)

        pay = net_pay_flag(phi_arr, sw_arr, vsh_arr,
                           phi_cutoff=phi_cutoff, sw_cutoff=sw_cutoff, vsh_cutoff=vsh_cutoff)
        pay.mnemonic = output

        self._store_curve(output, pay)
        return pay

    def buckles(self,
                phi: Optional[Union[str, 'Curve', np.ndarray]] = None,
                sw: Optional[Union[str, 'Curve', np.ndarray]] = None,
                output: str = 'BVW') -> 'Curve':
        """
        Calculate Buckles number (bulk volume water).

        Args:
            phi: Porosity curve/array
            sw: Water saturation curve/array
            output: Output curve name

        Returns:
            Curve object with BVW values
        """
        if phi is None:
            phi_curve = self._get_curve('PHIE', required=False) or self._get_curve('PHIT', required=False)
            if phi_curve is None:
                raise MissingCurveError('PHIE/PHIT', ['PHIE', 'PHIT'], list(self.well.data.keys()))
            phi_arr = phi_curve.values
        elif isinstance(phi, str):
            phi_arr = self._get_curve(phi).values
        elif hasattr(phi, 'values'):
            phi_arr = phi.values
        else:
            phi_arr = np.asarray(phi)

        if sw is None:
            sw_curve = self._get_curve('SW', required=False)
            if sw_curve is None:
                raise MissingCurveError('SW', ['SW'], list(self.well.data.keys()))
            sw_arr = sw_curve.values
        elif isinstance(sw, str):
            sw_arr = self._get_curve(sw).values
        elif hasattr(sw, 'values'):
            sw_arr = sw.values
        else:
            sw_arr = np.asarray(sw)

        bvw = buckles_number(phi_arr, sw_arr)
        bvw.mnemonic = output

        self._store_curve(output, bvw)
        return bvw

    # =========================================================================
    # Workflow Methods
    # =========================================================================

    def run_standard_interpretation(self,
                                    vshale_method: str = 'larionov',
                                    porosity_method: str = 'density',
                                    sw_method: str = 'archie',
                                    compute_perm: bool = True,
                                    compute_pay: bool = True,
                                    phi_cutoff: float = 0.08,
                                    sw_cutoff: float = 0.50,
                                    vsh_cutoff: float = 0.40) -> Dict[str, Any]:
        """
        Run a standard petrophysical interpretation workflow.

        This method runs a complete interpretation sequence:
        1. Shale volume from GR
        2. Total porosity
        3. Effective porosity (corrected for shale)
        4. Water saturation
        5. Permeability (optional)
        6. Net pay (optional)

        Args:
            vshale_method: Method for Vshale calculation
            porosity_method: Method for porosity calculation
            sw_method: Method for Sw calculation
            compute_perm: Whether to compute permeability
            compute_pay: Whether to compute net pay
            phi_cutoff: Porosity cutoff for pay
            sw_cutoff: Sw cutoff for pay
            vsh_cutoff: Vsh cutoff for pay

        Returns:
            Dictionary with interpretation results and statistics
        """
        results = {
            'curves_computed': [],
            'warnings': [],
            'statistics': {},
        }

        # 1. Shale volume
        try:
            vsh = self.vshale(method=vshale_method, output='VSH')
            results['curves_computed'].append('VSH')
            results['statistics']['vsh_mean'] = float(np.nanmean(vsh.values))
        except MissingCurveError as e:
            results['warnings'].append(f"Could not compute Vshale: {e.mnemonic} not found")

        # 2. Total porosity
        try:
            phi_t = self.porosity(method=porosity_method, output='PHIT')
            results['curves_computed'].append('PHIT')
            results['statistics']['phit_mean'] = float(np.nanmean(phi_t.values))
        except MissingCurveError as e:
            results['warnings'].append(f"Could not compute porosity: {e.mnemonic} not found")
            return results  # Can't continue without porosity

        # 3. Effective porosity
        if 'VSH' in self.well.data:
            vsh_arr = self.well.data['VSH'].values
            phi_shale = self.params.clay.nphi_shale if self.params.clay else 0.0
            phi_e = porosity_effective(phi_t, vsh_arr, phi_shale=phi_shale)
            phi_e.mnemonic = 'PHIE'
            self._store_curve('PHIE', phi_e)
            results['curves_computed'].append('PHIE')
            results['statistics']['phie_mean'] = float(np.nanmean(phi_e.values))
        else:
            # Use total porosity as effective
            self._store_curve('PHIE', phi_t)
            results['curves_computed'].append('PHIE')

        # 4. Water saturation
        try:
            sw = self.sw(method=sw_method, phi='PHIE', output='SW')
            results['curves_computed'].append('SW')
            results['statistics']['sw_mean'] = float(np.nanmean(sw.values))
        except MissingCurveError as e:
            results['warnings'].append(f"Could not compute Sw: {e.mnemonic} not found")

        # 5. Permeability
        if compute_perm and 'SW' in self.well.data:
            try:
                k = self.permeability(method='timur', phi='PHIE', swirr='SW', output='PERM')
                results['curves_computed'].append('PERM')
                results['statistics']['perm_mean'] = float(np.nanmean(k.values))
            except Exception as e:
                results['warnings'].append(f"Could not compute permeability: {str(e)}")

        # 6. Net pay
        if compute_pay and 'SW' in self.well.data:
            try:
                pay = self.net_pay(phi='PHIE', sw='SW', vsh='VSH' if 'VSH' in self.well.data else None,
                                   phi_cutoff=phi_cutoff, sw_cutoff=sw_cutoff, vsh_cutoff=vsh_cutoff,
                                   output='PAY')
                results['curves_computed'].append('PAY')

                # Calculate N/G
                ntg = net_to_gross(pay.values)
                results['statistics']['net_to_gross'] = float(ntg)
            except Exception as e:
                results['warnings'].append(f"Could not compute net pay: {str(e)}")

        return results

    def qc_inputs(self) -> Dict[str, Dict[str, Any]]:
        """
        Quality check input curves.

        Returns statistics and warnings for each input curve.

        Returns:
            Dictionary with QC results for each curve
        """
        qc_results = {}

        curves_to_check = ['GR', 'RHOB', 'NPHI', 'RT', 'DT', 'SP', 'CALI']

        for std_name in curves_to_check:
            curve = self._get_curve(std_name, required=False)
            if curve is None:
                qc_results[std_name] = {'status': 'missing', 'actual_name': None}
                continue

            values = curve.values
            qc = {
                'status': 'ok',
                'actual_name': curve.mnemonic,
                'min': float(np.nanmin(values)),
                'max': float(np.nanmax(values)),
                'mean': float(np.nanmean(values)),
                'std': float(np.nanstd(values)),
                'nan_count': int(np.sum(np.isnan(values))),
                'nan_percent': float(np.sum(np.isnan(values)) / len(values) * 100),
            }

            # Check for issues
            warnings_list = []

            if qc['nan_percent'] > 10:
                warnings_list.append(f"High NaN percentage: {qc['nan_percent']:.1f}%")
                qc['status'] = 'warning'

            # Curve-specific checks
            if std_name == 'GR':
                if qc['min'] < 0:
                    warnings_list.append("Negative GR values")
                    qc['status'] = 'warning'
                if qc['max'] > 300:
                    warnings_list.append("Very high GR values (>300 API)")

            elif std_name == 'RHOB':
                if qc['min'] < 1.5:
                    warnings_list.append("Very low density (<1.5 g/cc)")
                    qc['status'] = 'warning'
                if qc['max'] > 3.0:
                    warnings_list.append("Very high density (>3.0 g/cc)")

            elif std_name == 'NPHI':
                if qc['min'] < -0.1:
                    warnings_list.append("Negative neutron porosity")
                if qc['max'] > 0.6:
                    warnings_list.append("Very high neutron porosity (>0.6)")

            elif std_name == 'RT':
                if qc['min'] <= 0:
                    warnings_list.append("Zero or negative resistivity")
                    qc['status'] = 'warning'

            qc['warnings'] = warnings_list
            qc_results[std_name] = qc

        return qc_results

    def summary(self) -> Dict[str, Any]:
        """
        Generate interpretation summary.

        Returns:
            Dictionary with well info and computed curve statistics
        """
        summary = {
            'well_name': getattr(self.well, 'name', 'Unknown'),
            'uwi': getattr(self.well, 'uwi', 'Unknown'),
            'parameters': self.params.name if self.params else 'Default',
            'curves_computed': list(self.results.keys()),
            'statistics': {},
        }

        # Add statistics for computed curves
        for curve_name in self.results.keys():
            if curve_name in self.well.data:
                values = self.well.data[curve_name].values
                summary['statistics'][curve_name] = {
                    'min': float(np.nanmin(values)),
                    'max': float(np.nanmax(values)),
                    'mean': float(np.nanmean(values)),
                    'std': float(np.nanstd(values)),
                }

        return summary

    # =========================================================================
    # Plotting Methods
    # =========================================================================

    def plot(self,
             tracks: Optional[List[str]] = None,
             depth_range: Optional[tuple] = None,
             figsize: Optional[tuple] = None,
             show_inputs: bool = True,
             show_pay: bool = True,
             title: Optional[str] = None) -> 'Figure':
        """
        Create a standard petrophysical interpretation plot.

        Generates a multi-track log plot showing input curves and
        computed petrophysical properties.

        Args:
            tracks: List of track names to display. If None, auto-selects
                    based on available curves. Options include:
                    'GR', 'RHOB_NPHI', 'RT', 'VSH', 'PHI', 'SW', 'PERM', 'PAY'
            depth_range: Tuple of (top, bottom) depth to display.
                        If None, uses full curve extent.
            figsize: Figure size as (width, height). If None, auto-calculated.
            show_inputs: Whether to show input curves (GR, RHOB, etc.)
            show_pay: Whether to show pay flag track
            title: Plot title. If None, uses well name.

        Returns:
            matplotlib.figure.Figure object

        Example:
            >>> interp.run_standard_interpretation()
            >>> fig = interp.plot()
            >>> fig.savefig('interpretation.png', dpi=150)
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        # Determine which tracks to show
        if tracks is None:
            tracks = self._auto_select_tracks(show_inputs, show_pay)

        n_tracks = len(tracks)
        if n_tracks == 0:
            raise ValueError("No tracks to plot. Run interpretation first.")

        # Set up figure size
        if figsize is None:
            figsize = (2.5 * n_tracks, 12)

        fig, axes = plt.subplots(1, n_tracks, figsize=figsize, sharey=True)
        if n_tracks == 1:
            axes = [axes]

        # Get depth basis
        depth = self._get_plot_depth()
        if depth is None:
            raise ValueError("Could not determine depth basis for plotting")

        # Apply depth range
        if depth_range is not None:
            top, bottom = depth_range
        else:
            top, bottom = np.nanmin(depth), np.nanmax(depth)

        # Plot each track
        for i, track in enumerate(tracks):
            ax = axes[i]
            self._plot_track(ax, track, depth)
            ax.set_ylim(bottom, top)  # Inverted
            ax.grid(True, alpha=0.3, axis='y')

        # Set title
        title = title or getattr(self.well, 'name', 'Petrophysical Interpretation')
        fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def _auto_select_tracks(self, show_inputs: bool, show_pay: bool) -> List[str]:
        """Auto-select tracks based on available curves."""
        tracks = []

        if show_inputs:
            # Input curves
            if self._get_curve('GR', required=False) is not None:
                tracks.append('GR')
            if (self._get_curve('RHOB', required=False) is not None or
                self._get_curve('NPHI', required=False) is not None):
                tracks.append('RHOB_NPHI')
            if self._get_curve('RT', required=False) is not None:
                tracks.append('RT')

        # Computed curves
        if 'VSH' in self.well.data:
            tracks.append('VSH')
        if 'PHIE' in self.well.data or 'PHIT' in self.well.data:
            tracks.append('PHI')
        if 'SW' in self.well.data:
            tracks.append('SW')
        if 'PERM' in self.well.data:
            tracks.append('PERM')
        if show_pay and 'PAY' in self.well.data:
            tracks.append('PAY')

        return tracks

    def _get_plot_depth(self) -> Optional[np.ndarray]:
        """Get depth array for plotting."""
        # Try computed curves first
        for name in ['VSH', 'PHIE', 'PHIT', 'SW', 'PAY']:
            if name in self.well.data:
                return np.asarray(self.well.data[name].df.index)

        # Fall back to input curves
        for std_name in ['GR', 'RHOB', 'RT']:
            curve = self._get_curve(std_name, required=False)
            if curve is not None:
                return np.asarray(curve.df.index)

        return None

    def _plot_track(self, ax, track: str, depth: np.ndarray) -> None:
        """Plot a single track."""
        if track == 'GR':
            self._plot_gr_track(ax, depth)
        elif track == 'RHOB_NPHI':
            self._plot_density_neutron_track(ax, depth)
        elif track == 'RT':
            self._plot_resistivity_track(ax, depth)
        elif track == 'VSH':
            self._plot_vshale_track(ax, depth)
        elif track == 'PHI':
            self._plot_porosity_track(ax, depth)
        elif track == 'SW':
            self._plot_saturation_track(ax, depth)
        elif track == 'PERM':
            self._plot_permeability_track(ax, depth)
        elif track == 'PAY':
            self._plot_pay_track(ax, depth)
        else:
            # Try to plot as a generic curve
            if track in self.well.data:
                values = self.well.data[track].values
                ax.plot(values, depth, 'b-', lw=0.5)
                ax.set_xlabel(track)

    def _plot_gr_track(self, ax, depth: np.ndarray) -> None:
        """Plot gamma ray track."""
        gr = self._get_curve('GR', required=False)
        if gr is None:
            return

        values = gr.values
        ax.plot(values, depth, 'g-', lw=0.5)
        ax.fill_betweenx(depth, 0, values, color='green', alpha=0.3)
        ax.set_xlim(0, 150)
        ax.set_xlabel('GR (API)')
        ax.set_title('Gamma Ray')

        # Add clean/shale lines if params available
        if self.params and self.params.clay:
            ax.axvline(self.params.clay.gr_clean, color='yellow', ls='--', lw=1, alpha=0.7)
            ax.axvline(self.params.clay.gr_shale, color='gray', ls='--', lw=1, alpha=0.7)

    def _plot_density_neutron_track(self, ax, depth: np.ndarray) -> None:
        """Plot density-neutron track."""
        rhob = self._get_curve('RHOB', required=False)
        nphi = self._get_curve('NPHI', required=False)

        if rhob is not None:
            ax.plot(rhob.values, depth, 'r-', lw=0.5, label='RHOB')
        if nphi is not None:
            # Plot on secondary x-axis (neutron scale is reversed)
            ax2 = ax.twiny()
            ax2.plot(nphi.values, depth, 'b-', lw=0.5, label='NPHI')
            ax2.set_xlim(0.45, -0.15)
            ax2.set_xlabel('NPHI (v/v)', color='blue')
            ax2.tick_params(axis='x', colors='blue')

        ax.set_xlim(1.95, 2.95)
        ax.set_xlabel('RHOB (g/cc)', color='red')
        ax.tick_params(axis='x', colors='red')
        ax.set_title('Density/Neutron')

    def _plot_resistivity_track(self, ax, depth: np.ndarray) -> None:
        """Plot resistivity track."""
        rt = self._get_curve('RT', required=False)
        if rt is None:
            return

        values = np.clip(rt.values, 0.1, 10000)
        ax.plot(values, depth, 'k-', lw=0.5)
        ax.set_xscale('log')
        ax.set_xlim(0.1, 1000)
        ax.set_xlabel('RT (ohm.m)')
        ax.set_title('Resistivity')

    def _plot_vshale_track(self, ax, depth: np.ndarray) -> None:
        """Plot shale volume track."""
        if 'VSH' not in self.well.data:
            return

        values = self.well.data['VSH'].values
        ax.fill_betweenx(depth, 0, values, color='gray', alpha=0.6)
        ax.plot(values, depth, 'k-', lw=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Vsh (v/v)')
        ax.set_title('Shale Volume')

        # Add cutoff line
        ax.axvline(0.4, color='red', ls='--', lw=1, alpha=0.7)

    def _plot_porosity_track(self, ax, depth: np.ndarray) -> None:
        """Plot porosity track."""
        # Plot effective porosity if available, otherwise total
        if 'PHIE' in self.well.data:
            phie = self.well.data['PHIE'].values
            ax.fill_betweenx(depth, 0, phie, color='blue', alpha=0.3)
            ax.plot(phie, depth, 'b-', lw=0.5, label='PHIE')

        if 'PHIT' in self.well.data:
            phit = self.well.data['PHIT'].values
            ax.plot(phit, depth, 'b--', lw=0.5, alpha=0.7, label='PHIT')

        ax.set_xlim(0, 0.4)
        ax.set_xlabel('Porosity (v/v)')
        ax.set_title('Porosity')
        ax.legend(loc='upper right', fontsize=8)

        # Add cutoff line
        ax.axvline(0.08, color='red', ls='--', lw=1, alpha=0.7)

    def _plot_saturation_track(self, ax, depth: np.ndarray) -> None:
        """Plot water saturation track with hydrocarbon fill."""
        if 'SW' not in self.well.data:
            return

        sw = self.well.data['SW'].values

        # Fill hydrocarbon (1 - Sw) in green, water in blue
        ax.fill_betweenx(depth, sw, 1, color='green', alpha=0.4, label='HC')
        ax.fill_betweenx(depth, 0, sw, color='lightblue', alpha=0.4, label='Water')
        ax.plot(sw, depth, 'b-', lw=0.5)

        ax.set_xlim(0, 1)
        ax.set_xlabel('Sw (v/v)')
        ax.set_title('Saturation')
        ax.legend(loc='upper right', fontsize=8)

        # Add cutoff line
        ax.axvline(0.5, color='red', ls='--', lw=1, alpha=0.7)

    def _plot_permeability_track(self, ax, depth: np.ndarray) -> None:
        """Plot permeability track."""
        if 'PERM' not in self.well.data:
            return

        values = np.clip(self.well.data['PERM'].values, 0.001, 10000)
        ax.plot(values, depth, 'brown', lw=0.5)
        ax.fill_betweenx(depth, 0.001, values, color='brown', alpha=0.3)
        ax.set_xscale('log')
        ax.set_xlim(0.01, 10000)
        ax.set_xlabel('Perm (mD)')
        ax.set_title('Permeability')

    def _plot_pay_track(self, ax, depth: np.ndarray) -> None:
        """Plot pay flag track."""
        if 'PAY' not in self.well.data:
            return

        pay = self.well.data['PAY'].values
        ax.fill_betweenx(depth, 0, pay, color='gold', alpha=0.8)
        ax.set_xlim(0, 1.5)
        ax.set_xlabel('Pay')
        ax.set_title('Net Pay')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Non-Pay', 'Pay'])

    def detect_contacts(self,
                        sw_threshold: float = 0.5,
                        method: str = 'threshold') -> Dict[str, Optional[float]]:
        """
        Detect fluid contacts from saturation data.

        Args:
            sw_threshold: Sw threshold for OWC detection
            method: Detection method ('threshold' or 'gradient')

        Returns:
            Dictionary with detected contact depths
        """
        contacts = {'owc': None, 'goc': None, 'fwl': None}

        sw_curve = self._get_curve('SW', required=False)
        if sw_curve is None:
            return contacts

        depth = sw_curve.index

        owc = detect_owc(sw_curve.values, depth, sw_threshold=sw_threshold, method=method)
        contacts['owc'] = float(owc) if owc is not None else None

        return contacts


class MissingCurveError(ValueError):
    """
    Raised when a required curve is not found in the well.

    Attributes:
        mnemonic: The standard curve name that was requested
        tried: List of mnemonics that were tried
        available: List of curves available in the well
    """

    def __init__(self, mnemonic: str, tried: List[str], available: List[str]):
        self.mnemonic = mnemonic
        self.tried = tried
        self.available = available

        msg = (
            f"Curve '{mnemonic}' not found.\n"
            f"  Tried: {tried}\n"
            f"  Available: {available}\n"
            f"  Hint: Use interp.add_alias('{mnemonic}', ['YOUR_CURVE']) "
            f"to add a mapping."
        )
        super().__init__(msg)
