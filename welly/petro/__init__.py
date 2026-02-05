"""
Petrophysics module for welly.

This module provides enterprise-grade petrophysical calculations including:
- Shale volume estimation (multiple methods)
- Porosity calculations (density, neutron, sonic, combination)
- Water saturation models (Archie, Simandoux, Indonesia, Waxman-Smits, Dual-Water)
- Permeability estimation (multiple correlations)
- Net pay determination
- Fluid contact detection

All functions are designed to work seamlessly with welly Curve objects
and return Curve objects with appropriate metadata.

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
    
    Waxman, M.H. & Smits, L.J.M. (1968). Electrical conductivities in 
        oil-bearing shaly sands. SPE Journal, 8(2), 107-122.
    
    Clavier, C., Coates, G. & Dumanoir, J. (1984). Theoretical and 
        experimental bases for the dual-water model for interpretation 
        of shaly sands. SPE Journal, 24(2), 153-168.
"""

from .shale_volume import (
    vshale_linear,
    vshale_larionov,
    vshale_larionov_older,
    vshale_steiber,
    vshale_clavier,
    vshale_from_sp,
    vshale_from_neutron_density,
)

from .porosity import (
    porosity_density,
    porosity_neutron,
    porosity_sonic_wyllie,
    porosity_sonic_raymer,
    porosity_neutron_density,
    porosity_effective,
    porosity_total_from_density,
)

from .saturation import (
    archie,
    simandoux,
    indonesia,
    fertl,
    waxman_smits,
    dual_water,
    buckles_number,
    bulk_volume_water,
)

from .permeability import (
    perm_timur,
    perm_coates,
    perm_tixier,
    perm_morris_biggs,
    perm_wyllie_rose,
    perm_kozeny_carman,
)

from .net_pay import (
    net_pay_flag,
    net_to_gross,
    pay_summary,
    reservoir_summary,
)

from .fluid_contacts import (
    detect_owc,
    detect_goc,
    detect_fwl,
    gradient_intersection,
    analyze_transition_zone,
)

from .parameters import (
    PetrophysicalParameters,
    ClayParameters,
    FluidParameters,
    MatrixParameters,
)

from .interpreter import PetroInterpreter, DEFAULT_ALIAS

__all__ = [
    # Shale volume
    'vshale_linear',
    'vshale_larionov',
    'vshale_larionov_older',
    'vshale_steiber',
    'vshale_clavier',
    'vshale_from_sp',
    'vshale_from_neutron_density',
    # Porosity
    'porosity_density',
    'porosity_neutron',
    'porosity_sonic_wyllie',
    'porosity_sonic_raymer',
    'porosity_neutron_density',
    'porosity_effective',
    'porosity_total_from_density',
    # Saturation
    'archie',
    'simandoux',
    'indonesia',
    'fertl',
    'waxman_smits',
    'dual_water',
    'buckles_number',
    'bulk_volume_water',
    # Permeability
    'perm_timur',
    'perm_coates',
    'perm_tixier',
    'perm_morris_biggs',
    'perm_wyllie_rose',
    'perm_kozeny_carman',
    # Net pay
    'net_pay_flag',
    'net_to_gross',
    'pay_summary',
    'reservoir_summary',
    # Fluid contacts
    'detect_owc',
    'detect_goc',
    'detect_fwl',
    'gradient_intersection',
    'analyze_transition_zone',
    # Parameters
    'PetrophysicalParameters',
    'ClayParameters',
    'FluidParameters',
    'MatrixParameters',
    # Interpreter
    'PetroInterpreter',
    'DEFAULT_ALIAS',
]
