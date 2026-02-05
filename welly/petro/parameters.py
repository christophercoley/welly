"""
Petrophysical parameter classes for consistent parameter management.

These dataclasses provide a structured way to manage petrophysical
parameters across calculations, ensuring consistency and enabling
parameter sets to be saved, loaded, and shared.

:copyright: 2024 Agile Scientific
:license: Apache 2.0
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json


@dataclass
class MatrixParameters:
    """
    Matrix (rock framework) parameters for petrophysical calculations.
    
    Attributes:
        rho_matrix: Matrix density (g/cc). Common values:
            - Sandstone (quartz): 2.65
            - Limestone (calcite): 2.71
            - Dite (dolomite): 2.87
            - Anhydrite: 2.98
        dt_matrix: Matrix sonic travel time (µs/ft). Common values:
            - Sandstone: 55.5-51.0
            - Limestone: 47.5
            - Dolomite: 43.5
        nphi_matrix: Matrix neutron porosity (v/v). Common values:
            - Sandstone: -0.02 to 0.0
            - Limestone: 0.0
            - Dolomite: 0.02
        lithology: Descriptive lithology name
    
    Example:
        >>> sandstone = MatrixParameters(
        ...     rho_matrix=2.65,
        ...     dt_matrix=55.5,
        ...     nphi_matrix=-0.02,
        ...     lithology='Clean Sandstone'
        ... )
    """
    rho_matrix: float = 2.65
    dt_matrix: float = 55.5
    nphi_matrix: float = 0.0
    lithology: str = 'sandstone'
    
    @classmethod
    def sandstone(cls) -> 'MatrixParameters':
        """Return typical sandstone parameters."""
        return cls(rho_matrix=2.65, dt_matrix=55.5, nphi_matrix=-0.02, 
                   lithology='sandstone')
    
    @classmethod
    def limestone(cls) -> 'MatrixParameters':
        """Return typical limestone parameters."""
        return cls(rho_matrix=2.71, dt_matrix=47.5, nphi_matrix=0.0,
                   lithology='limestone')
    
    @classmethod
    def dolomite(cls) -> 'MatrixParameters':
        """Return typical dolomite parameters."""
        return cls(rho_matrix=2.87, dt_matrix=43.5, nphi_matrix=0.02,
                   lithology='dolomite')
    
    @classmethod
    def shale(cls) -> 'MatrixParameters':
        """Return typical shale parameters."""
        return cls(rho_matrix=2.65, dt_matrix=100.0, nphi_matrix=0.30,
                   lithology='shale')


@dataclass
class FluidParameters:
    """
    Fluid parameters for petrophysical calculations.
    
    Attributes:
        rho_fluid: Fluid density (g/cc). Common values:
            - Fresh water: 1.0
            - Salt water: 1.0-1.2 (depends on salinity)
            - Oil: 0.7-0.9
            - Gas: 0.1-0.3
        dt_fluid: Fluid sonic travel time (µs/ft). Common values:
            - Water: 189
            - Oil: 230
            - Gas: 600+
        nphi_fluid: Fluid neutron response (v/v). Common values:
            - Water: 1.0
            - Oil: 1.0
            - Gas: 0.0-0.4
        rw: Formation water resistivity (ohm.m)
        rw_temp: Temperature at which Rw was measured (°F)
        salinity_ppm: Formation water salinity (ppm NaCl equivalent)
        fluid_type: Descriptive fluid type
    
    Example:
        >>> brine = FluidParameters(
        ...     rho_fluid=1.05,
        ...     rw=0.05,
        ...     rw_temp=75,
        ...     salinity_ppm=80000,
        ...     fluid_type='brine'
        ... )
    """
    rho_fluid: float = 1.0
    dt_fluid: float = 189.0
    nphi_fluid: float = 1.0
    rw: float = 0.1
    rw_temp: float = 75.0
    salinity_ppm: Optional[float] = None
    fluid_type: str = 'water'
    
    def rw_at_temperature(self, temp: float) -> float:
        """
        Calculate Rw at a different temperature using Arps equation.
        
        Args:
            temp: Target temperature (°F)
            
        Returns:
            Rw at the target temperature (ohm.m)
            
        Reference:
            Arps, J.J. (1953). The effect of temperature on the density 
            and electrical resistivity of sodium chloride solutions.
        """
        return self.rw * (self.rw_temp + 6.77) / (temp + 6.77)
    
    @classmethod
    def fresh_water(cls) -> 'FluidParameters':
        """Return typical fresh water parameters."""
        return cls(rho_fluid=1.0, dt_fluid=189.0, nphi_fluid=1.0,
                   rw=1.0, fluid_type='fresh_water')
    
    @classmethod
    def brine(cls, salinity_ppm: float = 50000, temp: float = 75) -> 'FluidParameters':
        """
        Return brine parameters calculated from salinity.
        
        Args:
            salinity_ppm: Salinity in ppm NaCl equivalent
            temp: Temperature in °F
        """
        # Approximate Rw from salinity using simplified relationship
        # More accurate methods exist but this is reasonable for estimation
        rw = (400000 / salinity_ppm) * ((temp + 6.77) / 81.77) ** -1
        rho = 1.0 + salinity_ppm * 0.7e-6  # Approximate density increase
        return cls(rho_fluid=rho, dt_fluid=189.0, nphi_fluid=1.0,
                   rw=rw, rw_temp=temp, salinity_ppm=salinity_ppm,
                   fluid_type='brine')


@dataclass
class ClayParameters:
    """
    Clay/shale parameters for shaly sand corrections.
    
    Attributes:
        rho_shale: Shale density (g/cc), typically 2.4-2.7
        dt_shale: Shale sonic travel time (µs/ft), typically 80-130
        nphi_shale: Shale neutron porosity (v/v), typically 0.25-0.45
        gr_shale: Shale gamma ray (API), typically 100-150
        gr_clean: Clean sand gamma ray (API), typically 10-30
        sp_shale: Shale SP (mV)
        sp_clean: Clean sand SP (mV)
        rt_shale: Shale resistivity (ohm.m), typically 1-10
        cec: Cation exchange capacity (meq/100g), for Waxman-Smits
        qv: Specific clay surface area (meq/ml), for Waxman-Smits
    
    Example:
        >>> clay = ClayParameters(
        ...     rho_shale=2.55,
        ...     nphi_shale=0.35,
        ...     gr_shale=120,
        ...     gr_clean=20,
        ...     rt_shale=5.0
        ... )
    """
    rho_shale: float = 2.55
    dt_shale: float = 100.0
    nphi_shale: float = 0.35
    gr_shale: float = 120.0
    gr_clean: float = 20.0
    sp_shale: Optional[float] = None
    sp_clean: Optional[float] = None
    rt_shale: float = 5.0
    cec: Optional[float] = None
    qv: Optional[float] = None
    
    def compute_qv(self, phi: float, rho_grain: float = 2.65) -> float:
        """
        Compute Qv from CEC if available.
        
        Qv = CEC * rho_grain * (1 - phi) / phi
        
        Args:
            phi: Porosity (v/v)
            rho_grain: Grain density (g/cc)
            
        Returns:
            Qv in meq/ml
        """
        if self.cec is None:
            raise ValueError("CEC not set, cannot compute Qv")
        if phi <= 0:
            return 0.0
        return self.cec * rho_grain * (1 - phi) / (100 * phi)


@dataclass 
class PetrophysicalParameters:
    """
    Complete parameter set for petrophysical interpretation.
    
    This class aggregates all parameters needed for a complete
    petrophysical evaluation, providing a single source of truth
    for an interpretation.
    
    Attributes:
        matrix: Matrix parameters
        fluid: Fluid parameters  
        clay: Clay/shale parameters
        a: Archie tortuosity factor (typically 0.62-1.0)
        m: Archie cementation exponent (typically 1.8-2.2)
        n: Archie saturation exponent (typically 1.8-2.2)
        bvw_irreducible: Irreducible bulk volume water (for Buckles)
        temperature: Formation temperature (°F)
        pressure: Formation pressure (psi)
        name: Name/identifier for this parameter set
        notes: Free-form notes about the interpretation
    
    Example:
        >>> params = PetrophysicalParameters(
        ...     matrix=MatrixParameters.sandstone(),
        ...     fluid=FluidParameters.brine(50000),
        ...     clay=ClayParameters(gr_shale=120, gr_clean=20),
        ...     a=0.81, m=2.0, n=2.0,
        ...     name='Montney Formation'
        ... )
        >>> params.to_json('montney_params.json')
    """
    matrix: MatrixParameters = field(default_factory=MatrixParameters.sandstone)
    fluid: FluidParameters = field(default_factory=FluidParameters)
    clay: ClayParameters = field(default_factory=ClayParameters)
    a: float = 1.0
    m: float = 2.0
    n: float = 2.0
    bvw_irreducible: float = 0.035
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    name: str = ''
    notes: str = ''
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            'matrix': asdict(self.matrix),
            'fluid': asdict(self.fluid),
            'clay': asdict(self.clay),
            'a': self.a,
            'm': self.m,
            'n': self.n,
            'bvw_irreducible': self.bvw_irreducible,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'name': self.name,
            'notes': self.notes,
        }
    
    def to_json(self, filepath: str) -> None:
        """Save parameters to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PetrophysicalParameters':
        """Create parameters from dictionary."""
        return cls(
            matrix=MatrixParameters(**d.get('matrix', {})),
            fluid=FluidParameters(**d.get('fluid', {})),
            clay=ClayParameters(**d.get('clay', {})),
            a=d.get('a', 1.0),
            m=d.get('m', 2.0),
            n=d.get('n', 2.0),
            bvw_irreducible=d.get('bvw_irreducible', 0.035),
            temperature=d.get('temperature'),
            pressure=d.get('pressure'),
            name=d.get('name', ''),
            notes=d.get('notes', ''),
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> 'PetrophysicalParameters':
        """Load parameters from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))
    
    @classmethod
    def default_sandstone(cls) -> 'PetrophysicalParameters':
        """Return default parameters for sandstone interpretation."""
        return cls(
            matrix=MatrixParameters.sandstone(),
            fluid=FluidParameters(),
            clay=ClayParameters(),
            a=0.81, m=2.0, n=2.0,
            name='Default Sandstone'
        )
    
    @classmethod
    def default_carbonate(cls) -> 'PetrophysicalParameters':
        """Return default parameters for carbonate interpretation."""
        return cls(
            matrix=MatrixParameters.limestone(),
            fluid=FluidParameters(),
            clay=ClayParameters(gr_shale=80, gr_clean=10),
            a=1.0, m=2.0, n=2.0,
            name='Default Carbonate'
        )
