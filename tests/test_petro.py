"""
Tests for the petrophysics module.

:copyright: 2024 Agile Scientific
:license: Apache 2.0
"""
import numpy as np
import pytest

from welly import Curve
from welly import petro
from welly.petro import (
    PetrophysicalParameters,
    MatrixParameters,
    FluidParameters,
    ClayParameters,
)


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def sample_curves():
    """Create sample curves for testing."""
    depth = np.arange(1000, 1100, 0.5)
    n = len(depth)
    
    # Gamma ray: clean sand with some shale
    gr = 30 + 20 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 5, n)
    gr = np.clip(gr, 10, 150)
    
    # Density: typical sandstone
    rhob = 2.35 + 0.1 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 0.02, n)
    rhob = np.clip(rhob, 2.0, 2.7)
    
    # Neutron: typical sandstone
    nphi = 0.20 + 0.05 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 0.02, n)
    nphi = np.clip(nphi, 0.05, 0.45)
    
    # Resistivity: oil-bearing zone
    rt = 50 + 30 * np.sin(np.linspace(0, 2*np.pi, n)) + np.random.normal(0, 5, n)
    rt = np.clip(rt, 5, 200)
    
    # Sonic
    dt = 80 + 10 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 2, n)
    dt = np.clip(dt, 55, 120)
    
    return {
        'depth': depth,
        'GR': Curve(data=gr, index=depth, mnemonic='GR', units='API'),
        'RHOB': Curve(data=rhob, index=depth, mnemonic='RHOB', units='g/cc'),
        'NPHI': Curve(data=nphi, index=depth, mnemonic='NPHI', units='v/v'),
        'RT': Curve(data=rt, index=depth, mnemonic='RT', units='ohm.m'),
        'DT': Curve(data=dt, index=depth, mnemonic='DT', units='us/ft'),
    }


@pytest.fixture
def simple_arrays():
    """Create simple arrays for basic testing."""
    return {
        'gr': np.array([20, 50, 80, 100, 120]),
        'rhob': np.array([2.65, 2.50, 2.35, 2.20, 2.10]),
        'nphi': np.array([0.0, 0.10, 0.20, 0.30, 0.40]),
        'rt': np.array([100, 50, 20, 10, 5]),
        'phi': np.array([0.05, 0.10, 0.15, 0.20, 0.25]),
        'sw': np.array([0.2, 0.3, 0.5, 0.7, 1.0]),
        'vsh': np.array([0.0, 0.1, 0.3, 0.5, 0.8]),
    }


# =============================================================================
# Parameter classes tests
# =============================================================================

class TestParameters:
    """Tests for parameter classes."""
    
    def test_matrix_parameters_defaults(self):
        """Test default matrix parameters."""
        mp = MatrixParameters()
        assert mp.rho_matrix == 2.65
        assert mp.dt_matrix == 55.5
        assert mp.lithology == 'sandstone'
    
    def test_matrix_parameters_presets(self):
        """Test preset matrix parameters."""
        ss = MatrixParameters.sandstone()
        assert ss.rho_matrix == 2.65
        
        ls = MatrixParameters.limestone()
        assert ls.rho_matrix == 2.71
        
        dol = MatrixParameters.dolomite()
        assert dol.rho_matrix == 2.87
    
    def test_fluid_parameters_temperature_correction(self):
        """Test Rw temperature correction."""
        fp = FluidParameters(rw=0.1, rw_temp=75)
        
        # Rw should increase at lower temperature
        rw_60 = fp.rw_at_temperature(60)
        assert rw_60 > fp.rw
        
        # Rw should decrease at higher temperature
        rw_150 = fp.rw_at_temperature(150)
        assert rw_150 < fp.rw
    
    def test_clay_parameters_qv_computation(self):
        """Test Qv computation from CEC."""
        cp = ClayParameters(cec=10)  # 10 meq/100g
        
        qv = cp.compute_qv(phi=0.20, rho_grain=2.65)
        assert qv > 0
        
        # Higher porosity should give lower Qv
        qv_high_phi = cp.compute_qv(phi=0.30)
        assert qv_high_phi < qv
    
    def test_petrophysical_parameters_json_roundtrip(self, tmp_path):
        """Test saving and loading parameters to JSON."""
        params = PetrophysicalParameters(
            matrix=MatrixParameters.sandstone(),
            fluid=FluidParameters(rw=0.05),
            clay=ClayParameters(gr_shale=120, gr_clean=20),
            a=0.81, m=2.0, n=2.0,
            name='Test Formation'
        )
        
        # Save to JSON
        json_path = tmp_path / 'params.json'
        params.to_json(str(json_path))
        
        # Load from JSON
        loaded = PetrophysicalParameters.from_json(str(json_path))
        
        assert loaded.name == 'Test Formation'
        assert loaded.a == 0.81
        assert loaded.matrix.rho_matrix == 2.65
        assert loaded.fluid.rw == 0.05


# =============================================================================
# Shale volume tests
# =============================================================================

class TestShaleVolume:
    """Tests for shale volume calculations."""
    
    def test_vshale_linear_basic(self, simple_arrays):
        """Test linear Vshale calculation."""
        vsh = petro.vshale_linear(
            simple_arrays['gr'],
            gr_clean=20,
            gr_shale=120,
            return_curve=False
        )
        
        # First value (GR=20) should be 0
        assert np.isclose(vsh[0], 0.0)
        
        # Last value (GR=120) should be 1
        assert np.isclose(vsh[4], 1.0)
        
        # Middle value (GR=80) should be 0.6
        assert np.isclose(vsh[2], 0.6)
    
    def test_vshale_linear_returns_curve(self, sample_curves):
        """Test that vshale_linear returns a Curve object."""
        vsh = petro.vshale_linear(
            sample_curves['GR'],
            gr_clean=20,
            gr_shale=120
        )
        
        assert isinstance(vsh, Curve)
        assert vsh.mnemonic == 'VSH_LIN'
        assert vsh.units == 'v/v'
        assert len(vsh) == len(sample_curves['GR'])
    
    def test_vshale_larionov_less_than_linear(self, simple_arrays):
        """Test that Larionov gives lower Vsh than linear."""
        vsh_lin = petro.vshale_linear(
            simple_arrays['gr'], 20, 120, return_curve=False
        )
        vsh_lar = petro.vshale_larionov(
            simple_arrays['gr'], 20, 120, return_curve=False
        )
        
        # Larionov should be <= linear for all values
        assert np.all(vsh_lar <= vsh_lin + 0.01)  # Small tolerance
    
    def test_vshale_steiber_between_linear_and_larionov(self, simple_arrays):
        """Test that Steiber is between linear and Larionov."""
        vsh_lin = petro.vshale_linear(
            simple_arrays['gr'], 20, 120, return_curve=False
        )
        vsh_stb = petro.vshale_steiber(
            simple_arrays['gr'], 20, 120, return_curve=False
        )
        vsh_lar = petro.vshale_larionov(
            simple_arrays['gr'], 20, 120, return_curve=False
        )
        
        # Steiber should be between linear and Larionov
        assert np.all(vsh_stb <= vsh_lin + 0.01)
        assert np.all(vsh_stb >= vsh_lar - 0.01)
    
    def test_vshale_clipped_to_0_1(self, simple_arrays):
        """Test that Vshale is clipped to [0, 1]."""
        # Create extreme GR values
        gr_extreme = np.array([0, 10, 150, 200])
        
        vsh = petro.vshale_linear(gr_extreme, 20, 120, return_curve=False)
        
        assert np.all(vsh >= 0)
        assert np.all(vsh <= 1)
    
    def test_vshale_from_neutron_density(self, simple_arrays):
        """Test neutron-density Vshale."""
        vsh = petro.vshale_from_neutron_density(
            simple_arrays['nphi'],
            simple_arrays['rhob'],
            nphi_clean=0.0,
            nphi_shale=0.35,
            rhob_clean=2.65,
            rhob_shale=2.45,
            return_curve=False
        )
        
        assert len(vsh) == len(simple_arrays['nphi'])
        assert np.all(vsh >= 0)
        assert np.all(vsh <= 1)


# =============================================================================
# Porosity tests
# =============================================================================

class TestPorosity:
    """Tests for porosity calculations."""
    
    def test_porosity_density_basic(self):
        """Test basic density porosity calculation."""
        rhob = np.array([2.65, 2.50, 2.35, 2.20])
        
        phi = petro.porosity_density(
            rhob, rho_matrix=2.65, rho_fluid=1.0, return_curve=False
        )
        
        # At matrix density, porosity should be 0
        assert np.isclose(phi[0], 0.0)
        
        # Check calculation: phi = (2.65 - 2.35) / (2.65 - 1.0) = 0.182
        assert np.isclose(phi[2], 0.182, atol=0.01)
    
    def test_porosity_density_returns_curve(self, sample_curves):
        """Test that porosity_density returns a Curve."""
        phi = petro.porosity_density(sample_curves['RHOB'])
        
        assert isinstance(phi, Curve)
        assert phi.mnemonic == 'PHID'
        assert phi.units == 'v/v'
    
    def test_porosity_sonic_wyllie(self):
        """Test Wyllie sonic porosity."""
        dt = np.array([55.5, 80, 100, 120])
        
        phi = petro.porosity_sonic_wyllie(
            dt, dt_matrix=55.5, dt_fluid=189, return_curve=False
        )
        
        # At matrix DT, porosity should be 0
        assert np.isclose(phi[0], 0.0)
        
        # Check calculation: phi = (80 - 55.5) / (189 - 55.5) = 0.184
        assert np.isclose(phi[1], 0.184, atol=0.01)
    
    def test_porosity_sonic_raymer(self):
        """Test Raymer sonic porosity."""
        dt = np.array([55.5, 80, 100])
        
        phi = petro.porosity_sonic_raymer(dt, dt_matrix=55.5, return_curve=False)
        
        # At matrix DT, porosity should be 0
        assert np.isclose(phi[0], 0.0)
        
        # Raymer should give different values than Wyllie
        phi_wyl = petro.porosity_sonic_wyllie(
            dt, dt_matrix=55.5, return_curve=False
        )
        assert not np.allclose(phi, phi_wyl)
    
    def test_porosity_neutron_density_combination(self, simple_arrays):
        """Test neutron-density combination porosity."""
        phi = petro.porosity_neutron_density(
            simple_arrays['nphi'],
            simple_arrays['rhob'],
            method='rms',
            return_curve=False
        )
        
        assert len(phi) == len(simple_arrays['nphi'])
        assert np.all(phi >= 0)
        assert np.all(phi <= 1)
    
    def test_porosity_effective(self, simple_arrays):
        """Test effective porosity calculation."""
        phi_t = np.array([0.20, 0.20, 0.20, 0.20])
        vsh = np.array([0.0, 0.1, 0.2, 0.3])
        
        phi_e = petro.porosity_effective(
            phi_t, vsh, phi_shale=0.0, return_curve=False
        )
        
        # With phi_shale=0, phi_e = phi_t
        assert np.allclose(phi_e, phi_t)
        
        # With phi_shale > 0, phi_e < phi_t for shaly zones
        phi_e_wet = petro.porosity_effective(
            phi_t, vsh, phi_shale=0.15, return_curve=False
        )
        assert np.all(phi_e_wet <= phi_t)


# =============================================================================
# Saturation tests
# =============================================================================

class TestSaturation:
    """Tests for water saturation calculations."""
    
    def test_archie_basic(self):
        """Test basic Archie calculation."""
        rt = np.array([100, 50, 20, 10])
        phi = np.array([0.20, 0.20, 0.20, 0.20])
        
        sw = petro.archie(rt, phi, rw=0.05, a=1.0, m=2.0, n=2.0, return_curve=False)
        
        # Higher Rt should give lower Sw
        assert sw[0] < sw[1] < sw[2] < sw[3]
        
        # All values should be between 0 and 1
        assert np.all(sw >= 0)
        assert np.all(sw <= 1)
    
    def test_archie_returns_curve(self, sample_curves):
        """Test that Archie returns a Curve."""
        phi = petro.porosity_density(sample_curves['RHOB'], return_curve=False)
        phi = np.clip(phi, 0.05, 0.4)  # Ensure valid porosity
        
        sw = petro.archie(
            sample_curves['RT'],
            phi,
            rw=0.05
        )
        
        assert isinstance(sw, Curve)
        assert sw.mnemonic == 'SW_ARCH'
    
    def test_simandoux_less_than_archie_in_shaly_zones(self):
        """Test that Simandoux gives lower Sw than Archie in shaly zones."""
        rt = np.array([20, 20, 20, 20])
        phi = np.array([0.15, 0.15, 0.15, 0.15])
        vsh = np.array([0.0, 0.1, 0.2, 0.3])
        
        sw_arch = petro.archie(rt, phi, rw=0.05, return_curve=False)
        sw_sim = petro.simandoux(rt, phi, vsh, rw=0.05, rsh=5.0, return_curve=False)
        
        # In clean zone (vsh=0), should be similar
        assert np.isclose(sw_arch[0], sw_sim[0], atol=0.05)
        
        # In shaly zones, Simandoux should give lower Sw
        assert sw_sim[2] < sw_arch[2]
        assert sw_sim[3] < sw_arch[3]
    
    def test_indonesia_equation(self):
        """Test Indonesia equation."""
        rt = np.array([20, 20, 20])
        phi = np.array([0.15, 0.15, 0.15])
        vsh = np.array([0.0, 0.2, 0.4])
        
        sw = petro.indonesia(rt, phi, vsh, rw=0.05, rsh=5.0, return_curve=False)
        
        assert len(sw) == 3
        assert np.all(sw >= 0)
        assert np.all(sw <= 1)
    
    def test_buckles_number(self, simple_arrays):
        """Test Buckles number calculation."""
        bvw = petro.buckles_number(
            simple_arrays['phi'],
            simple_arrays['sw'],
            return_curve=False
        )
        
        # BVW = phi * sw
        expected = simple_arrays['phi'] * simple_arrays['sw']
        assert np.allclose(bvw, expected)


# =============================================================================
# Permeability tests
# =============================================================================

class TestPermeability:
    """Tests for permeability calculations."""
    
    def test_perm_timur(self):
        """Test Timur permeability."""
        phi = np.array([0.10, 0.15, 0.20, 0.25])
        swirr = np.array([0.30, 0.25, 0.20, 0.15])
        
        k = petro.perm_timur(phi, swirr, return_curve=False)
        
        # Higher porosity and lower Swirr should give higher perm
        assert k[3] > k[2] > k[1] > k[0]
        
        # All values should be positive
        assert np.all(k > 0)
    
    def test_perm_coates(self):
        """Test Coates permeability."""
        phi = np.array([0.15, 0.20, 0.25])
        swirr = np.array([0.25, 0.20, 0.15])
        
        k = petro.perm_coates(phi, swirr, return_curve=False)
        
        assert len(k) == 3
        assert np.all(k > 0)
    
    def test_permeability_methods_give_different_results(self):
        """Test that different perm methods give different results."""
        phi = np.array([0.15, 0.20])
        swirr = np.array([0.25, 0.20])
        
        k_timur = petro.perm_timur(phi, swirr, return_curve=False)
        k_coates = petro.perm_coates(phi, swirr, return_curve=False)
        k_tixier = petro.perm_tixier(phi, swirr, return_curve=False)
        
        # Methods should give different results
        assert not np.allclose(k_timur, k_coates)
        assert not np.allclose(k_timur, k_tixier)


# =============================================================================
# Net pay tests
# =============================================================================

class TestNetPay:
    """Tests for net pay calculations."""
    
    def test_net_pay_flag_basic(self):
        """Test basic net pay flagging."""
        phi = np.array([0.05, 0.10, 0.15, 0.20])
        sw = np.array([0.80, 0.60, 0.40, 0.20])
        vsh = np.array([0.50, 0.30, 0.20, 0.10])
        
        pay = petro.net_pay_flag(
            phi, sw, vsh,
            phi_cutoff=0.08,
            sw_cutoff=0.50,
            vsh_cutoff=0.35,
            return_curve=False
        )
        
        # First sample: phi too low
        assert pay[0] == 0
        
        # Second sample: sw too high
        assert pay[1] == 0
        
        # Third and fourth samples: should be pay
        assert pay[2] == 1
        assert pay[3] == 1
    
    def test_net_to_gross(self):
        """Test net-to-gross calculation."""
        pay_flag = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        
        ntg = petro.net_to_gross(pay_flag)
        
        # 4 pay samples out of 8 = 0.5
        assert np.isclose(ntg, 0.5)
    
    def test_pay_summary(self):
        """Test pay summary statistics."""
        depth = np.arange(1000, 1010, 1.0)
        phi = np.array([0.15, 0.18, 0.20, 0.22, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08])
        sw = np.array([0.30, 0.25, 0.20, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80])
        pay = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        
        summary = petro.pay_summary(phi, sw, pay, index=depth)
        
        assert 'gross_thickness' in summary
        assert 'net_thickness' in summary
        assert 'net_to_gross' in summary
        assert 'avg_phi_pay' in summary
        assert 'avg_sw_pay' in summary
        
        # Check values
        assert summary['gross_thickness'] == 10.0
        assert summary['net_thickness'] == 6.0
        assert np.isclose(summary['net_to_gross'], 0.6)


# =============================================================================
# Fluid contacts tests
# =============================================================================

class TestFluidContacts:
    """Tests for fluid contact detection."""
    
    def test_detect_owc_threshold(self):
        """Test OWC detection using threshold method."""
        depth = np.arange(1000, 1100, 1.0)
        # Sw increases with depth (transition zone)
        sw = 0.2 + 0.008 * (depth - 1000)
        sw = np.clip(sw, 0, 1)
        
        owc = petro.detect_owc(sw, depth, sw_threshold=0.5, method='threshold')
        
        # OWC should be around depth where Sw = 0.5
        # Sw = 0.5 when 0.2 + 0.008*(d-1000) = 0.5 -> d = 1037.5
        assert owc is not None
        assert 1035 < owc < 1040
    
    def test_gradient_intersection(self):
        """Test pressure gradient intersection."""
        # Create synthetic pressure data
        depth = np.array([8000, 8050, 8100, 8150, 8200, 8250, 8300, 8350, 8400])
        
        # Oil gradient: 0.35 psi/ft, Water gradient: 0.45 psi/ft
        # Contact at 8200 ft
        pressure = np.where(
            depth < 8200,
            3500 + 0.35 * (depth - 8000),  # Oil
            3500 + 0.35 * 200 + 0.45 * (depth - 8200)  # Water
        )
        
        owc = petro.gradient_intersection(
            pressure, depth,
            fluid1_gradient=0.35,
            fluid2_gradient=0.45,
            fluid1_depth_range=(8000, 8150),
            fluid2_depth_range=(8250, 8400)
        )
        
        assert owc is not None
        assert 8150 < owc < 8250


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_petrophysical_workflow(self, sample_curves):
        """Test a complete petrophysical interpretation workflow."""
        # 1. Calculate shale volume
        vsh = petro.vshale_larionov(
            sample_curves['GR'],
            gr_clean=20,
            gr_shale=120
        )
        
        # 2. Calculate porosity
        phi_d = petro.porosity_density(
            sample_curves['RHOB'],
            rho_matrix=2.65,
            rho_fluid=1.0
        )
        
        # 3. Calculate effective porosity
        phi_e = petro.porosity_effective(phi_d, vsh, phi_shale=0.0)
        
        # 4. Calculate water saturation
        # Ensure valid porosity values
        phi_e_arr = np.clip(phi_e.values, 0.05, 0.4)
        
        sw = petro.archie(
            sample_curves['RT'],
            phi_e_arr,
            rw=0.05,
            a=0.81, m=2.0, n=2.0
        )
        
        # 5. Calculate permeability
        sw_arr = np.clip(sw.values, 0.1, 1.0)
        k = petro.perm_timur(phi_e_arr, sw_arr)
        
        # 6. Determine net pay
        pay = petro.net_pay_flag(
            phi_e_arr,
            sw_arr,
            vsh.values,
            phi_cutoff=0.08,
            sw_cutoff=0.50,
            vsh_cutoff=0.40
        )
        
        # Verify all outputs are valid
        assert isinstance(vsh, Curve)
        assert isinstance(phi_d, Curve)
        assert isinstance(phi_e, Curve)
        assert isinstance(sw, Curve)
        assert isinstance(k, Curve)
        assert isinstance(pay, Curve)
        
        # Verify reasonable values
        assert np.all(vsh.values >= 0)
        assert np.all(vsh.values <= 1)
        assert np.nanmax(phi_e.values) < 0.5
        assert np.all(sw.values >= 0)
        assert np.all(sw.values <= 1)
    
    def test_workflow_with_parameters_object(self, sample_curves):
        """Test workflow using PetrophysicalParameters object."""
        # Create parameter set
        params = PetrophysicalParameters(
            matrix=MatrixParameters.sandstone(),
            fluid=FluidParameters(rw=0.05),
            clay=ClayParameters(gr_shale=120, gr_clean=20),
            a=0.81, m=2.0, n=2.0,
            name='Test Formation'
        )
        
        # Use parameters in calculations
        vsh = petro.vshale_larionov(
            sample_curves['GR'],
            gr_clean=params.clay.gr_clean,
            gr_shale=params.clay.gr_shale
        )
        
        phi = petro.porosity_density(
            sample_curves['RHOB'],
            rho_matrix=params.matrix.rho_matrix,
            rho_fluid=params.fluid.rho_fluid
        )
        
        phi_arr = np.clip(phi.values, 0.05, 0.4)
        
        sw = petro.archie(
            sample_curves['RT'],
            phi_arr,
            rw=params.fluid.rw,
            a=params.a,
            m=params.m,
            n=params.n
        )
        
        assert isinstance(sw, Curve)
        assert np.all(sw.values >= 0)
        assert np.all(sw.values <= 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# =============================================================================
# PetroInterpreter tests
# =============================================================================

class TestPetroInterpreter:
    """Tests for PetroInterpreter class."""

    @pytest.fixture
    def mock_well(self, sample_curves):
        """Create a mock Well object for testing."""
        class MockWell:
            def __init__(self):
                self.data = {}
                self.name = 'Test Well'
                self.uwi = 'TEST-001'

            def get_curve(self, mnemonic, alias=None):
                return self.data.get(mnemonic)

        well = MockWell()
        well.data = {
            'GR': sample_curves['GR'],
            'RHOB': sample_curves['RHOB'],
            'NPHI': sample_curves['NPHI'],
            'RT': sample_curves['RT'],
            'DT': sample_curves['DT'],
        }
        return well

    def test_interpreter_creation(self, mock_well):
        """Test basic interpreter creation."""
        from welly.petro import PetroInterpreter

        interp = PetroInterpreter(mock_well)
        assert interp.well is mock_well
        assert interp.params is not None
        assert interp.alias is not None

    def test_interpreter_with_custom_params(self, mock_well):
        """Test interpreter with custom parameters."""
        from welly.petro import PetroInterpreter

        params = PetrophysicalParameters(
            matrix=MatrixParameters.limestone(),
            a=1.0, m=2.0, n=2.0,
            name='Custom Params'
        )
        interp = PetroInterpreter(mock_well, params=params)
        assert interp.params.name == 'Custom Params'
        assert interp.params.matrix.rho_matrix == 2.71

    def test_vshale_computation(self, mock_well):
        """Test Vshale computation through interpreter."""
        from welly.petro import PetroInterpreter

        params = PetrophysicalParameters.default_sandstone()
        params.clay.gr_clean = 20
        params.clay.gr_shale = 120

        interp = PetroInterpreter(mock_well, params=params)
        vsh = interp.vshale(method='larionov')

        assert 'VSH' in mock_well.data
        assert np.all(vsh.values >= 0)
        assert np.all(vsh.values <= 1)

    def test_porosity_computation(self, mock_well):
        """Test porosity computation through interpreter."""
        from welly.petro import PetroInterpreter

        interp = PetroInterpreter(mock_well)
        phi = interp.porosity(method='density')

        assert 'PHID' in mock_well.data
        assert phi.mnemonic == 'PHID'

    def test_sw_computation(self, mock_well):
        """Test Sw computation through interpreter."""
        from welly.petro import PetroInterpreter

        params = PetrophysicalParameters.default_sandstone()
        params.fluid.rw = 0.05

        interp = PetroInterpreter(mock_well, params=params)

        # First compute porosity
        interp.porosity(method='density', output='PHIE')

        # Then compute Sw
        sw = interp.sw(method='archie', phi='PHIE')

        assert 'SW' in mock_well.data
        assert np.all(sw.values >= 0)
        assert np.all(sw.values <= 1)

    def test_standard_workflow(self, mock_well):
        """Test standard interpretation workflow."""
        from welly.petro import PetroInterpreter

        params = PetrophysicalParameters.default_sandstone()
        params.clay.gr_clean = 20
        params.clay.gr_shale = 120
        params.fluid.rw = 0.05

        interp = PetroInterpreter(mock_well, params=params)
        results = interp.run_standard_interpretation(
            porosity_method='density',
            sw_method='archie'
        )

        assert 'VSH' in results['curves_computed']
        assert 'PHIT' in results['curves_computed']
        assert 'SW' in results['curves_computed']

    def test_alias_lookup(self, mock_well):
        """Test curve alias lookup."""
        from welly.petro import PetroInterpreter

        # Add curve with non-standard name
        mock_well.data['GRGC'] = mock_well.data['GR']
        del mock_well.data['GR']

        interp = PetroInterpreter(mock_well)

        # Should find GRGC via alias
        curve = interp._get_curve('GR', required=False)
        assert curve is not None

    def test_qc_inputs(self, mock_well):
        """Test input QC functionality."""
        from welly.petro import PetroInterpreter

        interp = PetroInterpreter(mock_well)
        qc = interp.qc_inputs()

        assert 'GR' in qc
        assert 'RHOB' in qc
        assert qc['GR']['status'] in ['ok', 'warning']

    def test_summary(self, mock_well):
        """Test summary generation."""
        from welly.petro import PetroInterpreter

        params = PetrophysicalParameters.default_sandstone()
        params.clay.gr_clean = 20
        params.clay.gr_shale = 120
        params.fluid.rw = 0.05

        interp = PetroInterpreter(mock_well, params=params)
        interp.run_standard_interpretation(porosity_method='density')

        summary = interp.summary()

        assert 'well_name' in summary
        assert 'curves_computed' in summary
        assert len(summary['curves_computed']) > 0


    def test_add_alias(self, mock_well):
        """Test adding aliases programmatically."""
        from welly.petro import PetroInterpreter

        interp = PetroInterpreter(mock_well)

        # Add a new alias
        interp.add_alias('GR', ['GAMMA_RAY', 'GR_CORR'])

        # Check it was added
        assert 'GAMMA_RAY' in interp.alias['GR']
        assert 'GR_CORR' in interp.alias['GR']

        # Original aliases should still be there
        assert 'GR' in interp.alias['GR']
        assert 'GRGC' in interp.alias['GR']

    def test_remove_alias(self, mock_well):
        """Test removing aliases."""
        from welly.petro import PetroInterpreter

        interp = PetroInterpreter(mock_well)

        # Remove specific mnemonics
        interp.remove_alias('GR', ['GRGC', 'GRD'])
        assert 'GRGC' not in interp.alias['GR']
        assert 'GRD' not in interp.alias['GR']
        assert 'GR' in interp.alias['GR']  # Original still there

        # Remove entire group
        interp.remove_alias('SP')
        assert 'SP' not in interp.alias

    def test_update_aliases_merge(self, mock_well):
        """Test updating aliases with merge behavior."""
        from welly.petro import PetroInterpreter

        interp = PetroInterpreter(mock_well)

        # Update with merge (default)
        interp.update_aliases({'GR': ['NEW_GR'], 'CUSTOM': ['CUSTOM1', 'CUSTOM2']})

        # GR should have both old and new
        assert 'GR' in interp.alias['GR']
        assert 'NEW_GR' in interp.alias['GR']

        # New group should be added
        assert 'CUSTOM' in interp.alias
        assert 'CUSTOM1' in interp.alias['CUSTOM']

    def test_update_aliases_replace(self, mock_well):
        """Test updating aliases with replace behavior."""
        from welly.petro import PetroInterpreter

        interp = PetroInterpreter(mock_well)

        # Update with replace
        interp.update_aliases({'GR': ['ONLY_GR']}, replace=True)

        # Should only have the new aliases
        assert interp.alias == {'GR': ['ONLY_GR']}

    def test_save_load_aliases(self, mock_well, tmp_path):
        """Test saving and loading aliases to/from JSON."""
        from welly.petro import PetroInterpreter

        interp = PetroInterpreter(mock_well)

        # Add custom alias
        interp.add_alias('GR', ['MY_GAMMA'])

        # Save to file
        alias_file = tmp_path / 'aliases.json'
        interp.save_aliases(alias_file, description='Test aliases')

        # Create new interpreter and load aliases
        interp2 = PetroInterpreter(mock_well)
        interp2.load_aliases(alias_file)

        # Check the custom alias was loaded
        assert 'MY_GAMMA' in interp2.alias['GR']

    def test_get_alias_matches(self, mock_well):
        """Test getting alias matches for available curves."""
        from welly.petro import PetroInterpreter

        interp = PetroInterpreter(mock_well)
        matches = interp.get_alias_matches()

        # Should find GR, RHOB, NPHI, RT, DT
        assert matches['GR'] == 'GR'
        assert matches['RHOB'] == 'RHOB'
        assert matches['NPHI'] == 'NPHI'
        assert matches['RT'] == 'RT'
        assert matches['DT'] == 'DT'

        # SP should not be found
        assert matches['SP'] is None

    def test_check_inputs(self, mock_well):
        """Test input checking functionality."""
        from welly.petro import PetroInterpreter

        interp = PetroInterpreter(mock_well)
        status = interp.check_inputs()

        # Should have found curves
        assert 'GR' in status['curves_found']
        assert 'RHOB' in status['curves_found']

        # SP should be missing
        assert 'SP' in status['curves_missing']

        # vshale_gr should be ready
        assert 'vshale_gr' in status['calculations_ready']

        # vshale_sp should be blocked
        assert 'vshale_sp' in status['calculations_blocked']

    def test_missing_curve_error(self, mock_well):
        """Test MissingCurveError is raised with helpful message."""
        from welly.petro import PetroInterpreter
        from welly.petro.interpreter import MissingCurveError

        # Remove GR curve
        del mock_well.data['GR']

        interp = PetroInterpreter(mock_well)

        with pytest.raises(MissingCurveError) as exc_info:
            interp.vshale()

        # Check error message contains helpful info
        assert 'GR' in str(exc_info.value)
        assert 'add_alias' in str(exc_info.value)

    def test_interpreter_with_alias_file(self, mock_well, tmp_path):
        """Test creating interpreter with alias file."""
        from welly.petro import PetroInterpreter

        # Create alias file
        alias_file = tmp_path / 'aliases.json'
        alias_file.write_text('{"aliases": {"GR": ["GAMMA_RAY"]}}')

        # Create interpreter with alias file
        interp = PetroInterpreter(mock_well, alias_file=alias_file)

        # Should have merged aliases
        assert 'GAMMA_RAY' in interp.alias['GR']
        assert 'GR' in interp.alias['GR']  # Default still there

    def test_interpreter_with_custom_alias_dict(self, mock_well):
        """Test creating interpreter with custom alias dict."""
        from welly.petro import PetroInterpreter

        custom_alias = {'GR': ['MY_GR'], 'CUSTOM': ['C1', 'C2']}
        interp = PetroInterpreter(mock_well, alias=custom_alias)

        # Should have merged
        assert 'MY_GR' in interp.alias['GR']
        assert 'GR' in interp.alias['GR']  # Default still there
        assert 'CUSTOM' in interp.alias


    def test_plot_basic(self, mock_well):
        """Test basic plot generation."""
        from welly.petro import PetroInterpreter
        import matplotlib.pyplot as plt

        params = PetrophysicalParameters.default_sandstone()
        params.clay.gr_clean = 20
        params.clay.gr_shale = 120
        params.fluid.rw = 0.05

        interp = PetroInterpreter(mock_well, params=params)
        interp.run_standard_interpretation(porosity_method='density')

        # Generate plot
        fig = interp.plot()

        assert fig is not None
        assert hasattr(fig, 'savefig')  # It's a matplotlib Figure

        plt.close(fig)

    def test_plot_custom_tracks(self, mock_well):
        """Test plot with custom track selection."""
        from welly.petro import PetroInterpreter
        import matplotlib.pyplot as plt

        params = PetrophysicalParameters.default_sandstone()
        params.clay.gr_clean = 20
        params.clay.gr_shale = 120
        params.fluid.rw = 0.05

        interp = PetroInterpreter(mock_well, params=params)
        interp.run_standard_interpretation(porosity_method='density')

        # Generate plot with specific tracks
        fig = interp.plot(tracks=['GR', 'VSH', 'PHI'])

        assert fig is not None
        # Should have 3 axes
        assert len(fig.axes) >= 3

        plt.close(fig)

    def test_plot_depth_range(self, mock_well):
        """Test plot with depth range."""
        from welly.petro import PetroInterpreter
        import matplotlib.pyplot as plt

        params = PetrophysicalParameters.default_sandstone()
        params.clay.gr_clean = 20
        params.clay.gr_shale = 120
        params.fluid.rw = 0.05

        interp = PetroInterpreter(mock_well, params=params)
        interp.run_standard_interpretation(porosity_method='density')

        # Generate plot with depth range
        fig = interp.plot(depth_range=(1020, 1080))

        assert fig is not None
        # Check y-axis limits
        ax = fig.axes[0]
        ylim = ax.get_ylim()
        assert ylim[0] >= 1020 or ylim[1] <= 1080

        plt.close(fig)

    def test_plot_show_inputs_false(self, mock_well):
        """Test plot without input curves."""
        from welly.petro import PetroInterpreter
        import matplotlib.pyplot as plt

        params = PetrophysicalParameters.default_sandstone()
        params.clay.gr_clean = 20
        params.clay.gr_shale = 120
        params.fluid.rw = 0.05

        interp = PetroInterpreter(mock_well, params=params)
        interp.run_standard_interpretation(porosity_method='density')

        # Generate plot without input curves
        fig = interp.plot(show_inputs=False)

        assert fig is not None
        # Should have fewer axes than default
        fig_default = interp.plot()
        assert len(fig.axes) <= len(fig_default.axes)

        plt.close(fig)
        plt.close(fig_default)

    def test_plot_custom_figsize(self, mock_well):
        """Test plot with custom figure size."""
        from welly.petro import PetroInterpreter
        import matplotlib.pyplot as plt

        params = PetrophysicalParameters.default_sandstone()
        params.clay.gr_clean = 20
        params.clay.gr_shale = 120
        params.fluid.rw = 0.05

        interp = PetroInterpreter(mock_well, params=params)
        interp.run_standard_interpretation(porosity_method='density')

        # Generate plot with custom size
        fig = interp.plot(figsize=(20, 15))

        assert fig is not None
        # Check figure size
        size = fig.get_size_inches()
        assert size[0] == 20
        assert size[1] == 15

        plt.close(fig)

    def test_plot_no_interpretation_raises(self, mock_well):
        """Test that plot raises error if no interpretation run."""
        from welly.petro import PetroInterpreter

        interp = PetroInterpreter(mock_well)

        # Should raise because no computed curves
        with pytest.raises(ValueError, match="No tracks to plot"):
            interp.plot(show_inputs=False)
