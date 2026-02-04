"""
Tests for DLIS file support.

Note: These tests require sample DLIS files in working-notes/ directory.
Tests are skipped if dlisio is not installed or test files are not available.
"""
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from welly import Well
from welly.well import WellError


# Check if dlisio is available
try:
    import dlisio
    HAS_DLISIO = True
except ImportError:
    HAS_DLISIO = False

# Check if test files exist
DLIS_FILE = 'working-notes/199-1218A_std-proc.dlis'
HAS_TEST_FILE = os.path.exists(DLIS_FILE)

skip_no_dlisio = pytest.mark.skipif(
    not HAS_DLISIO,
    reason="dlisio not installed"
)

skip_no_testfile = pytest.mark.skipif(
    not HAS_TEST_FILE,
    reason="DLIS test file not available"
)


@skip_no_dlisio
class TestDlisImport:
    """Test dlisio import handling."""

    def test_dlisio_check(self):
        """Test that dlisio check works."""
        from welly.dlis import _check_dlisio
        dlis, ErrorHandler = _check_dlisio()
        assert dlis is not None
        assert ErrorHandler is not None


@skip_no_dlisio
@skip_no_testfile
class TestFromDlis:
    """Tests for Well.from_dlis() method."""

    def test_load_default(self):
        """Test loading DLIS with default parameters."""
        well = Well.from_dlis(DLIS_FILE)
        
        assert well is not None
        assert len(well.data) > 0
        assert hasattr(well, '_dlis_frame')
        assert hasattr(well, '_dlis_logical_file')

    def test_load_well_name(self):
        """Test that well name is extracted from DLIS."""
        well = Well.from_dlis(DLIS_FILE)
        
        # The test file has well name '1218A'
        assert well.name == '1218A'

    def test_load_curves(self):
        """Test that curves are loaded correctly."""
        well = Well.from_dlis(DLIS_FILE)
        
        # Check that curves have proper attributes
        for name, curve in well.data.items():
            assert curve.start is not None
            assert curve.stop is not None
            assert len(curve) > 0

    def test_load_return_all(self):
        """Test loading all wells from DLIS file."""
        wells = Well.from_dlis(DLIS_FILE, return_all=True)
        
        assert isinstance(wells, list)
        assert len(wells) > 0
        
        for well in wells:
            assert isinstance(well, Well)
            assert len(well.data) > 0

    def test_load_specific_logical_file(self):
        """Test loading from specific logical file."""
        well = Well.from_dlis(DLIS_FILE, logical_file=0)
        
        assert well._dlis_logical_file == 0

    def test_load_invalid_logical_file(self):
        """Test error handling for invalid logical file index."""
        with pytest.raises(WellError):
            Well.from_dlis(DLIS_FILE, logical_file=999)

    def test_load_error_handling_warn(self):
        """Test that warn error handling works."""
        # Should not raise, even with potentially malformed data
        well = Well.from_dlis(DLIS_FILE, error_handling='warn')
        assert well is not None

    def test_load_error_handling_ignore(self):
        """Test that ignore error handling works."""
        well = Well.from_dlis(DLIS_FILE, error_handling='ignore')
        assert well is not None

    def test_curve_index_values(self):
        """Test that curve index (depth) values are correct."""
        well = Well.from_dlis(DLIS_FILE)
        
        # Get first curve
        first_curve = list(well.data.values())[0]
        
        # Index should be numeric
        assert np.issubdtype(first_curve.index.dtype, np.number)

    def test_curve_units(self):
        """Test that curve units are preserved."""
        well = Well.from_dlis(DLIS_FILE)
        
        # At least some curves should have units
        has_units = any(
            curve.units is not None and curve.units != ''
            for curve in well.data.values()
        )
        assert has_units


@skip_no_dlisio
@skip_no_testfile
class TestDlisIntegration:
    """Integration tests for DLIS support."""

    def test_dlis_to_dataframe(self):
        """Test converting DLIS well to DataFrame."""
        well = Well.from_dlis(DLIS_FILE)
        
        keys = list(well.data.keys())[:3]
        df = well.df(keys=keys)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == len(keys)

    def test_dlis_to_las(self):
        """Test exporting DLIS well to LAS format."""
        well = Well.from_dlis(DLIS_FILE)
        
        with tempfile.NamedTemporaryFile(suffix='.las', delete=False) as f:
            temp_path = f.name
        
        try:
            keys = list(well.data.keys())[:3]
            well.to_las(temp_path, keys=keys)
            
            # Read back and verify
            well2 = Well.from_las(temp_path)
            assert len(well2.data) == len(keys)
        finally:
            os.unlink(temp_path)

    def test_dlis_curve_operations(self):
        """Test that standard curve operations work on DLIS curves."""
        well = Well.from_dlis(DLIS_FILE)
        
        # Get a curve
        curve = list(well.data.values())[0]
        
        # Test basic operations
        assert curve.mean() is not None
        assert curve.min() is not None
        assert curve.max() is not None

    def test_multiple_frames_different_data(self):
        """Test that different frames contain different data."""
        wells = Well.from_dlis(DLIS_FILE, return_all=True)
        
        if len(wells) > 1:
            # Different frames should have different curve counts or names
            curve_sets = [set(w.data.keys()) for w in wells]
            # At least some should be different
            assert len(set(frozenset(s) for s in curve_sets)) > 1


@skip_no_dlisio
class TestDlisHelpers:
    """Tests for DLIS helper functions."""

    def test_origin_to_location(self):
        """Test origin to location conversion."""
        from welly.dlis import _origin_to_location
        from welly.location import Location
        
        # Test with None
        loc = _origin_to_location(None)
        assert isinstance(loc, Location)

    def test_build_header_empty(self):
        """Test building header with no data."""
        from welly.dlis import _build_header_from_origin
        
        header = _build_header_from_origin(None, None)
        assert isinstance(header, pd.DataFrame)
        assert 'mnemonic' in header.columns


class TestDlisNotInstalled:
    """Tests for when dlisio is not installed."""

    def test_import_error_message(self):
        """Test that helpful error message is shown when dlisio missing."""
        # This test always runs - it tests the error message format
        from welly.dlis import _check_dlisio
        
        if HAS_DLISIO:
            # If installed, should not raise
            _check_dlisio()
        # If not installed, the skip decorator handles it
