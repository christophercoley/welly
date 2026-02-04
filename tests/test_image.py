"""
Tests for borehole image functionality.

:copyright: 2024 Agile Scientific
:license: Apache 2.0
"""
import os
import tempfile

import numpy as np
import pytest

from welly.image import ImageCurve


class TestImageCurve:
    """Tests for the ImageCurve class."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image curve for testing."""
        n_depths = 1000
        n_azimuths = 360
        
        # Create synthetic image data
        depths = np.linspace(5000, 5100, n_depths)
        data = np.random.rand(n_depths, n_azimuths) * 100
        
        # Add some null values
        data[0:10, :] = -9999.0
        
        return ImageCurve(
            data=data,
            index=depths,
            mnemonic='TEST_IMAGE',
            units='mS/m',
            index_units='ft',
            description='Test image curve',
        )
    
    def test_image_creation(self, sample_image):
        """Test basic ImageCurve creation."""
        assert sample_image.mnemonic == 'TEST_IMAGE'
        assert sample_image.units == 'mS/m'
        assert sample_image.index_units == 'ft'
        assert sample_image.n_azimuths == 360
        assert sample_image.n_samples == 1000
        assert sample_image.shape == (1000, 360)
    
    def test_image_depth_range(self, sample_image):
        """Test depth range properties."""
        assert sample_image.start == 5000.0
        assert sample_image.stop == 5100.0
    
    def test_null_value_replacement(self, sample_image):
        """Test that null values are replaced with NaN."""
        # First 10 rows should be NaN (were -9999)
        assert np.all(np.isnan(sample_image.data[0:10, :]))
        # Rest should not be NaN
        assert not np.any(np.isnan(sample_image.data[10:, :]))
    
    def test_get_section(self, sample_image):
        """Test extracting a depth section."""
        section = sample_image.get_section(5020, 5050)
        
        assert section.start >= 5020
        assert section.stop <= 5050
        assert section.n_azimuths == 360
        assert section.mnemonic == 'TEST_IMAGE'
    
    def test_repr(self, sample_image):
        """Test string representation."""
        repr_str = repr(sample_image)
        assert 'TEST_IMAGE' in repr_str
        assert '1000' in repr_str or '360' in repr_str
    
    def test_invalid_data_dimensions(self):
        """Test that 1D data raises an error."""
        with pytest.raises(ValueError, match="must be 2D"):
            ImageCurve(
                data=np.random.rand(100),  # 1D data
                index=np.linspace(0, 100, 100),
            )
    
    def test_mismatched_index_length(self):
        """Test that mismatched index length raises an error."""
        with pytest.raises(ValueError, match="must match"):
            ImageCurve(
                data=np.random.rand(100, 360),
                index=np.linspace(0, 100, 50),  # Wrong length
            )


class TestImagePlotting:
    """Tests for ImageCurve plotting functionality."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image curve for testing."""
        n_depths = 500
        n_azimuths = 360
        depths = np.linspace(5000, 5050, n_depths)
        data = np.random.rand(n_depths, n_azimuths) * 100
        
        return ImageCurve(
            data=data,
            index=depths,
            mnemonic='TEST_IMAGE',
            units='mS/m',
        )
    
    def test_plot_returns_axes(self, sample_image):
        """Test that plot() returns an axes object."""
        import matplotlib.pyplot as plt
        
        ax = sample_image.plot()
        assert ax is not None
        plt.close('all')
    
    def test_plot_with_custom_colormap(self, sample_image):
        """Test plotting with custom colormap."""
        import matplotlib.pyplot as plt
        
        ax = sample_image.plot(cmap='viridis')
        assert ax is not None
        plt.close('all')
    
    def test_plot_with_custom_limits(self, sample_image):
        """Test plotting with custom vmin/vmax."""
        import matplotlib.pyplot as plt
        
        ax = sample_image.plot(vmin=10, vmax=90)
        assert ax is not None
        plt.close('all')


class TestImageExport:
    """Tests for ImageCurve export functionality."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image curve for testing."""
        n_depths = 2000
        n_azimuths = 360
        depths = np.linspace(5000, 5200, n_depths)
        data = np.random.rand(n_depths, n_azimuths) * 100
        
        return ImageCurve(
            data=data,
            index=depths,
            mnemonic='TEST_IMAGE',
            units='mS/m',
        )
    
    def test_to_pdf(self, sample_image):
        """Test PDF export."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_path = f.name
        
        try:
            n_pages = sample_image.to_pdf(
                temp_path,
                feet_per_page=100,
                show_progress=False
            )
            
            assert n_pages == 2  # 200ft / 100ft per page
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_to_png_series(self, sample_image):
        """Test PNG series export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filenames = sample_image.to_png_series(
                temp_dir,
                prefix='test',
                feet_per_image=100,
                show_progress=False
            )
            
            assert len(filenames) == 2
            for fname in filenames:
                assert os.path.exists(fname)
                assert fname.endswith('.png')
