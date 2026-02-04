"""
Module for handling borehole image logs (FMI, UBI, etc.)

:copyright: 2024 Agile Scientific
:license: Apache 2.0
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class ImageCurve:
    """
    A class for handling 2D borehole image data (e.g., FMI, UBI).
    
    Borehole images are 2D arrays where:
    - Rows correspond to depth samples
    - Columns correspond to azimuthal samples (typically 0-360 degrees)
    
    Args:
        data (ndarray): 2D array of image data (n_depths x n_azimuths)
        index (ndarray): 1D array of depth values
        mnemonic (str): Name/mnemonic of the image curve
        units (str): Units of the image values (e.g., 'mS/m', 'ohm.m')
        index_units (str): Units of the depth index (e.g., 'ft', 'm')
        description (str): Description of the image curve
        null_value (float): Value representing null/missing data
        azimuth_reference (str): Reference for azimuth (e.g., 'north', 'highside')
        
    Attributes:
        data (ndarray): The 2D image data
        index (ndarray): The depth index
        mnemonic (str): Curve mnemonic
        units (str): Data units
        index_units (str): Index units
        description (str): Curve description
        n_azimuths (int): Number of azimuthal samples
        start (float): Start depth
        stop (float): Stop depth
    """
    
    def __init__(self,
                 data,
                 index,
                 mnemonic='IMAGE',
                 units=None,
                 index_units='ft',
                 description='',
                 null_value=-9999.0,
                 azimuth_reference='north'):
        
        # Validate inputs
        if data.ndim != 2:
            raise ValueError(f"Image data must be 2D, got {data.ndim}D")
        
        if len(index) != data.shape[0]:
            raise ValueError(
                f"Index length ({len(index)}) must match data rows ({data.shape[0]})"
            )
        
        # Store data, replacing null values with NaN
        self.data = data.astype(float).copy()
        if null_value is not None:
            self.data[np.isclose(self.data, null_value, rtol=1e-5)] = np.nan
        
        self.index = np.asarray(index)
        self.mnemonic = mnemonic
        self.units = units
        self.index_units = index_units
        self.description = description
        self.azimuth_reference = azimuth_reference
        
    @property
    def n_azimuths(self):
        """Number of azimuthal samples."""
        return self.data.shape[1]
    
    @property
    def n_samples(self):
        """Number of depth samples."""
        return self.data.shape[0]
    
    @property
    def start(self):
        """Start depth."""
        return float(self.index[0])
    
    @property
    def stop(self):
        """Stop depth."""
        return float(self.index[-1])
    
    @property
    def shape(self):
        """Shape of the image data."""
        return self.data.shape
    
    def __repr__(self):
        return (
            f"ImageCurve(mnemonic={self.mnemonic}, "
            f"shape={self.shape}, "
            f"start={self.start:.1f}, stop={self.stop:.1f}, "
            f"units={self.units})"
        )
    
    def _repr_html_(self):
        """Jupyter notebook representation."""
        rows = f'<tr><th colspan="2">{self.mnemonic}</th></tr>'
        rows += f'<tr><td>Shape</td><td>{self.shape}</td></tr>'
        rows += f'<tr><td>Depth range</td><td>{self.start:.1f} - {self.stop:.1f} {self.index_units}</td></tr>'
        rows += f'<tr><td>Azimuths</td><td>{self.n_azimuths}</td></tr>'
        rows += f'<tr><td>Units</td><td>{self.units}</td></tr>'
        rows += f'<tr><td>Description</td><td>{self.description}</td></tr>'
        return f'<table>{rows}</table>'
    
    def get_section(self, top, bottom):
        """
        Extract a depth section from the image.
        
        Args:
            top (float): Top depth of section
            bottom (float): Bottom depth of section
            
        Returns:
            ImageCurve: New ImageCurve containing only the section
        """
        top_idx = np.searchsorted(self.index, top)
        bottom_idx = np.searchsorted(self.index, bottom)
        
        return ImageCurve(
            data=self.data[top_idx:bottom_idx],
            index=self.index[top_idx:bottom_idx],
            mnemonic=self.mnemonic,
            units=self.units,
            index_units=self.index_units,
            description=self.description,
            null_value=None,  # Already handled
            azimuth_reference=self.azimuth_reference,
        )
    
    def plot(self,
             ax=None,
             cmap='YlOrBr',
             vmin=None,
             vmax=None,
             percentile_clip=(5, 95),
             show_colorbar=True,
             azimuth_labels=True,
             title=None,
             **kwargs):
        """
        Plot the borehole image.
        
        Args:
            ax: Matplotlib axes. If None, creates new figure.
            cmap (str): Colormap name. Default 'YlOrBr' (common for FMI).
            vmin (float): Minimum value for colormap. If None, uses percentile.
            vmax (float): Maximum value for colormap. If None, uses percentile.
            percentile_clip (tuple): Percentiles for auto vmin/vmax. Default (5, 95).
            show_colorbar (bool): Whether to show colorbar. Default True.
            azimuth_labels (bool): Show N/E/S/W labels. Default True.
            title (str): Plot title. If None, uses mnemonic.
            **kwargs: Additional arguments passed to imshow.
            
        Returns:
            matplotlib.axes.Axes: The axes object
        """
        if ax is None:
            # Calculate figure height based on depth range
            depth_range = abs(self.stop - self.start)
            fig_height = min(max(depth_range / 10, 8), 24)  # 8-24 inches
            fig, ax = plt.subplots(figsize=(8, fig_height))
        
        # Calculate color limits
        if vmin is None:
            vmin = np.nanpercentile(self.data, percentile_clip[0])
        if vmax is None:
            vmax = np.nanpercentile(self.data, percentile_clip[1])
        
        # Extent: [left, right, bottom, top]
        # For depth plots, bottom > top so depth increases downward
        extent = [0, 360, self.stop, self.start]
        
        im = ax.imshow(
            self.data,
            aspect='auto',
            cmap=cmap,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )
        
        # Labels
        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel(f'Depth ({self.index_units})')
        
        if title is None:
            title = self.mnemonic
        ax.set_title(title)
        
        # Azimuth ticks
        if azimuth_labels:
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set_xticklabels(['N', 'E', 'S', 'W', 'N'])
        
        # Colorbar
        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            if self.units:
                cbar.set_label(self.units)
        
        return ax
    
    def to_pdf(self,
               filename,
               feet_per_page=100,
               cmap='YlOrBr',
               vmin=None,
               vmax=None,
               percentile_clip=(5, 95),
               dpi=100,
               show_progress=True):
        """
        Export the image to a multi-page PDF.
        
        Creates a PDF with one page per depth interval, similar to
        traditional well log prints.
        
        Args:
            filename (str): Output PDF filename
            feet_per_page (float): Depth interval per page. Default 100.
            cmap (str): Colormap name. Default 'YlOrBr'.
            vmin (float): Minimum value for colormap.
            vmax (float): Maximum value for colormap.
            percentile_clip (tuple): Percentiles for auto vmin/vmax.
            dpi (int): Resolution. Default 100.
            show_progress (bool): Print progress. Default True.
            
        Returns:
            int: Number of pages created
        """
        # Calculate global color limits for consistency across pages
        if vmin is None:
            vmin = np.nanpercentile(self.data, percentile_clip[0])
        if vmax is None:
            vmax = np.nanpercentile(self.data, percentile_clip[1])
        
        # Determine page boundaries
        start_depth = int(np.ceil(self.start / feet_per_page) * feet_per_page)
        end_depth = int(np.floor(self.stop / feet_per_page) * feet_per_page)
        
        n_pages = (end_depth - start_depth) // int(feet_per_page)
        
        if show_progress:
            print(f"Creating {n_pages} page PDF: {filename}")
            print(f"Depth range: {start_depth} to {end_depth} {self.index_units}")
        
        page_count = 0
        
        with PdfPages(filename) as pdf:
            for page_top in range(start_depth, end_depth, int(feet_per_page)):
                page_bottom = page_top + feet_per_page
                
                # Find indices
                top_idx = np.searchsorted(self.index, page_top)
                bottom_idx = np.searchsorted(self.index, page_bottom)
                
                if bottom_idx <= top_idx:
                    continue
                
                section_depth = self.index[top_idx:bottom_idx]
                section_image = self.data[top_idx:bottom_idx, :]
                
                # Calculate figure height based on samples
                # Target ~500 samples per inch for good resolution
                n_samples = len(section_depth)
                fig_height = max(n_samples / 500, 12)
                fig_height = min(fig_height, 36)  # Cap at 36 inches
                
                fig, ax = plt.subplots(figsize=(8, fig_height))
                
                extent = [0, 360, section_depth[-1], section_depth[0]]
                
                im = ax.imshow(
                    section_image,
                    aspect='auto',
                    cmap=cmap,
                    extent=extent,
                    vmin=vmin,
                    vmax=vmax,
                )
                
                ax.set_xlabel('Azimuth (degrees)', fontsize=12)
                ax.set_ylabel(f'Depth ({self.index_units})', fontsize=12)
                ax.set_title(
                    f'{self.mnemonic}: {page_top}-{page_bottom} {self.index_units}',
                    fontsize=14
                )
                
                ax.set_xticks([0, 90, 180, 270, 360])
                ax.set_xticklabels(['N', 'E', 'S', 'W', 'N'])
                
                # Depth ticks every 10 units
                tick_interval = 10 if feet_per_page >= 50 else 5
                depth_ticks = np.arange(page_top, page_bottom + 1, tick_interval)
                ax.set_yticks(depth_ticks)
                
                cbar = plt.colorbar(im, ax=ax, shrink=0.5)
                if self.units:
                    cbar.set_label(self.units, fontsize=10)
                
                plt.tight_layout()
                pdf.savefig(fig, dpi=dpi)
                plt.close(fig)
                
                page_count += 1
                
                if show_progress and page_count % 10 == 0:
                    print(f"  Completed {page_count}/{n_pages} pages...")
        
        if show_progress:
            print(f"Saved: {filename} ({page_count} pages)")
        
        return page_count
    
    def to_png_series(self,
                      output_dir,
                      prefix='image',
                      feet_per_image=100,
                      cmap='YlOrBr',
                      vmin=None,
                      vmax=None,
                      percentile_clip=(5, 95),
                      dpi=150,
                      show_progress=True):
        """
        Export the image as a series of PNG files.
        
        Args:
            output_dir (str): Output directory
            prefix (str): Filename prefix. Default 'image'.
            feet_per_image (float): Depth interval per image. Default 100.
            cmap (str): Colormap name.
            vmin (float): Minimum value for colormap.
            vmax (float): Maximum value for colormap.
            percentile_clip (tuple): Percentiles for auto vmin/vmax.
            dpi (int): Resolution. Default 150.
            show_progress (bool): Print progress.
            
        Returns:
            list: List of created filenames
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate global color limits
        if vmin is None:
            vmin = np.nanpercentile(self.data, percentile_clip[0])
        if vmax is None:
            vmax = np.nanpercentile(self.data, percentile_clip[1])
        
        start_depth = int(np.ceil(self.start / feet_per_image) * feet_per_image)
        end_depth = int(np.floor(self.stop / feet_per_image) * feet_per_image)
        
        filenames = []
        
        for page_top in range(start_depth, end_depth, int(feet_per_image)):
            page_bottom = page_top + feet_per_image
            
            section = self.get_section(page_top, page_bottom)
            
            if section.n_samples == 0:
                continue
            
            fig, ax = plt.subplots(figsize=(8, 12))
            section.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
            
            filename = os.path.join(
                output_dir,
                f'{prefix}_{page_top:05d}_{page_bottom:05d}.png'
            )
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            filenames.append(filename)
            
            if show_progress:
                print(f"  Saved: {filename}")
        
        return filenames
