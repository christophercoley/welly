"""
==================
welly
==================
"""
import sys

from .project import Project
from .well import Well
from .header import Header
from .curve import Curve
from .synthetic import Synthetic
from .location import Location
from .crs import CRS
from . import tools
from . import quality
from . import defaults
from . import petro


def read_las(path, **kwargs):
    """
    A package namespace method to be called as `welly.read_las`.

    Just wraps `Project.from_las()`. Creates a `Project` from a .LAS file.

    Args:
        path (str): path or URL where LAS is located. `*.las` to load all files
            in dir
        **kwargs (): See `Project.from_las()`` for addictional arguments

    Returns:
        welly.Project. The Project object.
    """
    return Project.from_las(path, **kwargs)


def read_df(df, **kwargs):
    """
    A package namespace method to be called as `welly.read_df`.

    Just wraps `Well.from_df()`. Creates a `Well` from your pd.DataFrame.

    Args:
        df (pd.DataFrame): Column data and column names

        Optional **kwargs:
            units (dict): Optional. Units of measurement of the curves in `df`.
            req (list): Optional. An alias list, giving all required curves.
            uwi (str): Unique Well Identifier (UWI)
            name (str): Name

    Returns:
        Well. The `Well` object.
    """
    return Well.from_df(df, **kwargs)


def describe_dlis(fname, **kwargs):
    """
    Describe the contents of a DLIS file without fully loading it.
    
    This function provides a summary of what's in a DLIS file, including:
    - Logical files and their origins (well name, company, etc.)
    - Frames and their curves
    - Tools used for logging
    
    Requires dlisio: pip install welly[dlis]
    
    Args:
        fname (str): Path to the DLIS file.
        **kwargs: Additional arguments passed to the underlying function.
    
    Returns:
        dict: Summary of the DLIS file contents.
        
    Example:
        >>> import welly
        >>> info = welly.describe_dlis('well.dlis')
        >>> for lf in info['logical_files']:
        ...     print(f"Well: {lf['well_name']}")
        ...     for frame in lf['frames']:
        ...         print(f"  Frame {frame['name']}: {frame['n_curves']} curves")
    """
    from .dlis import describe_dlis as _describe_dlis
    return _describe_dlis(fname, **kwargs)


def load_images(fname, **kwargs):
    """
    Load borehole image data from a DLIS file.
    
    This function loads 2D image data (like FMI, UBI) from DLIS files.
    For 1D curves, use Well.from_dlis() instead.
    
    Requires dlisio: pip install welly[dlis]
    
    Args:
        fname (str): Path to the DLIS file.
        frame (str): Optional. Name of the frame to load.
        logical_file (int): Optional. Index of the logical file. Default 0.
        **kwargs: Additional arguments.
    
    Returns:
        dict: Dictionary mapping image names to ImageCurve objects.
        
    Example:
        >>> import welly
        >>> images = welly.load_images('fmi_data.dlis')
        >>> fmi = images['FMI_DYN']
        >>> fmi.plot()  # Quick plot
        >>> fmi.to_pdf('fmi_output.pdf', feet_per_page=100)  # Multi-page PDF
    """
    from .dlis import load_images_from_dlis
    return load_images_from_dlis(fname, **kwargs)


def describe_image_channels(fname, **kwargs):
    """
    Describe image channels in a DLIS file using only metadata.

    No image array data is loaded — only the depth index is read.
    Safe to call on multi-GB files.

    Requires dlisio: pip install welly[dlis]

    Args:
        fname (str): Path to the DLIS file.
        **kwargs: Additional arguments (logical_file, error_handling).

    Returns:
        list: List of dicts describing each image channel.
    """
    from .dlis import describe_image_channels as _describe
    return _describe(fname, **kwargs)


def load_single_image(fname, channel_name, **kwargs):
    """
    Load a single image channel from a DLIS file.

    Memory-efficient alternative to load_images() when you only need
    one channel. Loads only the depth index, the requested channel,
    and any orientation curve.

    Requires dlisio: pip install welly[dlis]

    Args:
        fname (str): Path to the DLIS file.
        channel_name (str): Name of the image channel to load.
        **kwargs: Additional arguments (logical_file, error_handling).

    Returns:
        ImageCurve: The requested image channel.
    """
    from .dlis import load_single_image as _load
    return _load(fname, channel_name, **kwargs)


__all__ = [
           'Project',
           'Well',
           'Header',
           'Curve',
           'Synthetic',
           'Location',
           'CRS',
           'quality',
           'tools',  # Various classes in here
           'petro',  # Petrophysics module
           'read_las',
           'describe_dlis',
           'describe_image_channels',
           'load_images',
           'load_single_image',
          ]


if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

__version__ = metadata.version(__name__)

