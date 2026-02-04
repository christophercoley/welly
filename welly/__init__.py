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
           'read_las',
           'describe_dlis',
           'load_images',
          ]


if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

__version__ = metadata.version(__name__)

