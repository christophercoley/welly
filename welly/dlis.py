"""
Module for reading DLIS files.

:copyright: 2024 Agile Scientific
:license: Apache 2.0
"""
import warnings

import numpy as np
import pandas as pd

from .curve import Curve
from .location import Location


def _check_dlisio():
    """Check if dlisio is installed and return it."""
    try:
        from dlisio import dlis
        from dlisio.common import ErrorHandler
        return dlis, ErrorHandler
    except ImportError:
        raise ImportError(
            "dlisio is required for DLIS support. "
            "Install with: pip install dlisio"
        )


def describe_dlis(fname, error_handling='warn'):
    """
    Describe the contents of a DLIS file without fully loading it.
    
    This function provides a summary of what's in a DLIS file, including:
    - Logical files and their origins (well name, company, etc.)
    - Frames and their curves
    - Tools used for logging
    
    Args:
        fname (str): Path to the DLIS file.
        error_handling (str): How to handle DLIS parsing errors.
            'warn' (default), 'strict', or 'ignore'.
    
    Returns:
        dict: Summary of the DLIS file contents with keys:
            - 'filename': The file path
            - 'logical_files': List of logical file summaries
            
    Example:
        >>> from welly import describe_dlis
        >>> info = describe_dlis('well.dlis')
        >>> print(info['logical_files'][0]['well_name'])
        'My Well'
        >>> for frame in info['logical_files'][0]['frames']:
        ...     print(f"{frame['name']}: {frame['curves']}")
    """
    dlis_module, ErrorHandler = _check_dlisio()
    
    # Configure error handling
    if error_handling == 'strict':
        handler = None
    elif error_handling == 'ignore':
        handler = ErrorHandler(
            critical=ErrorHandler.swallow,
            major=ErrorHandler.swallow,
            minor=ErrorHandler.swallow,
        )
    else:
        handler = ErrorHandler(critical=ErrorHandler.swallow)
    
    load_kwargs = {'error_handler': handler} if handler else {}
    
    result = {
        'filename': str(fname),
        'logical_files': []
    }
    
    with dlis_module.load(fname, **load_kwargs) as files:
        for lf_idx, logical_f in enumerate(files):
            lf_info = {
                'index': lf_idx,
                'well_name': None,
                'field_name': None,
                'company': None,
                'creation_time': None,
                'tools': [],
                'frames': []
            }
            
            # Extract origin info
            try:
                origins = logical_f.origins
                if origins:
                    origin = origins[0]
                    lf_info['well_name'] = getattr(origin, 'well_name', None)
                    lf_info['field_name'] = getattr(origin, 'field_name', None)
                    lf_info['company'] = getattr(origin, 'company', None)
                    lf_info['creation_time'] = str(getattr(origin, 'creation_time', None))
            except Exception:
                pass
            
            # Extract tools info
            try:
                tools = logical_f.tools
                for tool in tools:
                    tool_info = {
                        'name': tool.name,
                        'description': getattr(tool, 'description', None),
                        'trademark': getattr(tool, 'trademark_name', None),
                        'generic_name': getattr(tool, 'generic_name', None),
                    }
                    lf_info['tools'].append(tool_info)
            except Exception:
                pass
            
            # Extract frames info
            try:
                frames = logical_f.frames
                for frame in frames:
                    frame_info = {
                        'name': frame.name,
                        'description': getattr(frame, 'description', None),
                        'index_type': getattr(frame, 'index_type', None),
                        'n_curves': len(frame.channels) - 1,  # Exclude index
                        'curves': [ch.name for ch in frame.channels[1:]],  # Exclude index
                    }
                    
                    # Try to get depth range (load only the index channel)
                    try:
                        index_ch = frame.channels[0]
                        data = _safe_curves(frame, channels=[index_ch])
                        if data is not None and len(data) > 0:
                            index_vals = data[index_ch.name]
                            frame_info['start'] = float(index_vals[0])
                            frame_info['stop'] = float(index_vals[-1])
                            frame_info['n_samples'] = len(index_vals)
                            frame_info['index_units'] = getattr(index_ch, 'units', None)
                    except Exception:
                        pass
                    
                    lf_info['frames'].append(frame_info)
            except Exception:
                pass
            
            result['logical_files'].append(lf_info)
    
    return result


def _get_tools_from_logical_file(logical_file):
    """
    Extract tool information from a DLIS logical file.
    
    Args:
        logical_file: dlisio LogicalFile object
        
    Returns:
        list: List of tool info dicts
    """
    tools = []
    try:
        for tool in logical_file.tools:
            tool_info = {
                'name': tool.name,
                'description': getattr(tool, 'description', None),
                'trademark': getattr(tool, 'trademark_name', None),
                'generic_name': getattr(tool, 'generic_name', None),
            }
            tools.append(tool_info)
    except Exception:
        pass
    return tools


# Unit conversion factors to feet
_UNIT_TO_FEET = {
    # Length units
    'ft': 1.0,
    'f': 1.0,
    'feet': 1.0,
    'foot': 1.0,
    'm': 3.28084,
    'meter': 3.28084,
    'meters': 3.28084,
    'metre': 3.28084,
    'metres': 3.28084,
    'in': 1/12,
    'inch': 1/12,
    'inches': 1/12,
    '0.1 in': 1/120,  # Tenths of an inch (common in DLIS)
    '0.1in': 1/120,
    'cm': 0.0328084,
    'mm': 0.00328084,
}


def _convert_index_to_feet(index_values, units):
    """
    Convert index values to feet if possible.
    
    Args:
        index_values: numpy array of index values
        units: string unit identifier
        
    Returns:
        tuple: (converted_values, new_units) or (original_values, original_units)
    """
    if units is None:
        return index_values, units
    
    # Normalize unit string
    unit_lower = str(units).lower().strip()
    
    # Check if we have a conversion factor
    if unit_lower in _UNIT_TO_FEET:
        factor = _UNIT_TO_FEET[unit_lower]
        if factor != 1.0:
            converted = index_values * factor
            return converted, 'ft'
    
    return index_values, units


def _safe_curves(frame, channels=None):
    """
    Load curves from a frame, with fallback for older dlisio versions.

    Tries selective loading via ``frame.curves(channels=...)`` first.
    Falls back to ``frame.curves()`` if the keyword isn't supported
    (dlisio <= 1.0.4).

    Args:
        frame: dlisio Frame object
        channels: Optional list of channel objects to load selectively.

    Returns:
        Structured numpy array of curve data.
    """
    if channels is not None:
        try:
            return frame.curves(channels=channels)
        except TypeError:
            pass
    return frame.curves()


def _get_index_channel(frame):
    """
    Get the index channel from a frame.
    
    The index channel is typically the first channel and represents
    depth or time.
    """
    if not frame.channels:
        return None
    return frame.channels[0]


def _read_index_only(logical_file, frame):
    """
    Read only the depth index channel using dlisio's low-level read_fdata.

    This avoids loading the entire frame (which may include multi-GB image
    arrays) just to get the depth values.

    Args:
        logical_file: dlisio LogicalFile object (must be inside a load context).
        frame: dlisio Frame object.

    Returns:
        numpy array of depth values, or None on failure.
    """
    try:
        import dlisio.core as core
    except ImportError:
        return None

    try:
        indices = logical_file.fdata_index[frame.fingerprint]
        full_dtype = frame.dtype()
        index_ch = frame.channels[0]

        pre, fmt, post = frame.fmtstrchannel(index_ch)
        idx_np_type = full_dtype[index_ch.name]
        idx_dtype = np.dtype([('FRAMENO', np.int32), (index_ch.name, idx_np_type)])

        def alloc(size):
            return np.empty(shape=size, dtype=idx_dtype)

        data = core.read_fdata(
            pre, "i" + fmt, post,
            logical_file.file, indices, idx_dtype.itemsize,
            alloc, logical_file.error_handler
        )
        return data[index_ch.name]
    except Exception:
        return None


# Common null values in DLIS files
_DLIS_NULL_VALUES = [-9999.0, -9999.25, -999.25, -999.0, 9999.0, 9999.25]


def _replace_null_values(data, null_values=None):
    """
    Replace common null values with NaN.
    
    Args:
        data: numpy array
        null_values: list of values to treat as null (default: common DLIS nulls)
        
    Returns:
        numpy array with nulls replaced by NaN
    """
    if null_values is None:
        null_values = _DLIS_NULL_VALUES
    
    # Make a copy to avoid modifying original
    result = data.astype(float)
    
    for null_val in null_values:
        result[np.isclose(result, null_val, rtol=1e-5)] = np.nan
    
    return result


def _frame_to_curves(frame, index_units=None, convert_index=True,
                     replace_nulls=True, logical_file=None,
                     top=None, bottom=None):
    """
    Convert a DLIS frame to a dictionary of Curve objects.
    
    Args:
        frame: dlisio Frame object
        index_units: Optional units for the index (overrides detected units)
        convert_index: If True, convert index to feet when possible
        replace_nulls: If True, replace common null values (-9999, etc.) with NaN
        logical_file: Optional dlisio LogicalFile, required for depth windowing.
        top: Optional top depth for depth-windowed loading.
        bottom: Optional bottom depth for depth-windowed loading.
        
    Returns:
        dict: Dictionary mapping channel names to Curve objects
    """
    curves = {}
    
    # Filter to 1D channels only (skip image arrays) to avoid loading
    # multi-GB image data into memory when we only need 1D curves.
    channels_1d = [ch for ch in frame.channels if ch.dimension[0] <= 1]
    
    use_windowed = (top is not None and bottom is not None
                    and logical_file is not None)
    
    try:
        if use_windowed:
            data = _read_depth_windowed(logical_file, frame, top, bottom)
        else:
            data = _safe_curves(frame, channels=channels_1d)
    except Exception as e:
        warnings.warn(f"Could not read curves from frame {frame.name}: {e}")
        return curves
    
    if data is None or len(data) == 0:
        return curves
    
    # Get index channel (first channel, typically depth)
    index_channel = _get_index_channel(frame)
    if index_channel is None:
        warnings.warn(f"No index channel found in frame {frame.name}")
        return curves
    
    index_name = index_channel.name
    
    # Extract index values
    try:
        index_values = data[index_name]
    except (KeyError, ValueError):
        # Try using the first field
        first_field = data.dtype.names[1] if len(data.dtype.names) > 1 else data.dtype.names[0]
        index_values = data[first_field]
        index_name = first_field
    
    # Determine index units
    original_units = getattr(index_channel, 'units', None)
    if index_units is None:
        index_units = original_units
    
    # Convert index to standard units (feet) if possible
    if convert_index and index_units:
        index_values, index_units = _convert_index_to_feet(index_values, index_units)
    
    # Create curves for each channel (skip index and FRAMENO)
    for channel in frame.channels[1:]:  # Skip index channel
        ch_name = channel.name
        
        try:
            ch_data = data[ch_name]
        except (KeyError, ValueError):
            continue
        
        # Handle multi-dimensional channels
        if len(ch_data.shape) > 1:
            # For now, skip multi-dimensional channels
            # TODO: Support 2D curves
            continue
        
        # Replace null values with NaN
        if replace_nulls:
            ch_data = _replace_null_values(ch_data)
        
        curve = Curve(
            data=ch_data,
            index=index_values,
            mnemonic=ch_name,
            units=getattr(channel, 'units', None),
            description=getattr(channel, 'long_name', ''),
            index_units=index_units,
        )
        curves[ch_name] = curve
    
    return curves


def _origin_to_location(origin):
    """
    Extract location information from a DLIS origin object.
    
    Args:
        origin: dlisio Origin object
        
    Returns:
        Location: welly Location object
    """
    loc = Location()
    
    if origin is None:
        return loc
    
    # Map DLIS origin attributes to Location attributes
    attr_map = {
        'well_name': 'well',
        'field_name': 'field',
        'company': 'company',
        'country': 'country',
        'province': 'province',
    }
    
    for dlis_attr, loc_attr in attr_map.items():
        value = getattr(origin, dlis_attr, None)
        if value is not None:
            setattr(loc, loc_attr, value)
    
    return loc


def _build_header_from_origin(origin, frame=None, logical_file=None):
    """
    Build a header DataFrame from DLIS origin and frame info.
    
    Args:
        origin: dlisio Origin object
        frame: dlisio Frame object (optional)
        logical_file: dlisio LogicalFile (optional). When provided, uses
            low-level reading to get depth range without loading the full frame.
        
    Returns:
        pd.DataFrame: Header DataFrame compatible with welly Well
    """
    rows = []
    
    # Add well info from origin
    if origin is not None:
        if origin.well_name:
            rows.append({
                'original_mnemonic': 'WELL',
                'mnemonic': 'WELL',
                'unit': '',
                'value': origin.well_name,
                'descr': 'Well name',
                'section': 'Well'
            })
        
        if origin.field_name:
            rows.append({
                'original_mnemonic': 'FLD',
                'mnemonic': 'FLD',
                'unit': '',
                'value': origin.field_name,
                'descr': 'Field name',
                'section': 'Well'
            })
        
        if origin.company:
            rows.append({
                'original_mnemonic': 'COMP',
                'mnemonic': 'COMP',
                'unit': '',
                'value': origin.company,
                'descr': 'Company',
                'section': 'Well'
            })
    
    # Add depth info from frame
    if frame is not None:
        index_ch = _get_index_channel(frame)
        if index_ch is not None:
            try:
                # Try lightweight index-only read first (avoids loading
                # multi-GB image arrays just to get depth range).
                index_values = None
                if logical_file is not None:
                    index_values = _read_index_only(logical_file, frame)

                # Fall back to _safe_curves if low-level read unavailable
                if index_values is None:
                    data = _safe_curves(frame, channels=[index_ch])
                    if data is not None and len(data) > 0:
                        index_values = data[index_ch.name]

                if index_values is not None and len(index_values) > 0:
                    index_unit = getattr(index_ch, 'units', 'm')
                    
                    rows.append({
                        'original_mnemonic': 'STRT',
                        'mnemonic': 'STRT',
                        'unit': index_unit,
                        'value': float(index_values[0]),
                        'descr': 'Start depth',
                        'section': 'Well'
                    })
                    rows.append({
                        'original_mnemonic': 'STOP',
                        'mnemonic': 'STOP',
                        'unit': index_unit,
                        'value': float(index_values[-1]),
                        'descr': 'Stop depth',
                        'section': 'Well'
                    })
                    
                    if frame.spacing:
                        rows.append({
                            'original_mnemonic': 'STEP',
                            'mnemonic': 'STEP',
                            'unit': index_unit,
                            'value': float(frame.spacing),
                            'descr': 'Step',
                            'section': 'Well'
                        })
            except Exception:
                pass
    
    if not rows:
        # Return empty header with correct columns
        return pd.DataFrame(columns=['original_mnemonic', 'mnemonic',
                                     'unit', 'value', 'descr', 'section'])
    
    return pd.DataFrame(rows)


def _frame_to_image_curves(frame, index_units=None, convert_index=True):
    """
    Extract 2D image curves from a DLIS frame.
    
    This function extracts multi-dimensional channels (like FMI, UBI images)
    that are skipped by _frame_to_curves(). It also looks for orientation
    curves (like P1NO) to enable de-rotation.
    
    Args:
        frame: dlisio Frame object
        index_units: Optional units for the index (overrides detected units)
        convert_index: If True, convert index to feet when possible
        
    Returns:
        dict: Dictionary mapping channel names to ImageCurve objects
    """
    from .image import ImageCurve
    
    images = {}
    
    # Get the curve data as a structured numpy array
    try:
        data = frame.curves()
    except Exception as e:
        warnings.warn(f"Could not read curves from frame {frame.name}: {e}")
        return images
    
    if data is None or len(data) == 0:
        return images
    
    # Get index channel (first channel, typically depth)
    index_channel = _get_index_channel(frame)
    if index_channel is None:
        warnings.warn(f"No index channel found in frame {frame.name}")
        return images
    
    index_name = index_channel.name
    
    # Extract index values
    try:
        index_values = data[index_name]
    except (KeyError, ValueError):
        first_field = data.dtype.names[1] if len(data.dtype.names) > 1 else data.dtype.names[0]
        index_values = data[first_field]
        index_name = first_field
    
    # Determine index units
    original_units = getattr(index_channel, 'units', None)
    if index_units is None:
        index_units = original_units
    
    # Convert index to standard units (feet) if possible
    if convert_index and index_units:
        index_values, index_units = _convert_index_to_feet(index_values, index_units)
    
    # Look for orientation curve (pad 1 azimuth) for de-rotation
    # Common names: P1NO, P1NO_FBST, P1NO_FBST_S, P1AZ, PAD1_AZ
    orientation_names = ['P1NO', 'P1NO_FBST', 'P1NO_FBST_S', 'P1AZ', 'PAD1_AZ',
                         'P1_NO', 'RB', 'RB_FBST', 'RB_FBST_S']
    orientation = None
    orientation_name = None
    
    for ch in frame.channels:
        if ch.name in orientation_names or any(n in ch.name for n in ['P1NO', 'P1AZ']):
            try:
                ch_data = data[ch.name]
                if len(ch_data.shape) == 1:  # Must be 1D
                    orientation = ch_data.astype(float)
                    orientation_name = ch.name
                    break
            except (KeyError, ValueError):
                continue
    
    # Create ImageCurve for each 2D channel
    for channel in frame.channels[1:]:  # Skip index channel
        ch_name = channel.name
        
        try:
            ch_data = data[ch_name]
        except (KeyError, ValueError):
            continue
        
        # Only process multi-dimensional channels
        if len(ch_data.shape) != 2:
            continue
        
        # Create ImageCurve with orientation if available
        image = ImageCurve(
            data=ch_data,
            index=index_values,
            mnemonic=ch_name,
            units=getattr(channel, 'units', None),
            index_units=index_units or 'ft',
            description=getattr(channel, 'long_name', ''),
            null_value=-9999.0,
            orientation=orientation,
        )
        images[ch_name] = image
        
        if orientation is not None and len(images) == 1:
            # Only print once
            warnings.warn(
                f"Found orientation curve '{orientation_name}' - "
                f"use image.derotate() to align pads vertically",
                stacklevel=2
            )
    
    return images


def load_images_from_dlis(fname, frame=None, logical_file=0, error_handling='warn'):
    """
    Load image curves from a DLIS file.
    
    This function specifically loads 2D image data (like FMI, UBI) from
    DLIS files. For 1D curves, use Well.from_dlis() instead.
    
    Args:
        fname (str): Path to the DLIS file.
        frame (str): Optional. Name of the frame to load. If None, loads
            the first frame with image data.
        logical_file (int): Optional. Index of the logical file to load.
            Default is 0 (first logical file).
        error_handling (str): Optional. How to handle DLIS parsing errors.
            'warn' (default), 'strict', or 'ignore'.
    
    Returns:
        dict: Dictionary mapping image names to ImageCurve objects.
        
    Example:
        >>> from welly.dlis import load_images_from_dlis
        >>> images = load_images_from_dlis('fmi_data.dlis')
        >>> fmi = images['FMI_DYN']
        >>> fmi.plot()
        >>> fmi.to_pdf('fmi_output.pdf', feet_per_page=100)
    """
    from . import utils
    
    dlis_module, ErrorHandler = _check_dlisio()
    
    fname = utils.to_filename(fname)
    
    # Configure error handling
    if error_handling == 'strict':
        handler = None
    elif error_handling == 'ignore':
        handler = ErrorHandler(
            critical=ErrorHandler.swallow,
            major=ErrorHandler.swallow,
            minor=ErrorHandler.swallow,
        )
    else:
        handler = ErrorHandler(critical=ErrorHandler.swallow)
    
    load_kwargs = {'error_handler': handler} if handler else {}
    
    with dlis_module.load(fname, **load_kwargs) as files:
        if logical_file >= len(files):
            raise ValueError(
                f"Logical file index {logical_file} out of range. "
                f"File contains {len(files)} logical file(s)."
            )
        
        logical_f = files[logical_file]
        
        try:
            frames = logical_f.frames
        except Exception as e:
            raise ValueError(f"Could not read frames: {e}")
        
        if not frames:
            raise ValueError("No frames found in the logical file.")
        
        # Find the requested frame
        target_frame = None
        if frame is not None:
            for fr in frames:
                if fr.name == frame:
                    target_frame = fr
                    break
            if target_frame is None:
                available = [fr.name for fr in frames]
                raise ValueError(
                    f"Frame '{frame}' not found. "
                    f"Available frames: {available}"
                )
        else:
            # Use first frame with 2D data (check channel metadata, no data loading)
            for fr in frames:
                try:
                    for ch in fr.channels[1:]:
                        if ch.dimension[0] > 1:
                            target_frame = fr
                            break
                    if target_frame is not None:
                        break
                except Exception:
                    continue
            
            if target_frame is None:
                raise ValueError("No frames with image data found.")
        
        return _frame_to_image_curves(target_frame)


def describe_image_channels(fname, logical_file=0, error_handling='warn'):
    """
    Describe image channels in a DLIS file using only metadata.

    No image array data is loaded — only the depth index is read to
    determine the depth range. This is safe to call on multi-GB files.

    Args:
        fname (str): Path to the DLIS file.
        logical_file (int): Index of the logical file. Default 0.
        error_handling (str): 'warn' (default), 'strict', or 'ignore'.

    Returns:
        list: List of dicts, one per image channel, with keys:
            - name: channel name
            - n_azimuths: number of azimuthal samples (e.g. 128)
            - units: channel units
            - frame: name of the containing frame
            - start: first depth value
            - stop: last depth value
            - n_samples: number of depth samples
            - index_units: units of the depth index

    Example:
        >>> from welly.dlis import describe_image_channels
        >>> channels = describe_image_channels('fmi_data.dlis')
        >>> for ch in channels:
        ...     print(f"{ch['name']}: {ch['n_azimuths']} azimuths, "
        ...           f"{ch['start']:.1f}-{ch['stop']:.1f} {ch['index_units']}")
    """
    dlis_module, ErrorHandler = _check_dlisio()

    if error_handling == 'strict':
        handler = None
    elif error_handling == 'ignore':
        handler = ErrorHandler(
            critical=ErrorHandler.swallow,
            major=ErrorHandler.swallow,
            minor=ErrorHandler.swallow,
        )
    else:
        handler = ErrorHandler(critical=ErrorHandler.swallow)

    load_kwargs = {'error_handler': handler} if handler else {}

    channels = []

    with dlis_module.load(fname, **load_kwargs) as files:
        if logical_file >= len(files):
            raise ValueError(
                f"Logical file index {logical_file} out of range. "
                f"File contains {len(files)} logical file(s)."
            )

        logical_f = files[logical_file]

        for frame in logical_f.frames:
            image_channels = [
                ch for ch in frame.channels[1:]
                if ch.dimension[0] > 1
            ]
            if not image_channels:
                continue

            # Load only the index channel for depth range
            index_ch = frame.channels[0]
            index_data = None
            try:
                data = _safe_curves(frame, channels=[index_ch])
                if data is not None and len(data) > 0:
                    index_data = data[index_ch.name]
            except Exception:
                pass

            for ch in image_channels:
                info = {
                    'name': ch.name,
                    'n_azimuths': ch.dimension[0],
                    'units': getattr(ch, 'units', None),
                    'description': getattr(ch, 'long_name', ''),
                    'frame': frame.name,
                    'start': float(index_data[0]) if index_data is not None else None,
                    'stop': float(index_data[-1]) if index_data is not None else None,
                    'n_samples': len(index_data) if index_data is not None else None,
                    'index_units': getattr(index_ch, 'units', None),
                }
                channels.append(info)

    return channels


def _read_depth_windowed(logical_file, frame, top, bottom):
    """
    Two-pass depth-windowed reading using dlisio's low-level read_fdata.

    Pass 1: Read only FRAMENO + depth index for all rows (tiny memory).
    Pass 2: Read full row data for only the rows in the depth window.

    This avoids loading the entire frame into memory, which is critical
    for large DLIS files (multi-GB) with high-resolution image channels.

    Args:
        logical_file: dlisio LogicalFile object (must be inside a load context).
        frame: dlisio Frame object containing the target data.
        top (float): Top depth of the window (in file's native units).
        bottom (float): Bottom depth of the window (in file's native units).

    Returns:
        numpy structured array: Full row data for the depth-windowed rows,
            with the same dtype as frame.dtype().

    Raises:
        ValueError: If no rows fall within the depth window.
    """
    import dlisio.core as core

    indices = logical_file.fdata_index[frame.fingerprint]
    full_dtype = frame.dtype()
    full_fmtstr = frame.fmtstr()
    index_ch = frame.channels[0]

    # --- Pass 1: read depth index only ---
    pre, fmt, post = frame.fmtstrchannel(index_ch)
    idx_np_type = full_dtype[index_ch.name]
    idx_dtype = np.dtype([('FRAMENO', np.int32), (index_ch.name, idx_np_type)])

    def alloc_idx(size):
        return np.empty(shape=size, dtype=idx_dtype)

    idx_data = core.read_fdata(
        pre, "i" + fmt, post,
        logical_file.file, indices, idx_dtype.itemsize,
        alloc_idx, logical_file.error_handler
    )
    depths = idx_data[index_ch.name]

    # Handle both ascending and descending depth order
    d_top, d_bottom = min(top, bottom), max(top, bottom)
    mask = (depths >= d_top) & (depths <= d_bottom)
    row_idx = np.where(mask)[0]

    del idx_data  # free the index-only array

    if len(row_idx) == 0:
        raise ValueError(
            f"No data in depth range {top}-{bottom}. "
            f"File depth range: {float(depths[0]):.1f} to {float(depths[-1]):.1f}"
        )

    # --- Pass 2: read full row data for subset ---
    subset_offsets = [indices[i] for i in row_idx]

    def alloc_full(size):
        return np.empty(shape=size, dtype=full_dtype)

    data = core.read_fdata(
        "", full_fmtstr, "",
        logical_file.file, subset_offsets, full_dtype.itemsize,
        alloc_full, logical_file.error_handler
    )

    return data


def load_single_image(fname, channel_name, logical_file=0, error_handling='warn',
                      top=None, bottom=None):
    """
    Load a single image channel from a DLIS file.

    Memory-efficient alternative to load_images_from_dlis() when you
    only need one channel. Loads only the depth index, the requested
    channel, and any orientation curve — not the entire frame.

    When ``top`` and ``bottom`` are provided, uses a two-pass approach
    that reads only the rows within the depth window, reducing memory
    from gigabytes to megabytes for large files.

    Args:
        fname (str): Path to the DLIS file.
        channel_name (str): Name of the image channel to load.
        logical_file (int): Index of the logical file. Default 0.
        error_handling (str): 'warn' (default), 'strict', or 'ignore'.
        top (float): Optional. Top depth of the window to load.
            When provided with ``bottom``, enables depth-windowed reading.
        bottom (float): Optional. Bottom depth of the window to load.
            When provided with ``top``, enables depth-windowed reading.

    Returns:
        ImageCurve: The requested image channel.

    Raises:
        ValueError: If the channel is not found or is not 2D.

    Example:
        >>> from welly.dlis import load_single_image
        >>> fmi = load_single_image('fmi_data.dlis', 'FMI_DYN')
        >>> fmi.plot()

        Load only a depth window (memory-efficient for large files):

        >>> fmi = load_single_image('big_file.dlis', 'FMI_DYN',
        ...                         top=5000, bottom=5100)
    """
    from .image import ImageCurve
    from . import utils

    dlis_module, ErrorHandler = _check_dlisio()

    fname = utils.to_filename(fname)

    if error_handling == 'strict':
        handler = None
    elif error_handling == 'ignore':
        handler = ErrorHandler(
            critical=ErrorHandler.swallow,
            major=ErrorHandler.swallow,
            minor=ErrorHandler.swallow,
        )
    else:
        handler = ErrorHandler(critical=ErrorHandler.swallow)

    load_kwargs = {'error_handler': handler} if handler else {}

    orientation_names = [
        'P1NO', 'P1NO_FBST', 'P1NO_FBST_S', 'P1AZ', 'PAD1_AZ',
        'P1_NO', 'RB', 'RB_FBST', 'RB_FBST_S',
    ]

    use_windowed = top is not None and bottom is not None

    with dlis_module.load(fname, **load_kwargs) as files:
        if logical_file >= len(files):
            raise ValueError(
                f"Logical file index {logical_file} out of range. "
                f"File contains {len(files)} logical file(s)."
            )

        logical_f = files[logical_file]

        # Search all frames for the requested channel
        for frame in logical_f.frames:
            target_ch = None
            orientation_ch = None
            index_ch = frame.channels[0]

            for ch in frame.channels[1:]:
                if ch.name == channel_name:
                    if ch.dimension[0] <= 1:
                        raise ValueError(
                            f"Channel '{channel_name}' is 1D, not an image. "
                            f"Use Well.from_dlis() for 1D curves."
                        )
                    target_ch = ch
                if orientation_ch is None and (
                    ch.name in orientation_names or any(
                        n in ch.name for n in ['P1NO', 'P1AZ']
                    )
                ):
                    orientation_ch = ch

            if target_ch is None:
                continue

            # --- Load data ---
            if use_windowed:
                # Two-pass depth-windowed read (memory-efficient)
                data = _read_depth_windowed(logical_f, frame, top, bottom)
            else:
                # Original path: try selective channels, fall back to full
                channels_to_load = [index_ch, target_ch]
                if orientation_ch is not None:
                    channels_to_load.append(orientation_ch)
                data = _safe_curves(frame, channels=channels_to_load)

            if data is None or len(data) == 0:
                raise ValueError(f"No data returned for channel '{channel_name}'")

            # Extract index
            index_values = data[index_ch.name]
            index_units = getattr(index_ch, 'units', None)
            if index_units:
                index_values, index_units = _convert_index_to_feet(
                    index_values, index_units
                )

            # Extract orientation
            orientation = None
            if orientation_ch is not None:
                try:
                    orientation = data[orientation_ch.name].astype(float)
                except (KeyError, ValueError):
                    pass

            image = ImageCurve(
                data=data[target_ch.name],
                index=index_values,
                mnemonic=target_ch.name,
                units=getattr(target_ch, 'units', None),
                index_units=index_units or 'ft',
                description=getattr(target_ch, 'long_name', ''),
                null_value=-9999.0,
                orientation=orientation,
            )
            return image

    raise ValueError(
        f"Channel '{channel_name}' not found in any frame."
    )
