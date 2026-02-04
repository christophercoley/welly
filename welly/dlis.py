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
                    
                    # Try to get depth range
                    try:
                        data = frame.curves()
                        if data is not None and len(data) > 0:
                            index_ch = frame.channels[0]
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


def _get_index_channel(frame):
    """
    Get the index channel from a frame.
    
    The index channel is typically the first channel and represents
    depth or time.
    """
    if not frame.channels:
        return None
    return frame.channels[0]


def _frame_to_curves(frame, index_units=None, convert_index=True):
    """
    Convert a DLIS frame to a dictionary of Curve objects.
    
    Args:
        frame: dlisio Frame object
        index_units: Optional units for the index (overrides detected units)
        convert_index: If True, convert index to feet when possible
        
    Returns:
        dict: Dictionary mapping channel names to Curve objects
    """
    curves = {}
    
    # Get the curve data as a structured numpy array
    try:
        data = frame.curves()
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


def _build_header_from_origin(origin, frame=None):
    """
    Build a header DataFrame from DLIS origin and frame info.
    
    Args:
        origin: dlisio Origin object
        frame: dlisio Frame object (optional)
        
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
                data = frame.curves()
                if data is not None and len(data) > 0:
                    index_name = index_ch.name
                    index_values = data[index_name]
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
