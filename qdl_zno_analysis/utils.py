"""
Module containing various utility functions for this package.
"""
import dataclasses
import re
import types
import typing
from dataclasses import dataclass, fields
from inspect import getfullargspec
from operator import attrgetter
from typing import List, Dict, Type, Iterable, Union

import numpy as np
import pint
import pint_xarray
import scipy.interpolate as spi
import xarray as xr
from xarray.core.types import ExtendOptions
from xarray.plot import dataarray_plot
from xarray.plot.utils import _determine_cmap_params, _add_colorbar

try:  # visualization dependencies
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except ModuleNotFoundError:
    pass

from pint_xarray.conversion import extract_units, strip_units, attach_units

from qdl_zno_analysis import ureg, Qty
from qdl_zno_analysis.constants import default_units
from qdl_zno_analysis.errors import MethodInputError
from qdl_zno_analysis.typevars import EnhancedNumeric, XRObject


def is_unit_valid(unit_str: str) -> bool:
    """
    Check if a unit string is a valid unit recognized by the `pint` library.

    Parameters
    ----------
    unit_str : str
        A string representing a unit, e.g. 'm', 'km/h', 'kg*m/s^2', etc.

    Returns
    -------
    bool
        True if the unit string is valid, False otherwise.

    Examples
    --------
    >>> is_unit_valid('m')
    True

    >>> is_unit_valid('km/h')
    True

    >>> is_unit_valid('kg*m/s^2')
    True

    >>> is_unit_valid('potato')
    False

    """
    try:
        ureg.Unit(unit_str)
        return True
    except pint.errors.UndefinedUnitError:
        return False


def get_class_arg_type_hints(cls, arg_name: str) -> Type | None:
    """
    Get the type hint of a class method argument.

    Parameters
    ----------
    cls : class
        The class containing the method.
    arg_name : str
        The name of the method argument.

    Returns
    -------
    Type | None
        The type hint of the method argument, or None if it is not defined.

    Examples
    --------
    >>> class Test:
    ...     a: int
    ...     b: str | bool
    ...     c = 'potato'
    >>> get_class_arg_type_hints(Test, 'a')
    <class 'int'>

    >>> get_class_arg_type_hints(Test, 'b')
    (<class 'str'>, <class 'bool'>)

    >>> get_class_arg_type_hints(Test, 'c')


    >>> get_class_arg_type_hints(Test, 'd')


    """
    try:
        arg_type = cls.__annotations__[arg_name]
        if typing.get_origin(arg_type) in [typing.Union, types.UnionType]:
            arg_type = typing.get_args(arg_type)
        return arg_type
    except Exception:
        return None


def to_qty_force_units(value: EnhancedNumeric | np.ndarray[int | float] | list[int | float],
                       physical_type: str = "dimensionless", context=None, **context_kwargs) -> Qty:
    """
    Converts the input value a pint.Quantity object with the default units of the provided physical type
    as defined in the `default_units` dictionary. This method, unlike `to_qty` does enforce the default units, even on
    pint.Quantity inputs.


    Parameters
    ----------
    value : int | float | pint.Quantity | np.ndarray[int | float] | list[int | float]
        The object to convert.
    physical_type : str, optional
        The physical type of the value, by default "dimensionless".
    context : str, optional
        A context string for unit conversions (e.g. 'spectroscopy'), by default None.
    context_kwargs: Any, optional
        Additional keyword arguments for unit conversion contexts (e.g. n=1.33 for refractive index in the
        'spectroscopy' context).

    Returns
    -------
    pint.Quantity
        The converted object.

    Raises
    ------
    pint.errors.DimensionalityError
        If the units of the input value are not compatible with the units of the physical type.

    Examples
    --------
    >>> to_qty_force_units(1, 'length')  # converts to quantity with default length units
    <Quantity(1, 'nanometer')>

    >>> to_qty_force_units(1, 'frequency')  # converts to quantity with default frequency units
    <Quantity(1, 'gigahertz')>

    >>> to_qty_force_units(Qty(1, 'um'), 'length')  # converts the quantity object to default length units.
    <Quantity(1000.0, 'nanometer')>

    >>> to_qty_force_units(Qty(1, 'nm'), 'frequency')  # Error: the physical type units are not compatible with the quantity units
    Traceback (most recent call last):
        ...
    pint.errors.DimensionalityError: Cannot convert from 'nanometer' ([length]) to 'gigahertz' (1 / [time])

    >>> to_qty_force_units(Qty(1, 'nm'), 'frequency', 'spectroscopy')  # in the spectroscopy context, nm and frequency are compatible!
    <Quantity(2.99792458e+08, 'gigahertz')>

    >>> to_qty_force_units(Qty(1, 'nm'), 'frequency', 'sp', n=1.33)  # converts with n=1.33 in the spectroscopy context
    <Quantity(2.25407863e+08, 'gigahertz')>
    """

    if value is None:
        return None

    if context is not None:
        ureg.enable_contexts(context, **context_kwargs)

    units: str = default_units[physical_type]['main']
    if isinstance(value, int | float | List | np.ndarray):
        rv = Qty(value, units)
    elif isinstance(value, xr.DataArray):
        rv = value.pint.to(units)
    else:
        rv = value.to(units)

    ureg.disable_contexts()
    return rv


def to_qty(value: EnhancedNumeric | np.ndarray[int | float] | list[int | float],
           physical_type: str = "dimensionless") -> Qty:
    """
    Assures the input is a pint.Quantity object. If not, it converts the input value to pint.Quantity object with the
    provided physical type. The available physical types are defines in the `default_units` dictionary.

    Parameters
    ----------
    value : int | float | pint.Quantity | np.ndarray[int | float] | list[int | float]
        The object to convert.
    physical_type : str, optional
        The physical type of the value, by default "dimensionless".

    Returns
    -------
    pint.Quantity
        The converted object.

    Examples
    --------
    >>> to_qty(1, 'length')  # converts to quantity with default length units
    <Quantity(1, 'nanometer')>

    >>> to_qty(1, 'frequency')  # converts to quantity with default frequency units
    <Quantity(1, 'gigahertz')>

    >>> to_qty(Qty(1, 'nm'), 'length')  # nothing happens - the unit does not change to the default units
    <Quantity(1, 'nanometer')>

    >>> to_qty(Qty(1, 'nm'), 'frequency')  # the physical type is not enforced!
    <Quantity(1, 'nanometer')>

    """
    if isinstance(value, Qty):
        return value
    elif isinstance(value, xr.DataArray):
        if value.pint.units is not None:
            return value

    return to_qty_force_units(value, physical_type)


def varname_to_title_string(varname: str, ignore: str = '') -> str:
    """
    Convert a variable name to a title-type string.

    Parameters
    ----------
    varname : str
        The variable name to convert.
    ignore : str, optional
        A string containing characters to ignore during the conversion, by default ''.

    Returns
    -------
    str
        The converted variable name.

    Examples
    --------
    >>> varname_to_title_string('green_onions.rule')
    'Green Onions Rule'

    >>> varname_to_title_string('potatoes!', '!')
    'Potatoes!'
    """
    return re.sub(r'[^a-zA-Z0-9' + ignore + r']+|^(?=\d)', ' ', varname).title()


def str_to_valid_varname(string: str, ignore: str = '') -> str:
    """
    Convert a string to a valid Python variable name.

    Parameters
    ----------
    string : str
        The string to convert.
    ignore : str, optional
        A string containing characters to ignore during the conversion, by default ''.

    Returns
    -------
    str
        The converted string.

    Examples
    --------
    >>> str_to_valid_varname('a b c')
    'a_b_c'

    >>> str_to_valid_varname('a b.c', '.')
    'a_b.c'

    >>> str_to_valid_varname('a b,c.d', '.,')
    'a_b,c.d'
    """
    return re.sub(r'[^\w' + ignore + r']+|^(?=\d)', '_', string).lower()


def normalize_dict(dictionary: dict, parent_key='', separator='.') -> Dict:
    """
    Normalize a dictionary by converting its keys to valid Python variable names and flattening it.
    Each key in the flattened dictionary is a result of addition of all keys that resulted in it from the
    original dictionary.

    For example, if the original dictionary is `{'a': {'b': 1, 'c': 2}, 'd': 3}`,
    the normalized dictionary is `{'a.b': 1, 'a.c': 2, 'd': 3}`, where '.' is the default separator.

    Parameters
    ----------
    dictionary : dict
        The dictionary to normalize.
    parent_key : str, optional
        The prefix to add to the keys of the dictionary, by default ''.
    separator : str, optional
        The string to use as a separator between the prefix and the keys, by default '.'.

    Returns
    -------
    Dict
        The normalized dictionary.

    Examples
    --------
    >>> normalize_dict({'a': {'b': 1, 'c': 2}, 'd': 3})
    {'a.b': 1, 'a.c': 2, 'd': 3}

    >>> normalize_dict({'a': {'b': 1, 'c': 2}, 'd': 3}, 'prefix')
    {'prefix.a.b': 1, 'prefix.a.c': 2, 'prefix.d': 3}

    >>> normalize_dict({'a': {'b': 1, 'c': 2}, 'd': 3}, 'prefix', '/')
    {'prefix/a/b': 1, 'prefix/a/c': 2, 'prefix/d': 3}
    """
    normalized_dict = {}
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        new_key = str_to_valid_varname(new_key, separator)
        if isinstance(value, dict):
            normalized_dict.update(normalize_dict(value, new_key, separator))
        else:
            normalized_dict[new_key] = value
    return normalized_dict


def find_changing_values_in_list_of_dict(list_of_dicts: List[Dict], reverse_result=False) -> Dict:
    """
    Find the keys in a list of dictionaries whose values change between the dictionaries.

    Parameters
    ----------
    list_of_dicts: List[Dict]
        The list of dictionaries to analyze.
    reverse_result: bool, optional
        If True, returns only unchanged values. Defaults to False.

    Returns
    -------
    Dict
        A dictionary containing the keys whose values change (or not change), and their corresponding value(s).

    Examples
    --------
    >>> dict_list = [{'a': value, 'b': 'potato', 'c': [100, 200]} for value in range(3)]
    >>> find_changing_values_in_list_of_dict(dict_list)
    {'a': [0, 1, 2]}

    >>> find_changing_values_in_list_of_dict(dict_list, True)
    {'b': 'potato', 'c': [100, 200]}
    """

    result_dict = {}
    first_dict = list_of_dicts[0]

    for key in list_of_dicts[0].keys():

        first_value = getattr(first_dict[key], 'magnitude', first_dict[key])  # if Qty get magnitude, else get value.
        first_value = tuple(first_value) if len(np.shape(first_value)) else first_value  # make it a tuple.
        value_list = [first_value]

        for d in list_of_dicts[1:]:
            test_element = getattr(d[key], 'magnitude', d[key])  # if Qty get magnitude, else get value.
            test_element = tuple(test_element) if len(np.shape(test_element)) else test_element

            if test_element not in value_list:
                value_list = [d[key] for d in list_of_dicts]
                break  # if even one element is different, stop looking.

        if len(value_list) == 1 and reverse_result:
            # not the first element of the value list, since it might not be the original value.
            result_dict[key] = first_dict[key]
        elif len(value_list) > 1 and not reverse_result:
            if all(isinstance(v, Qty) for v in value_list):
                # can not make n-th dim Quantity arrays, but can make 1D Qty arrays.
                if not any(isinstance(v.m, np.ndarray) for v in value_list):
                    value_list = Qty.from_list(value_list)
            result_dict[key] = value_list

    return result_dict


def interpolated_integration(start: EnhancedNumeric, end: EnhancedNumeric,
                             x_data: np.ndarray | Qty, y_data: np.ndarray | Qty) -> EnhancedNumeric:
    """
    Performs an interpolated numerical integration of an array of `y_data` with correlated `x_data`, from the starting
    x-point until the end x-point.

    Parameters
    ----------
    start : int | float | pint.Quantity
        The start point of the integration interval.
    end : int | float | pint.Quantity
        The end point of the integration interval.
    x_data : np.ndarray | pint.Quantity
        The array of points at which the function was sampled.
    y_data : np.ndarray | pint.Quantity
        The array of function values at the sampling points.

    Returns
    -------
    int | float | pint.Quantity
        The result of the interpolated integration of the function.

    Raises
    ------
    ValueError
        If start or end are outside the range of `x_data`.
    ValueError
        If the length of `x_data` and `y_data` do not match.

    Examples
    --------
    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([0, 1, 0, -1])
    >>> interpolated_integration(0, 3, x, y)
    0.5

    >>> interpolated_integration(0, 1, x, y)
    0.5

    >>> interpolated_integration(1, 2, x, y)
    0.5

    >>> interpolated_integration(2.5, 3, x, y)
    -0.375

    >>> interpolated_integration(0, 4, x, y)  # end outside x_data
    Traceback (most recent call last):
        ...
    ValueError: start (0) and end (4) values must be within the provided x-data range [0, 3].

    >>> interpolated_integration(-1, 3, x, y)  # start outside x_data
    Traceback (most recent call last):
        ...
    ValueError: start (-1) and end (3) values must be within the provided x-data range [0, 3].

    >>> interpolated_integration(0, 3, x, np.array([0, 1, 0]))  # x_data and y_data length mismatch
    Traceback (most recent call last):
        ...
    ValueError: x_data (len=4) and y_data (len=3) must have the same length.

    """

    if len(x_data) != len(y_data):
        raise ValueError(f"x_data (len={len(x_data)}) and y_data (len={len(y_data)}) must have the same length.")

    sorted_idx = x_data.argsort()
    x_data = x_data[sorted_idx]
    y_data = y_data[sorted_idx]

    x_interp = x_data[(x_data > start) & (x_data < end)]
    x_interp = np.append(start, x_interp)
    x_interp = np.append(x_interp, end)
    y_interp = np.interp(x_interp, x_data, y_data, left=np.nan, right=np.nan)
    if np.isnan(y_interp).any():
        raise ValueError(f'start ({start}) and end ({end}) values must be within the provided x-data range'
                         f' [{x_data[0]}, {x_data[-1]}].')
    return np.trapz(y_interp, x_interp)


def find_nearest(array: Qty | np.ndarray | list, value: EnhancedNumeric) -> EnhancedNumeric:
    """
    Returns the value in the input array that is closest in value to the input `value`.

    Parameters
    ----------
    array : pint.Quantity | np.ndarray | list
        The input array to search for the closest value.
    value : EnhancedNumeric
        The value to search for in the input array.

    Returns
    -------
    int | float | pint.Quantity
        The value in the input array that is closest in value to `value`.

    Examples
    --------
    >>> find_nearest([1, 3, 5, 7, 9], 3.5)
    3

    >>> find_nearest(Qty([1, 3, 5, 7, 9], 'nm'), Qty(3.5, 'nm'))
    <Quantity(3, 'nanometer')>

    >>> find_nearest(Qty([1, 3, 5, 7, 9], 'nm'), Qty(4, 'nm'))
    <Quantity(3, 'nanometer')>
    """

    array = np.asarray(array) if not isinstance(array, Qty) else array
    idx = (np.abs(array - value)).argmin()
    return array[idx]


@dataclass(repr=False)
class Dataclass:
    """
    A dataclass that represents an object with attributes that can be converted to dictionaries and string
    representations.
    """

    def to_dict(self):
        """
        Returns a dictionary representation of the `Dataclass` object.
        If an attribute has a value that is an instance of `Dataclass`, its `to_dict()` method is called recursively.
        Utilizes the `dataclasses._asdict_inner` method to cover inner values to dict as well.

        Examples
        --------
        >>> from dataclasses import dataclass
        >>> @dataclass(repr=False)
        ... class Potato(Dataclass):
        ...     a: int
        ...     b: int
        ...     c: int | None | Dataclass = None
        >>> potato = Potato(1, 2)
        >>> potato.to_dict()
        {'a': 1, 'b': 2, 'c': None}
        >>> potato2 = Potato(1, 2, Potato(3, 4))
        >>> potato2.to_dict()
        {'a': 1, 'b': 2, 'c': {'a': 3, 'b': 4, 'c': None}}

        """
        result = []
        for f in fields(self):
            if '_attr_physical_type_dict' not in f.name:
                value = getattr(self, f.name)
                value = value.to_dict() if isinstance(value, Dataclass) else dataclasses._asdict_inner(value, dict)
                result.append((f.name, value))
        return dict(result)

    def to_normalized_dict(self, separator='.'):
        """
        Returns a normalized dictionary representation of the `Info` object.
        The argument `separator` is used to separate keys of nested dictionaries.

        Examples
        --------
        >>> from dataclasses import dataclass
        >>> @dataclass(repr=False)
        ... class Potato(Dataclass):
        ...     a: int
        ...     b: int
        ...     c: int | None | Dataclass = None
        >>> potato = Potato(1, 2)
        >>> potato.to_normalized_dict()
        {'a': 1, 'b': 2, 'c': None}
        >>> potato2 = Potato(1, 2, Potato(3, 4))
        >>> potato2.to_normalized_dict()
        {'a': 1, 'b': 2, 'c.a': 3, 'c.b': 4, 'c.c': None}

        """
        return normalize_dict(self.to_dict(), separator=separator)

    def __str__(self):
        """
        The string conversion of the `Dataclass` object. It deliberately omits all None values so that the print is
        not crowded.

        Examples
        --------
        >>> from dataclasses import dataclass
        >>> @dataclass(repr=False)
        ... class Potato(Dataclass):
        ...     a: int
        ...     b: int
        ...     c: int | None | Dataclass = None
        >>> potato = Potato(1, 2)
        >>> str(potato)  # None values are omitted
        'Potato(a=1, b=2)'
        >>> potato2 = Potato(1, 2, Potato(3, 4))
        >>> str(potato2)
        'Potato(a=1, b=2, c=Potato(a=3, b=4))'

        """
        # https://stackoverflow.com/questions/72161257/exclude-default-fields-from-python-dataclass-repr
        not_none_fields = ((f.name, attrgetter(f.name)(self))
                           for f in fields(self) if attrgetter(f.name)(self) is not None and f.repr)

        not_none_fields_repr = ", ".join(f"{name}={value!r}" for name, value in not_none_fields)
        return f"{self.__class__.__name__}({not_none_fields_repr})"

    def __repr__(self):
        """ Returns the string representation of the `Dataclass` object. """
        return str(self)


def convert_coord_to_dim(coord: xr.DataArray, coord_data_to_convert=None):
    coord_units = extract_units(coord)
    coord_stripped = strip_units(coord)

    dim_name = coord.dims[0]
    dim = coord[dim_name]
    dim_stripped = strip_units(dim)

    conversion_interpolation_function = spi.interp1d(coord_stripped, dim_stripped, 'cubic')

    if coord_data_to_convert is None:
        coord_data_to_convert = coord_stripped.data
    elif isinstance(coord_data_to_convert, Qty):
        coord_data_to_convert = coord_data_to_convert.m_as(coord_units)

    result_stripped_data = conversion_interpolation_function(coord_data_to_convert)

    result_stripped = xr.DataArray(result_stripped_data, coords=coord.coords, name=dim_name)
    return attach_units(result_stripped, extract_units(coord))


def integrate_xarray(
        xarray_data: XRObject,
        start: EnhancedNumeric | None = None,
        end: EnhancedNumeric | None = None,
        coord: str | None = None,
        var: str | None = None,
) -> XRObject:
    """
    Integrates the area under the curve of the data array/set within a given range.

    Parameters
    ----------
    xarray_data : xr.Dataset | xr.DataArray
        The dataset or data array to integrate. If you pass a xr.DataArray, `var` will be ignored.
    start : int | float | pint.Quantity | None, optional
        The start of the integration range. Defaults to the first element of the coordinate array.
        Must be within the range of the coordinate.
    end : int | float | pint.Quantity | None, optional
        The end of the integration range. Defaults to the last element of the coordinate array.
        Must be within the range of the range of the coordinate.
    coord : str | None, optional
        The data coordinate that will be used as the x-axis of the integration.
        Defaults to the last (deepest level) dimension of the dataset.
    var : str | None, optional
        The data variable that will be used as the y-axis of the integration.
        Defaults to the entire dataset.

    Returns
    -------
    xr.Dataset | xr.DataArray
        The reduced data array/set of the integrated array values.

    Raises
    ------
    MethodInputError
        If the specified coordinate or variable is not found in the data.

    Notes
    -----
    If the start or end points are not values in the coordinate data array (x-axis),
    this method uses cubic interpolation to find the y-axis values at the given start or end points.
    """

    if coord is None:
        # Set coord to the last axis (deepest level) if not provided
        coord = list(xarray_data.dims.keys())[-1]
    elif coord not in xarray_data.coords.keys():
        raise MethodInputError('coord', coord, list(xarray_data.coords.keys()), 'integrate_in_region')

    if var is None or isinstance(xarray_data, xr.DataArray):
        # Set the target data to the entire dataset
        data = xarray_data.copy(deep=True)
    elif var in xarray_data.data_vars.keys():
        # Set the target data to the specific data array
        data = xarray_data[var].copy(deep=True)
    else:
        raise MethodInputError('var', var, list(xarray_data.data_vars.keys()), 'integrate_in_region')

    data = data.sortby(coord)  # Sort data array/set by the coordinate of interest
    coord_data_array = data[coord].pint.dequantify()  # Get all coordinate data

    # Get starting point
    if start is None:
        start = coord_data_array[0]
    if isinstance(start, Qty):
        start = start.m_as(data[coord].data.u)  # convert to a value without units

    # Get ending point
    if end is None:
        end = coord_data_array[-1]
    if isinstance(end, Qty):
        end = end.m_as(data[coord].data.u)  # convert to a value without units

    # get coordinate data
    coord_data = data[coord].where(((coord_data_array >= start) & (coord_data_array <= end)), drop=True).data

    # # Add starting and ending points to the
    if start != coord_data[0]:
        coord_data = np.append(start, coord_data)
    if end != coord_data[-1]:
        coord_data = np.append(coord_data, end)

    dim_array = convert_coord_to_dim(xarray_data[coord], coord_data)
    data_for_integration = data.interp({dim_array.name: dim_array.data}, method='linear', assume_sorted=True)
    integrated_data = data_for_integration.integrate(coord)

    return integrated_data


def get_normalized_xarray(
        xarray_data: XRObject,
        norm_axis_val: EnhancedNumeric | xr.DataArray | np.ndarray | None = None,
        norm_axis: str | None = None,
        norm_var: str | None = None,
        mode='nearest',
        subtract_min=True
) -> XRObject:
    """
    Get normalized data based on specified parameters.

    Parameters
    ----------
    xarray_data : xr.Dataset | xr.DataArray
        The dataset or data array to integrate. If you pass a xr.DataArray, `var` will be ignored.
    norm_axis_val : int | float | Qty | xr.DataArray | np.ndarray | None, optional
        The value used for normalization along the `norm_axis`.
        If None, the maximum value of the normalization axis is used.
    norm_axis : str | None, optional
        The axis used for normalization. If None, the last axis (deepest level) is used.
    norm_var : str | None, optional
        The variable used for normalization. If None, the first variable in the dataset is used.
    mode : {'nearest', 'linear', 'quadratic', 'cubic'}, optional
        The interpolation mode for finding norm_axis values.
        'nearest' - Find the nearest `norm_axis` to the `norm_axis_val` .
        'linear' - Perform linear interpolation between adjacent `norm_axis` values.
        'quadratic' - Perform quadratic interpolation between adjacent `norm_axis` values.
        'cubic' - Perform cubic interpolation between adjacent `norm_axis` values.
        Default is 'nearest'.
    subtract_min : bool, optional
        If True, subtract the minimum values from the data before normalization.

    Returns
    -------
    xarray.Dataset
        The normalized data.

    Raises
    ------
    ValueError
        If `norm_axis` or `norm_var` is not found in the data.
    MethodInputError
        If the mode is not one of the allowed modes.
    """
    if norm_axis is None:
        # Set norm_axis to the last axis (deepest level) if not provided
        norm_axis = list(xarray_data.dims.keys())[-1]
    elif norm_axis not in xarray_data.coords.keys():
        raise MethodInputError('norm_axis', norm_axis, list(xarray_data.coords.keys()), 'get_normalized_data')

    if norm_axis not in xarray_data.dims.keys():
        # If norm_axis is not a dimension but a coordinate,
        # assume it's a coordinate with only one dimension
        norm_dim = xarray_data[norm_axis].dims[0]
    else:
        norm_dim = norm_axis

    if norm_var is None:
        # Set norm_var to the first variable if not provided
        norm_var = list(xarray_data.data_vars.keys())[0]
    elif norm_var not in xarray_data.data_vars.keys():
        raise MethodInputError('norm_var', norm_var, list(xarray_data.data_vars.keys()), 'get_normalized_data')

    allowed_modes = ['nearest', 'linear', 'quadratic', 'cubic']
    if mode not in allowed_modes:
        raise MethodInputError('mode', mode, allowed_modes, 'get_normalized_data')

    data = xarray_data.copy(deep=True)
    if subtract_min:
        # Subtract the minimum values from the data
        min_vals = data.min(norm_dim)
        data = data - min_vals

    if norm_axis_val is None:
        # Use the maximum value along the norm_axis if norm_axis_val is not provided
        norm_axis_val = data.argmax(norm_dim)[norm_var]
    elif not isinstance(norm_axis_val, xr.DataArray):
        # Create a DataArray with the same shape structure as data using norm_axis_val,
        # with the norm_axis having a length of 1.
        extra_dims = list(xarray_data.dims.keys())
        extra_dims.remove(norm_dim)

        norm_axis_val_shape = np.shape(norm_axis_val)
        norm_axis_val_dims = extra_dims[::-1][:len(norm_axis_val_shape)]  # dims other than the norm axis dim.

        norm_axis_val = xr.DataArray(norm_axis_val, dims=norm_axis_val_dims, name=norm_axis)
        norm_axis_val, _ = xr.broadcast(norm_axis_val, data.drop_dims(norm_dim))

    # coord_dim_reversal_array = xr.DataArray
    if mode == 'nearest':
        # Find the norm_axis value closest to norm_axis_val
        norm_dim_val = data[norm_dim][abs(data[norm_axis] - norm_axis_val).argmin(norm_dim)]
    else:  # mode == 'linear', 'quadratic', 'cubic'
        data_norm_axis_units = extract_units(data.coords[norm_axis])
        norm_axis_val = norm_axis_val.pint.to(data_norm_axis_units)
        norm_axis_val = strip_units(norm_axis_val)
        norm_dim_val = convert_coord_to_dim(data.coords[norm_axis], norm_axis_val.data)

    norm_value = data.pint.interp({norm_dim: norm_dim_val}, mode)
    norm_data = data / norm_value
    norm_data.update(data.coords)
    return norm_data


def get_quick_plot_labels(
        xarray_data: xr.Dataset | xr.DataArray,
) -> dict[str, str]:
    """
    A method to set quick plot labels for the data variables.
    Takes data variables names and turns them to title-styled strings with units when necessary.

    Parameters
    ----------
    xarray_data : xr.Dataset | xr.DataArray
        The data array/set of interest.
    """
    if isinstance(xarray_data, xr.Dataset):
        key_list = xarray_data.variables.keys()
    else:
        key_list = [xarray_data.name] + list(xarray_data.coords.keys())

    label_dict = {}
    for key in key_list:  # parse data columns
        key = str(key)
        label = varname_to_title_string(key.replace('_per_', '/'), '/')  # get title-styled string
        if key != getattr(xarray_data, 'name', None):
            values = xarray_data[key].data
        else:
            values = xarray_data.data
        if isinstance(values, Qty):  # get column units if they exist.
            label = label + f' [{values.units:~P}]'
        label_dict[key] = label

    return label_dict


def quick_plot_xarray(
        xarray_data: xr.Dataset,
        var: str | None = None,
        coord1: str | None = None,
        coord2: str | None = None,
        var_units: str | None = None,
        coord1_units: str | None = None,
        coord2_units: str | None = None,
        plot_method: str | None = None,
        **plot_kwargs
) -> Type[plt.Artist]:
    """
    Create a quick plot of an xarray data array/set.

    Parameters
    ----------
    xarray_data : xr.Dataset
        The dataset to plot.
    var : str | None, optional
        Name of the data variable to use as the y-axis for 2D data and z-axis for 3D data.
        If None, the first variable in the dataset is used.
    coord1 : str | None, optional
        Name of the coordinate to use as the x-axis.
    coord2 : str | None, optional
        Name of the coordinate to use as the y-axis (only for 3D data).
    var_units : str | None, optional
        The units to change the plotted variable to.
    coord1_units : str | None, optional
        The units to change the x-axis to.
    coord2_units : str | None, optional
        The units to change the y-axis to (only for 3D data).
    plot_method : str | None, optional
        The specific plotting method to use. See `xarray.DataArray.plot` for options.
        If None, the default method for the input variable will be used.
    **plot_kwargs
        Additional keyword arguments to pass to the matplotlib.pyplot.plot function.

    Returns
    -------
    plt.Artist
        The object returned by the `xarray.DataArray.plot` method.

    Raises
    ------
    ValueError
        If the specified column names are not present in the data.

    """

    # Check input validity
    if var is None:  # Set norm_var to the first variable if not provided
        var = list(xarray_data.data_vars.keys())[0]
    if var not in xarray_data.variables:
        raise MethodInputError('var', var, list(xarray_data.variables.keys()), 'quick_plot')
    if coord1 is not None and coord1 not in xarray_data.coords:
        raise MethodInputError('coord1', coord1, list(xarray_data.coords.keys()), 'quick_plot')
    if coord2 is not None and coord1 not in xarray_data.coords:
        raise MethodInputError('coord2', coord2, list(xarray_data.coords.keys()), 'quick_plot')

    # Define the data
    data = xarray_data[var].copy(deep=True)

    # Check the user input method is correct
    if plot_method is not None:
        available_plot_methods = \
            [method for method in dir(data.plot) if not method.startswith('_')] + \
            ['linecollection']
        if plot_method not in available_plot_methods:
            raise MethodInputError('plot_method', plot_method, available_plot_methods, 'quick_plot')

    # Change units if necessary
    if var_units is not None:
        data = data.pint.to({var: var_units})
    if coord1 is not None and coord1_units is not None:
        data = data.pint.to({coord1: coord1_units})
    if coord2 is not None and coord2_units is not None:
        data = data.pint.to({coord2: coord2_units})

    # making my own method (-ish)
    set_colors = False
    if plot_method == 'linecollection':
        plot_method = 'line'
        set_colors = True
        plot_kwargs.setdefault('add_legend', False)
        hue = plot_kwargs.get('hue', None)
        if hue is None:
            if coord1 is None:
                coord1 = data.dims[-1]
                hue = data.dims[-2]
            else:
                other_dims = list(data.dims)
                other_dims.remove(data[coord1].dims[0])
                hue = other_dims[-1]
            plot_kwargs.update({'hue': hue})

    # Get plot function with either the default method, or the user's preferred method
    plot = getattr(data.plot, plot_method) if plot_method is not None else data.plot

    # set some default plot keyword arguments
    plot_kwargs.setdefault('center', False)
    plot_kwargs.setdefault('cmap', 'Spectral')

    # Use only accepted plot that are allowed from the plot method. Ignore others.
    plot_args = getfullargspec(getattr(dataarray_plot, plot_method)).kwonlyargs
    updated_plot_kwargs = {a: plot_kwargs[a] for a in plot_kwargs if a in plot_args}

    # Plot data!
    result = plot(x=coord1, y=coord2, **updated_plot_kwargs)

    ax: plt.Axes = result[0].axes if isinstance(result, typing.Sequence) else result.axes

    if set_colors:
        cmap_args = getfullargspec(set_linecollection_cmap).args
        cmap_args += getfullargspec(set_linecollection_cmap).kwonlyargs
        cmap_kwargs = {a: plot_kwargs[a] for a in plot_kwargs if a in cmap_args}
        cbar = set_linecollection_cmap(xarray_data=data, ax=ax, lines=result, **cmap_kwargs)
    else:
        cbar = None

    # Set axis labels if necessary
    if plot_kwargs.get('add_labels', True):
        quick_plot_labels = get_quick_plot_labels(xarray_data)

        x_axis = re.sub(r' \[[^)]*]', '', ax.get_xlabel())
        y_axis = re.sub(r' \[[^)]*]', '', ax.get_ylabel())

        x_label = quick_plot_labels[x_axis]
        y_label = quick_plot_labels[y_axis]

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        cbar = getattr(result, 'colorbar', cbar)
        if cbar is not None:
            cbar_ax: plt.Axes = cbar.ax
            z_axis = re.sub(r' \[[^)]*]', '', cbar_ax.get_ylabel())

            z_label = quick_plot_labels[z_axis]
            cbar_ax.set_ylabel(z_label)

    return result


def set_linecollection_cmap(
    xarray_data: xr.Dataset | xr.DataArray,
    hue: str,
    lines: list[plt.Line2D],
    ax: plt.Axes,
    cmap: str | mpl.colors.Colormap | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    norm: mpl.colors.Normalize = None,
    extend: ExtendOptions = 'neither',
    levels: typing.Sequence = None,
    center: float | bool = False,
    robust: bool = False,
    colors: str | None = None,
    add_colorbar=True,
    cbar_ax: plt.Axes = None,
    cbar_kwargs: dict[str, typing.Any] | None = None,
):
    """
    Parameters
    ----------
    cmap : matplotlib colormap name or colormap, optional
        The mapping from data values to color space. Either a
        Matplotlib colormap name or object. If not provided, this will
        be either ``'viridis'`` (if the function infers a sequential
        dataset) or ``'RdBu_r'`` (if the function infers a diverging
        dataset).
        See :doc:`Choosing Colormaps in Matplotlib <matplotlib:tutorials/colors/colormaps>`
        for more information.

        If *seaborn* is installed, ``cmap`` may also be a
        `seaborn color palette <https://seaborn.pydata.org/tutorial/color_palettes.html>`_.
        Note: if ``cmap`` is a seaborn color palette,
        ``levels`` must also be specified.
    vmin : float or None, optional
        Lower value to anchor the colormap, otherwise it is inferred from the
        data and other keyword arguments. When a diverging dataset is inferred,
        setting `vmin` or `vmax` will fix the other by symmetry around
        ``center``. Setting both values prevents use of a diverging colormap.
        If discrete levels are provided as an explicit list, both of these
        values are ignored.
    vmax : float or None, optional
        Upper value to anchor the colormap, otherwise it is inferred from the
        data and other keyword arguments. When a diverging dataset is inferred,
        setting `vmin` or `vmax` will fix the other by symmetry around
        ``center``. Setting both values prevents use of a diverging colormap.
        If discrete levels are provided as an explicit list, both of these
        values are ignored.
    norm : matplotlib.colors.Normalize, optional
        If ``norm`` has ``vmin`` or ``vmax`` specified, the corresponding
        kwarg must be ``None``.
    extend : {'neither', 'both', 'min', 'max'}, optional
        How to draw arrows extending the colorbar beyond its limits. If not
        provided, ``extend`` is inferred from ``vmin``, ``vmax`` and the data limits.
    levels : int or array-like, optional
        Split the colormap (``cmap``) into discrete color intervals. If an integer
        is provided, "nice" levels are chosen based on the data range: this can
        imply that the final number of levels is not exactly the expected one.
        Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to
        setting ``levels=np.linspace(vmin, vmax, N)``.
    center : float, optional
        The value at which to center the colormap. Passing this value implies
        use of a diverging colormap. Setting it to ``False`` prevents use of a
        diverging colormap.
    robust : bool, optional
        If ``True`` and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with 2nd and 98th percentiles instead of the extreme values.
    colors : str or array-like of color-like, optional
        A single color or a sequence of colors. If the plot type is not ``'contour'``
        or ``'contourf'``, the ``levels`` argument is required.
    cbar_ax : matplotlib axes object, optional
        Axes in which to draw the colorbar.
    cbar_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the colorbar
        (see :meth:`matplotlib:matplotlib.figure.Figure.colorbar`).
    """

    if cmap and colors:
        raise ValueError("Can't specify both cmap and colors.")

    cbar_kwargs = {} if cbar_kwargs is None else dict(cbar_kwargs)

    plot_data = xarray_data[hue].data
    cmap_params = _determine_cmap_params(
        plot_data=plot_data,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        center=center,
        robust=robust,
        extend=extend,
        levels=levels,
        norm=norm,
    )

    cm = mpl.cm.ScalarMappable(norm=cmap_params['norm'], cmap=cmap_params['cmap'])
    colors = cm.to_rgba(plot_data)
    for color, line in zip(colors, lines):
        line.set_color(color)

    if ax.get_legend() is not None:
        ax.get_legend().remove()
        ax.legend(list(plot_data), title=get_quick_plot_labels(xarray_data)[hue])

    if add_colorbar:
        cbar: mpl.colorbar.Colorbar = _add_colorbar(cm, ax, cbar_ax, cbar_kwargs, cmap_params)
        cbar.ax.set_ylabel(hue)
        return cbar

    return None
