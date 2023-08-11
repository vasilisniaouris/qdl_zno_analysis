"""
Module containing various utility functions for this package.
"""
import dataclasses
import re
import types
import typing
import warnings
from dataclasses import dataclass, fields
from inspect import getfullargspec
from operator import attrgetter
from typing import Any, Sequence, Type, Union, TypeVar

import numpy as np
import pint
import pint_xarray
import scipy.interpolate as spi
import xarray as xr
from pint import UnitStrippedWarning
from scipy.ndimage import median_filter, uniform_filter1d
from xarray.core.types import ExtendOptions
from xarray.plot import dataarray_plot
from xarray.plot.utils import _determine_cmap_params, _add_colorbar
from pint_xarray.conversion import extract_units, strip_units, attach_units

from qdl_zno_analysis._extra_dependencies import mpl, plt, HAS_VISUALIZATION_DEP
from qdl_zno_analysis import ureg, Qty
from qdl_zno_analysis.constants import default_units
from qdl_zno_analysis.errors import assert_bounds, assert_options, ArrayShapeError, ValueOutOfOptionsError, \
    ArrayDimensionNumberError
from qdl_zno_analysis.typevars import EnhancedNumeric, XRObject, PLTArtist, ArrayObject


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
        if typing.get_origin(arg_type) in [Union, types.UnionType]:
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
    if isinstance(value, int | float | list | np.ndarray):
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


def normalize_dict(dictionary: dict, parent_key='', separator='.') -> dict:
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
    dict
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


def find_changing_values_in_list_of_dict(list_of_dicts: list[dict], reverse_result=False) -> dict:
    """
    Find the keys in a list of dictionaries whose values change between the dictionaries.

    Parameters
    ----------
    list_of_dicts: list[dict]
        The list of dictionaries to analyze.
    reverse_result: bool, optional
        If True, returns only unchanged values. Defaults to False.

    Returns
    -------
    dict
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

            if test_element not in value_list and not (np.isnan(test_element) and np.all(np.isnan(value_list))):
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
    ValueOutOfBoundsError
        If start or end are outside the range of `x_data`.
    ArrayShapeError
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
    assert_options(np.shape(y_data), {np.shape(x_data)}, 'y_data', ArrayShapeError, y_data)

    sorted_idx = x_data.argsort()
    x_data = x_data[sorted_idx]
    y_data = y_data[sorted_idx]

    assert_bounds(start, (-np.inf, x_data[0]), 'start')
    assert_bounds(end, (x_data[-1], np.inf), 'end')

    x_interp = x_data[(x_data > start) & (x_data < end)]
    x_interp = np.append(start, x_interp)
    x_interp = np.append(x_interp, end)
    y_interp = np.interp(x_interp, x_data, y_data, left=np.nan, right=np.nan)
    return np.trapz(y_interp, x_interp)


def find_nearest(array: Qty | np.ndarray | list, value: EnhancedNumeric) -> EnhancedNumeric:
    """
    Returns the value in the input array that is closest in value to the input `value`.

    Parameters
    ----------
    array: pint.Quantity | np.ndarray | list
        The input array to search for the closest value.
    value: EnhancedNumeric
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


def get_rolling_mean(data, window_size=5):
    """
    Rolling mean is a good way to look at the general behaviour of noisy data.
    """
    # https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    return uniform_filter1d(data, size=window_size)


def get_rolling_median(data, window_size=10):
    """
    Rolling median is better than rolling average when you want to identify the behaviour of your data even with a
    few points that are extreme.
    """
    return median_filter(data, size=window_size)


def get_rolling_stdev(data, window_size=10):
    """
    Rolling standard deviation is a way to infer a local standard deviation, when your data changes a lot overall but
    only varies a little within smaller scales.
    """
    rolling_average_of_squares = get_rolling_mean(data ** 2, window_size)
    square_of_rolling_average = get_rolling_mean(data, window_size) ** 2
    return np.sqrt(rolling_average_of_squares - square_of_rolling_average)


# Will always yield 0?
# def get_rolling_stdev_by_median(data, window_size=10):
#     """
#     Rolling standard deviation is a way to infer a local standard deviation, when your data changes a lot overall but
#     only varies a little within smaller scales.
#     """
#     rolling_average_of_squares = get_rolling_median(data ** 2, window_size)
#     square_of_rolling_average = get_rolling_median(data, window_size) ** 2
#     return np.sqrt(rolling_average_of_squares - square_of_rolling_average)


def get_pchip_interp_data(x_data, y_data, x_new_data):
    """
    Your generic linear interpolation function between points.
    """
    return spi.pchip_interpolate(x_data, y_data, x_new_data)


def get_lsq_univariate_spline_data(x_data, y_data, x_new_data, knot_percentage=0.03):
    """
    This spline fits piecewise cubic (or other order if chanfged) curves with continuous first and second derivatives
    between the chosen knots. A knot is a point where to separate two pieces of the dataset.
    Here we work with knot percentage, meaning that every N*percentage we get an equidistant knot.
    """
    total_knots = int(np.ceil(len(x_data) * knot_percentage))
    knots = x_data[0::len(x_data) // total_knots]
    knots = knots[1:-1]
    spline = spi.LSQUnivariateSpline(x_data, y_data, knots)
    return spline(x_new_data)


def get_univariate_spline_data(x_data, y_data, x_new_data):
    """
    This funciton gives a single cubic (or other order) spline over all data.
    """
    spline = spi.UnivariateSpline(x_data, y_data)
    return spline(x_new_data)


def get_cubic_spline_data(x_data, y_data, x_new_data):
    """
    Fits a piecewise cubic polynomial over the data. I believe every point is a knot.
    """
    spline = spi.CubicSpline(x_data, y_data)
    return spline(x_new_data)


def find_outliers(
        x_data: ArrayObject,
        y_data: ArrayObject,
        rolling_window_size: int = 5,
        flagging_significance: float = 3.,
        maximum_outlier_cluster_size: int = 2,
        hard_min_limit: EnhancedNumeric | np.ndarray = np.nan,
        hard_max_limit: EnhancedNumeric | np.ndarray = np.nan,
        rolling_data_method: str = 'median',
        approximate_local_stdev_method: str = 'mean',
        assume_sorted: bool = True,
        repeat: int = -1,
        return_entire_healed_y_data: bool = True,
) -> tuple[np.ndarray[int], ArrayObject] | tuple[np.ndarray[int], ArrayObject, ArrayObject]:
    """
    Find indexes of outliers in the given data based on rolling
    statistics and provides "healed" values for the aforementioned
    indexes.

    This method works best for evenly sampled data. Near-evenly
    sampled data would yield reasonable results, too.

    It calculates the rolling median or mean of the y-values
    using a specified rolling window size. Then, it calculates the
    rolling standard deviation. We use the rolling standard deviation
    to find the average standard deviation in the data.
    Outliers are flagged based on the difference between the y-values and
    the rolling mean/median $\pm$ the average/median rolling standard deviation
    multiplied by the flagging significance factor.

    Outliers occurring in clusters with a size less than or equal to
    the maximum outlier cluster size are filtered and returned as the
    final list of outliers.

    If assume_sorted is False, the input data is assumed to be unsorted,
    and it will be sorted internally based on the x-values.

    This process is repeat as many times requested by the user, or until the
    average standard deviation stabilizes (default).

    The returned healed outliers are the result of the iterative rolling means
    on the outlier index (depends on `repeat` parameter).

    Currently, can only accept 1-D arrays.

    Parameters:
    -----------
    x_data: numpy.ndarray | xr.DataArray
        Array of x-values.
    y_data: numpy.ndarray | xr.DataArray
        Array of y-values.
    rolling_window_size: int, optional
        Size of the rolling window for calculating statistics, by default 5 points.
    flagging_significance: float, optional
        Significance factor for flagging outliers, by default 5.0.
        The smaller the value the stricter the filtering (more outliers will be found).
    maximum_outlier_cluster_size: int, optional
        Maximum size of outlier clusters to consider, by default 1.
    hard_min_limit: int | float | Qty | np.ndarray, optional
        Either a value or an array of values with dimension size up to the same as `y_data`,
        this attribute is a minimum cut-off for the `y_data` that does not depend on the rolling statistics.
    hard_max_limit: int | float | Qty | np.ndarray, optional
        Either a value or an array of values with dimension size up to the same as `y_data`,
        this attribute is a maximum cut-off for the `y_data` that does not depend on the rolling statistics.
    rolling_data_method: str, {'mean', 'median'}, optional
        Method for calculating rolling data statistics. Options are 'mean' and 'median', by default 'median'.
    approximate_local_stdev_method: str, {'mean', 'median'}, optional
        Method for calculating the expected data jitter (mean/median local stdev of data). Defaults to 'mean'.
        If there is a small dataset with huge outliers, it would be better to use median. For larger
        datasets with smaller outliers, 'mean' may be better.
    assume_sorted: bool, optional
        Flag indicating whether the input data is assumed to be sorted, by default True.
    repeat: int, optional
        The amount of times this process will be repeated. Repetition will help when there are
        different levels of outliers in the data (some very strong, some softer, but still stronger than expected).
        The special value "-1" (default) will repeat until the average of the rolling stdev is stabilized.
    return_entire_healed_y_data: bool, optional
        If True, returns the `y_data` whose outliers have been replaced with the healed values.

    Returns:
    --------
    tuple[np.ndarray[int], np.ndarray | xr.DataArray]
        A tuple containing the list of outlier indexes in the original order of the input data,
        and the "healed" data for said outliers.

    Raises:
    -------
    ValueOutOfBoundsError
        If rolling_data_method or approximate_local_stdev_method is not one of the allowed options.
    """
    # Assert allowed options
    assert_options(rolling_data_method, {'mean', 'median'}, 'rolling_data_method', ValueOutOfOptionsError)
    assert_options(approximate_local_stdev_method, {'mean', 'median'},
                   'approximate_local_stdev_method', ValueOutOfOptionsError)

    original_y_data = y_data

    # Get x data array
    if isinstance(x_data, xr.DataArray):
        x_data = x_data.data
    if isinstance(x_data, Qty):
        x_data = x_data.m

    # Get y data array and get correct units for hard limits
    y_units = None
    if isinstance(y_data, xr.DataArray):
        y_data = y_data.data
    if isinstance(y_data, Qty):
        y_units = y_data.u
        if isinstance(hard_min_limit, Qty):
            hard_min_limit = hard_min_limit.m_as(y_units)
        if isinstance(hard_max_limit, Qty):
            hard_max_limit = hard_max_limit.m_as(y_units)
        y_data = y_data.m

    if hard_min_limit is None:
        hard_min_limit = np.nan
    if hard_max_limit is None:
        hard_max_limit = np.nan

    # Get sorted data
    if not assume_sorted:
        sorted_indices = np.argsort(np.array(x_data))
        sorted_y_data = np.asarray(y_data)[sorted_indices]
        original_indices = np.argsort(sorted_indices)
    else:
        sorted_y_data = np.asarray(y_data)

    if rolling_data_method == 'median':
        get_rolling_values = get_rolling_median
    else:
        get_rolling_values = get_rolling_mean

    # Get rolling statistics
    rolling_y_data = get_rolling_values(sorted_y_data, rolling_window_size)

    def get_average_rolling_stdev(method, yd):
        rolling_stdev = get_rolling_stdev(yd, rolling_window_size)
        return np.median(rolling_stdev) if method == 'median' else np.average(rolling_stdev)

    average_rolling_stdev = get_average_rolling_stdev(approximate_local_stdev_method, sorted_y_data)

    # get rolling limits
    rolling_max_limit = rolling_y_data + flagging_significance * average_rolling_stdev
    rolling_min_limit = rolling_y_data - flagging_significance * average_rolling_stdev

    # Find Outlier indexes
    outlier_indexes_on_sorted_data = np.where(
        (sorted_y_data > rolling_max_limit) | (sorted_y_data > hard_max_limit) |
        (sorted_y_data < rolling_min_limit) | (sorted_y_data < hard_min_limit))[0]

    def filter_clusters(indexes, max_size):
        indexes = np.array(indexes)
        sorted_indexes = np.sort(indexes)
        diff = np.diff(sorted_indexes)

        # Find the indices where the cluster breaks
        cluster_break_indices = np.where(diff > 1)[0]

        filtered_indexes = []
        start_index = 0

        for break_index in cluster_break_indices:
            cluster = sorted_indexes[start_index:break_index + 1]
            if len(cluster) <= max_size:
                filtered_indexes.extend(cluster)
            start_index = break_index + 1

        # Check the last cluster
        last_cluster = sorted_indexes[start_index:]
        if len(last_cluster) <= max_size:
            filtered_indexes.extend(last_cluster)

        return np.array(filtered_indexes, dtype=np.int_)

    filtered_outliers_on_sorted_data = \
        filter_clusters(outlier_indexes_on_sorted_data, maximum_outlier_cluster_size)

    if not assume_sorted:
        outliers = original_indices[filtered_outliers_on_sorted_data]
    else:
        outliers = filtered_outliers_on_sorted_data

    rolling_values = get_rolling_values(np.asarray(y_data), rolling_window_size)
    new_y_data = np.array([y_data[i] if i not in outliers else rolling_values[i] for i in range(len(x_data))])

    def do_rep(ols, new_yd):
        new_ols, _ = find_outliers(
            x_data, new_yd, rolling_window_size, flagging_significance, maximum_outlier_cluster_size,
            hard_min_limit, hard_max_limit, rolling_data_method, approximate_local_stdev_method, assume_sorted,
            repeat=0, return_entire_healed_y_data=False)

        ols = np.unique(np.append(new_ols, ols))
        rm = get_rolling_values(new_yd, rolling_window_size)
        new_yd = np.array([new_yd[i] if i not in ols else rm[i] for i in range(len(x_data))])

        return ols, new_yd

    if repeat >= 0:
        for rep in range(repeat):
            outliers, new_y_data = do_rep(outliers, new_y_data)
    else:
        flag = True
        while flag:
            outliers, new_y_data = do_rep(outliers, new_y_data)
            old_average_rolling_stdev = average_rolling_stdev
            average_rolling_stdev = get_average_rolling_stdev(approximate_local_stdev_method, new_y_data)
            flag = old_average_rolling_stdev / average_rolling_stdev - 1 > 1e-5  # continues if true

    if y_units:
        new_y_data = new_y_data * y_units

    if isinstance(original_y_data, xr.DataArray):
        original_y_data = original_y_data.copy(deep=True)
        original_y_data.data = new_y_data
        new_y_data = original_y_data

    if return_entire_healed_y_data:
        return outliers, new_y_data[outliers], new_y_data
    else:
        return outliers, new_y_data[outliers]


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


def convert_coord_to_dim(coord_array: xr.DataArray, coord_data_to_convert: np.ndarray | Qty = None):
    """
    Convert 1D xr.DataArray Coordinates to their corresponding xr.DataArray dimension.
    The conversion is performed using a linear interpolation.
    Linear interpolation is the only method that makes sense when the dimension/coordinate data may not
    contain unique values, nor be evenly spaced.

    Parameters
    ----------
    coord_array: xr.DataArray
        The coordinate data array.
    coord_data_to_convert: np.ndarray | Qty
        The coordinate data to be converted. If None, the xr.DataArray data will be used.

    Returns
    -------
    xr.DataArray
        The xr.DataArray dimensions corresponding to the input coordinate.

    """
    coord_array_stripped = strip_units(coord_array)

    # TODO: Make n-dimensional possible
    dim_name = coord_array.dims[0]
    dim_array = coord_array[dim_name]
    dim_array_stripped = strip_units(dim_array)

    conversion_interpolation_function = spi.interp1d(coord_array_stripped, dim_array_stripped, 'linear')

    if coord_data_to_convert is None:
        coord_data_to_convert = coord_array_stripped.data
    elif isinstance(coord_data_to_convert, Qty):
        coord_data_to_convert = coord_data_to_convert.m_as(coord_array.data.u)

    result_stripped_data = conversion_interpolation_function(coord_data_to_convert)

    result_stripped = xr.DataArray(
        result_stripped_data,
        coords=({coord_array.name: ((coord_array.name,), coord_data_to_convert)}),
        name=dim_name
    )
    # result_stripped = xr.DataArray(result_stripped_data, coords=coord_array.coords, name=dim_name)
    return attach_units(result_stripped, extract_units(coord_array))


def get_null(value: Any):
    if isinstance(value, Qty):
        return np.nan * value.u
    elif isinstance(value, int | float | np.int_ | np.float_):
        return np.nan
    else:
        return None


def uniquify_xarray_coord(
        xarray_data: XRObject,
        coord: str,
        data_aggregation_method: str = 'mean',
        coordinate_aggregation_method: str = 'nan',
        reindex_dim: bool = False,
) -> XRObject:
    """
    Takes an xarray object and removes uses the given 'coord' key values only once.
    The given coordinate must be 1D.
    All duplicate data are aggregated according to the 'data_aggregation_method', such as via a 'mean' method.
    All coordinates that are dependent on the same dimension as the main coordinate will be aggregated according to
    the 'coordinate_aggregation_method' method.

    This method is usefull when data were taken multiple times on a specific coordinate, but we need to plot
    only one data-point (1D plots) / pixel (2D plots).

    Parameters
    ----------
    xarray_data : xr.Dataset | xr.DataArray
        The dataset or data array that will be modified
    coord: str
        The coordinate key whose data will become unique. Must be 1D.
    data_aggregation_method: str, {'min', 'max', 'mean', 'median'}, Optional
        A string signifying the method with which to aggregate variable data of duplicate coordinate values.
        Defaults to 'mean'.
    coordinate_aggregation_method: str, {'min', 'max', 'mean', 'median', 'nan'}, Optional
        A string signifying the method with which to aggregate other coordinate data dependent on the same dimension
        as the coordinate values.
        Defaults to 'nan'.
        All methods other than 'nan' are part of the xarray library.
        The 'nan' method will set the value equal to nan of the extra coordinate values are not duplicated like the main
        coordinate values, otherwise it will use the extra coordinate duplicated values.
    reindex_dim: bool, Optional
        Defaults to True.
        - If true, will set the coordinate dimension to range(len(xarray_data[coord])).
        - If false, will use the first occurrence for each duplicate coordinate value
          to populate the corresponding dimension value.

    Returns
    -------
    xr.Dataset | xr.DataArray
        The updated xarray object with aggregated duplicate data.

    Raises
    ------
    ValueOutOfBoundsError
        If coordinatate array is not a 1D array.
    ValueOutOfOptionsError
        If data_aggregation_method or coordinate_aggregation_method in not within the accepted methods.

    Examples
    --------
    >>> x_index = [0, 1, 2, 3, 4]
    >>> x1 = [11, 6, 11, 3, 7]
    >>> x2 = [1 / (xi * 2) for xi in x1]
    >>> x3 = [2, 3, 4, 5, 6.]
    >>>
    >>> data = [25., 5, 22, 7, 1]
    >>> data2 = [31., 2, 37, 3, 5]
    >>>
    >>> # Create xarray DataArray
    >>> dataset = xr.Dataset({
    ...     'data': (('x_index',), data),
    ...     'data2': (('x_index',), data2),
    ...     },
    ...     coords={
    ...         'x1': (('x_index',), x1),
    ...         'x2': (('x_index',), x2),
    ...         'x3': (('x_index',), x3),
    ...         'x_index': (('x_index',), x_index),
    ...     })
    >>> dataset
    <xarray.Dataset>
    Dimensions:  (x_index: 5)
    Coordinates:
        x1       (x_index) int32 11 6 11 3 7
        x2       (x_index) float64 0.04545 0.08333 0.04545 0.1667 0.07143
        x3       (x_index) float64 2.0 3.0 4.0 5.0 6.0
      * x_index  (x_index) int32 0 1 2 3 4
    Data variables:
        data     (x_index) float64 25.0 5.0 22.0 7.0 1.0
        data2    (x_index) float64 31.0 2.0 37.0 3.0 5.0
    >>> # By default, will 'mean' the variables, and 'nan'
    >>> # other x_index-related coordinates if the concatenated
    >>> # duplicate values are not the same (e.g. see x3 in this example).
    >>> uniquify_xarray_coord(dataset, 'x2')
    <xarray.Dataset>
    Dimensions:  (x_index: 4)
    Coordinates:
        x1       (x_index) int32 11 6 3 7
        x2       (x_index) float64 0.04545 0.08333 0.1667 0.07143
        x3       (x_index) float64 nan 3.0 5.0 6.0
      * x_index  (x_index) int32 0 1 3 4
    Data variables:
        data     (x_index) float64 23.5 5.0 7.0 1.0
        data2    (x_index) float64 34.0 2.0 3.0 5.0


    """
    data = xarray_data.copy(deep=True)
    coord_array = data[coord].copy(deep=True)
    assert_options(len(list(coord_array.dims)), {1}, 'len(coord_array.dims)', ArrayDimensionNumberError)

    assert_options(
        data_aggregation_method,
        {'min', 'max', 'mean', 'median'},
        'data_aggregation_method',
        ValueOutOfOptionsError,
    )
    assert_options(
        coordinate_aggregation_method,
        {'min', 'max', 'mean', 'median', 'nan'},
        'coordinate_aggregation_method',
        ValueOutOfOptionsError,
    )

    dim_name = list(coord_array.dims)[0]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UnitStrippedWarning)
        coord_array = coord_array.sortby(coord_array)

    unique_values, unique_value_index, value_counts = np.unique(coord_array, return_index=True, return_counts=True)
    duplicate_values = unique_values[value_counts > 1]

    # Get coords that also depend solely on the same dim as the main coord.
    extra_coords = [str(c) for c in list(coord_array.coords.keys()) if str(c) not in [coord, dim_name]]

    # Get data for each unique coordinate
    # We sort the coord_array of unique coord values by the dim_name,
    # so that all sorting comes back to the original array.
    unique_coordinate_data = data.isel({dim_name: coord_array[dim_name][unique_value_index].sortby(dim_name)})

    # Update unique_coordinate_data data for each duplicate coordinate value according to selected
    # data_aggregation_method.
    for value in duplicate_values:
        # Update variables
        selected_data = data.isel({dim_name: data[coord] == value})
        aggregated_selected_data: XRObject = getattr(
            selected_data, data_aggregation_method)(dim_name, keepdims=True, keep_attrs=True)
        unique_coordinate_data.loc[{dim_name: unique_coordinate_data[coord] == value}] = aggregated_selected_data

        # Update extra coordinates
        for c in extra_coords:
            selected_coord = selected_data[c]
            if coordinate_aggregation_method == 'nan':
                updated_extra_coord_value = selected_coord.data[0]
                if not np.all(selected_coord == selected_coord.data[0]):
                    updated_extra_coord_value = get_null(selected_coord.data[0])
            else:
                updated_extra_coord_value: XRObject = getattr(
                    selected_coord, coordinate_aggregation_method)(dim_name)

            unique_coordinate_data[c].loc[
                {dim_name: unique_coordinate_data[coord] == value}] = updated_extra_coord_value

    # Re-index dimension of coordinate
    if reindex_dim:
        unique_coordinate_data[dim_name] = range(len(unique_coordinate_data[dim_name]))

    return unique_coordinate_data


def reveal_hidden_xarray_sub_dims(
        xarray_data: XRObject,
        stack_dim: str,
        unstacking_dims: tuple[str, ...],
) -> XRObject:
    """
    If there are hidden sub-dimensions within an xarray dimension, one can change "unstack" the "stack" dimensions.

    This method is usefull when taking data by varying two parameters, by these two parameters are stack on the
    file-related coordinate.

    Parameters
    ----------
    xarray_data: xr.Dataset | xr.DataArray
        The target xarray object.
    stack_dim: str
        The dimension which is stack.
    unstacking_dims: tuple[str, ...]
        The coordinates/new dimensions. Best use indeces for these, not quantities with units.
        An easy way to do so is to use the method `utils.indexify_xarray_coord`.

    Returns
    -------
    xr.Dataset | xr.DataArray
        An xarray object whose stack_dim is now unstack.

    Raises
    ------
    ValueOutOfOptionsError
        If stack_dim is not in the xarray_data.dims.

    Examples
    --------
    >>> x_index = [0, 1, 2, 3, 4]
    >>> x = [11, 6, 11, 3, 7]
    >>> x2 = [1 / (xi * 2) for xi in x]
    >>> x3 = [2, 3, 4, 5, 6.]
    >>>
    >>> y_index = [0, 1, 2, 3]
    >>> y1 = [1, 2, 1, 2]  # <--- hidden dimension 1 of y_index
    >>> y2 = [30., 30, 40, 40]  # <--- hidden dimension 2 of y_index
    >>> y3 = [55., 66, 77, 88]  # <--- coordinate with 1D relationship with y_index
    >>>
    >>> data_base = [25., 5, 22, 7, 1]
    >>> data = [[data_base[j] * y1[i] + y2[i] for i in y_index] for j in range(len(data_base))]
    >>> data2_base = [31., 2, 37, 3, 5]
    >>> data2 = [[data2_base[j] * y1[i] + y2[i] for i in y_index] for j in range(len(data2_base))]
    >>>
    >>> # Create xarray DataArray
    >>> dataset = xr.Dataset({
    ...     'data': (('x_index', 'y_index'), data),
    ...     'data2': (('x_index', 'y_index'), data2),
    ...     },
    ...     coords={
    ...         'x': (('x_index',), x),
    ...         'x2': (('x_index',), x2),
    ...         'y1': (('y_index',), y1),
    ...         'y2': (('y_index',), y2),
    ...         'y3': (('y_index',), y3),
    ...         'x_index': (('x_index',), x_index),
    ...         'y_index': (('y_index',), y_index),
    ...     })
    >>> dataset
    <xarray.Dataset>
    Dimensions:  (x_index: 5, y_index: 4)
    Coordinates:
        x        (x_index) int32 11 6 11 3 7
        x2       (x_index) float64 0.04545 0.08333 0.04545 0.1667 0.07143
        y1       (y_index) int32 1 2 1 2
        y2       (y_index) float64 30.0 30.0 40.0 40.0
        y3       (y_index) float64 55.0 66.0 77.0 88.0
      * x_index  (x_index) int32 0 1 2 3 4
      * y_index  (y_index) int32 0 1 2 3
    Data variables:
        data     (x_index, y_index) float64 55.0 80.0 65.0 90.0 ... 32.0 41.0 42.0
        data2    (x_index, y_index) float64 61.0 92.0 71.0 102.0 ... 40.0 45.0 50.0
    >>> # Create proper indexes for multi-indexed dimensions using unique coordinate values only,
    >>> # but we do not want to sort by the coordinates we are indexing.
    >>> dataset = indexify_xarray_coord(dataset, 'y1', use_unique_values_only=True)
    >>> dataset['y1_index'].data
    array([0, 1, 0, 1])
    >>> dataset = indexify_xarray_coord(dataset, 'y2', use_unique_values_only=True)
    >>> dataset['y2_index'].data
    array([0, 0, 1, 1])
    >>> dataset = reveal_hidden_xarray_sub_dims(dataset, 'y_index', ('y1_index', 'y2_index'))
    >>> dataset
    <xarray.Dataset>
    Dimensions:   (y1_index: 2, y2_index: 2, x_index: 5)
    Coordinates:
      * y1_index  (y1_index) int32 0 1
      * y2_index  (y2_index) int32 0 1
        x         (x_index) int32 11 6 11 3 7
        x2        (x_index) float64 0.04545 0.08333 0.04545 0.1667 0.07143
        y1        (y1_index) int32 1 2
        y2        (y2_index) float64 30.0 40.0
        y3        (y1_index, y2_index) float64 55.0 77.0 66.0 88.0
      * x_index   (x_index) int32 0 1 2 3 4
        y_index   (y1_index, y2_index) int32 0 2 1 3
    Data variables:
        data      (x_index, y1_index, y2_index) float64 55.0 65.0 80.0 ... 32.0 42.0
        data2     (x_index, y1_index, y2_index) float64 61.0 71.0 92.0 ... 40.0 50.0
    """
    assert_options(stack_dim, xarray_data.dims, 'stack_dim', ValueOutOfOptionsError)

    # We need to get a replica of the stack dim data, otherwise it is not retained through the unstacking process.
    coords_for_old_dim = list(unstacking_dims) + [stack_dim]
    old_dim_replica = xr.DataArray(
        xarray_data[stack_dim].data,
        coords={c: ((stack_dim,), xarray_data[c].data) for c in coords_for_old_dim},
        dims=stack_dim
    )

    # We unstack the stack_dim data replica
    old_dim_replica = old_dim_replica.set_xindex(unstacking_dims)
    old_dim_replica = old_dim_replica.unstack(stack_dim)

    # We unstack the rest of the xarray object
    xarray_data = xarray_data.set_xindex(unstacking_dims)
    xarray_data = xarray_data.unstack(stack_dim)

    # We re-include the lost stack_dim data
    xarray_data = xarray_data.assign_coords({stack_dim: old_dim_replica})

    # Update other coordinates that may be affected by this coordinate/dimension change
    for unstacking_dim in unstacking_dims:
        unstacking_dim_index_range = range(len(xarray_data[unstacking_dim]))
        for c in list(xarray_data.coords.keys()):
            if unstacking_dim in xarray_data[c].dims and c != unstacking_dim:
                first_value = xarray_data[c].isel({unstacking_dim: 0})
                if np.all(
                        [xarray_data[c].isel({unstacking_dim: i}).data == first_value.data
                         for i in unstacking_dim_index_range]
                ):
                    xarray_data[c] = xarray_data[c].isel({unstacking_dim: 0}, drop=True)

    return xarray_data


def indexify_xarray_coord(
        xarray_data: XRObject,
        coord: str,
        use_unique_values_only: bool = False,
        swap_dim: bool = False,
        new_coord_name: str | None = None,
        add_only_if_necessary: bool = False,
        sort_index_by_coord: bool = True,
        verbose: bool = True,
) -> XRObject:
    """
    Takes an xarray object, sorts it by the given coordinate values and adds a new coordinate index based
    on the coordinate's length.
    The given coordinate must be 1D.

    If swap_dims is true, the new coordinate index replaces the original coordinate dimension.
    In this case, the new xarray object is sorted by the new dimension.

    This method is usefull when there are data that are not sorted the same way as their corresponding dimension.

    Parameters
    ----------
    xarray_data : xr.Dataset | xr.DataArray
        The dataset or data array to which the new index will be added.
    coord: str
        The coordinate key whose data will be indexed. Must be 1D.
    swap_dim: bool, Optional
        Defaults to False. If true, swaps the original coordinate's dimension with the new coordinate index
        and sorts the xarray object by the new coordinate index.
    use_unique_values_only: bool
        If true, uses only the unique values of the coordinate data to create an index range, meaning that
        indexes may be duplicated along the new indexified coordinate. Very useful in creating MultiIndex
        dimensions for `reveal_hidden_xarray_sub_dims`. Defaults to False.
    new_coord_name: str | None, Optional
        The new coordinate name. Defaults to coord + '_index'.
    add_only_if_necessary: bool, Optional
        If true, will only add the new index, if the new index range does not already exist. Defaults to False.
    sort_index_by_coord: bool
        If True, it will return a xarray object whose new indexes will be generated for the sorted coordinate.
        Defaults to True.
    verbose: bool
        If add_only_if_necessary is true and verbose is true, it will print out the coordinate that matches the
        new index range. Defaults to True.

    Returns
    -------
    xr.Dataset | xr.DataArray
        The updated xarray object with a new index.

    Raises
    ------
    ValueOutOfBoundsError
        If coordinatate array is not a 1D array.

    Examples
    --------
    >>> x_index = [0, 1, 2, 3, 4]
    >>> x1 = [11, 6, 11, 3, 7]
    >>> x2 = [1 / (xi * 2) for xi in x1]
    >>> x3 = [2, 3, 4, 5, 6.]
    >>>
    >>> data = [25., 5, 22, 7, 1]
    >>> data2 = [31., 2, 37, 3, 5]
    >>>
    >>> # Create xarray DataArray
    >>> dataset = xr.Dataset({
    ...     'data': (('x_index',), data),
    ...     'data2': (('x_index',), data2),
    ...     },
    ...     coords={
    ...         'x1': (('x_index',), x1),
    ...         'x2': (('x_index',), x2),
    ...         'x3': (('x_index',), x3),
    ...         'x_index': (('x_index',), x_index),
    ...     })
    >>> dataset
    <xarray.Dataset>
    Dimensions:  (x_index: 5)
    Coordinates:
        x1       (x_index) int32 11 6 11 3 7
        x2       (x_index) float64 0.04545 0.08333 0.04545 0.1667 0.07143
        x3       (x_index) float64 2.0 3.0 4.0 5.0 6.0
      * x_index  (x_index) int32 0 1 2 3 4
    Data variables:
        data     (x_index) float64 25.0 5.0 22.0 7.0 1.0
        data2    (x_index) float64 31.0 2.0 37.0 3.0 5.0
    >>> indexify_xarray_coord(dataset, 'x2')
    <xarray.Dataset>
    Dimensions:   (x_index: 5)
    Coordinates:
        x1        (x_index) int32 11 6 11 3 7
        x2        (x_index) float64 0.04545 0.08333 0.04545 0.1667 0.07143
        x3        (x_index) float64 2.0 3.0 4.0 5.0 6.0
      * x_index   (x_index) int32 0 1 2 3 4
        x2_index  (x_index) int32 0 3 1 4 2
    Data variables:
        data      (x_index) float64 25.0 5.0 22.0 7.0 1.0
        data2     (x_index) float64 31.0 2.0 37.0 3.0 5.0
    >>> indexify_xarray_coord(dataset, 'x2', swap_dim=True)
    <xarray.Dataset>
    Dimensions:   (x2_index: 5)
    Coordinates:
        x1        (x2_index) int32 11 11 7 6 3
        x2        (x2_index) float64 0.04545 0.04545 0.07143 0.08333 0.1667
        x3        (x2_index) float64 2.0 4.0 6.0 3.0 5.0
        x_index   (x2_index) int32 0 2 4 1 3
      * x2_index  (x2_index) int32 0 1 2 3 4
    Data variables:
        data      (x2_index) float64 25.0 22.0 1.0 5.0 7.0
        data2     (x2_index) float64 31.0 37.0 5.0 2.0 3.0

    """
    data = xarray_data.copy(deep=True)
    coord_array = data[coord].copy(deep=True)
    assert_options(len(list(coord_array.dims)), {1}, 'len(coord_array.dims)', ArrayDimensionNumberError)

    dim_name = list(coord_array.dims)[0]

    if sort_index_by_coord:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UnitStrippedWarning)
            coord_array = coord_array.sortby(coord)

    if new_coord_name is None:
        new_coord_name = f'{coord}_index'

    if use_unique_values_only:
        unique_values, unique_inversion_index_array = np.unique(data[coord], return_inverse=True)
        new_coord_data = np.array(range(len(unique_values)))[unique_inversion_index_array]
    else:
        new_coord_data = range(len(data[coord].data))
    coord_array = coord_array.assign_coords({new_coord_name: ((dim_name,), new_coord_data)})

    # find if the potential new indexing already exists for another coordinate.

    is_value_index_in_data = False
    if add_only_if_necessary:
        for data_coord_key in data.coords:
            if not data.coords[data_coord_key].dims == coord_array.dims:
                continue
            if np.all(data[data_coord_key].data == coord_array[new_coord_name].data):
                is_value_index_in_data = True
                if verbose:
                    message = f'{new_coord_name} was not added in the ' \
                              f'xarray object since {data_coord_key}' \
                              f'already contains the same index data.'
                    print(message)
                break

    if not is_value_index_in_data or not add_only_if_necessary:
        data = data.assign_coords({new_coord_name: coord_array[new_coord_name]})

        if swap_dim:
            data = data.swap_dims({dim_name: new_coord_name})
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UnitStrippedWarning)
                data = data.sortby(new_coord_name)

    return data


def add_label_modifier_xarray(
        xarray_data: XRObject,
        label_modifier: str,
) -> XRObject:
    """
    Creates/Updates an xarray object attribute with key-name `label_modifier`.
    This will be appended to the axis label of quick plots through the use of `quick_plot_label`.

    Parameters
    ----------
    xarray_data : xr.Dataset | xr.DataArray
        The dataset or data array to append the `label_modifier` to.
    label_modifier: str
        The string that will be added to the plot labels (e.g. integrated, normalized etc.)
    Returns
    -------
    xr.Dataset | xr.DataArray
    The data array with an updated `label_modifier` in the xarray object attributes.

    """
    xarray_data.attrs.setdefault('label_modifier', [])
    xarray_data.attrs['label_modifier'] = list(xarray_data.attrs['label_modifier']) + [label_modifier]
    if isinstance(xarray_data, xr.Dataset):
        for key in xarray_data.data_vars.keys():
            xarray_data[key].attrs.setdefault('label_modifier', [])
            xarray_data[key].attrs['label_modifier'] = \
                list(xarray_data[key].attrs['label_modifier']) + [label_modifier]

    return xarray_data


def integrate_xarray(
        xarray_data: XRObject,
        start: EnhancedNumeric | None = None,
        end: EnhancedNumeric | None = None,
        coord: str | None = None,
        var: str | None = None,
        mode: str = 'nearest',
) -> XRObject:
    """
    Integrates the area under the curve of the data array/set within a given range.
    Make sure the integration coordinate axis and its dimension must be sorted in ascending order!

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
    mode: str {'nearest', 'interp'}, optional
        - If 'nearest', uses the edge points nearest to the existing dimension-points
        - If 'interp', and if the start or end points are not values in the coordinate data array (x-axis),
        this method uses cubic interpolation to find the y-axis values at the given start or end points.
        Defaults to 'nearest'.

    Returns
    -------
    xr.Dataset | xr.DataArray
        The reduced data array/set of the integrated array values.

    Raises
    ------
    ValueOutOfOptionsError
        If the specified coordinate or variable is not found in the data.

    """
    # TODO: Add warning that integration coordinate axis and its dimension must be sorted in ascending order!

    assert_options(
        str(coord),
        list(xarray_data.coords.keys()) + [str(None)],
        'coord',
        ValueOutOfOptionsError,
    )
    if coord is None:
        # Set coord_array to the last axis (deepest level) if not provided
        coord = list(xarray_data.dims.keys())[-1]

    assert_options(
        str(var),
        list(xarray_data.data_vars.keys()) + [str(None)],
        'var',
        ValueOutOfOptionsError,
    )
    data = None
    if var is None or isinstance(xarray_data, xr.DataArray):
        # Set the target data to the entire dataset
        data = xarray_data.copy(deep=True)
    elif var in xarray_data.data_vars.keys():
        # Set the target data to the specific data array
        data = xarray_data[var].copy(deep=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UnitStrippedWarning)
        data = data.sortby(coord)  # Sort data array/set by the coordinate of interest
    coord_data_array = data[coord]  # Get all coordinate data

    # Get starting point
    if start is None:
        start = coord_data_array.data[0]
    if not isinstance(start, Qty) and isinstance(coord_data_array.data, Qty):
        start = start * coord_data_array.data.u

    # Get ending point
    if end is None:
        end = coord_data_array.data[-1]
    if not isinstance(end, Qty) and isinstance(coord_data_array.data, Qty):
        end = end * coord_data_array.data.u

    if mode == 'interp':
        # get coordinate data
        coord_data_array_in_range = coord_data_array.where(
            ((coord_data_array >= start) & (coord_data_array <= end)), drop=True)
        coord_data = coord_data_array_in_range.data

        # Add starting and ending points to the coordinate data
        if start != coord_data[0]:
            coord_data = np.append(start, coord_data)
        if end != coord_data[-1]:
            coord_data = np.append(coord_data, end)

        dim_array = convert_coord_to_dim(xarray_data[coord], coord_data)
        data_for_integration = data.pint.interp({dim_array.name: dim_array.data}, method='linear', assume_sorted=False)
    else:
        # get nearest start and end points
        start_arg = np.argmin(np.abs(coord_data_array.data - start))
        start = coord_data_array.data[start_arg]
        end_arg = np.argmin(np.abs(coord_data_array.data - end))
        end = coord_data_array.data[end_arg]

        data_for_integration = data.where(
            ((coord_data_array >= start) & (coord_data_array <= end)), drop=True)

    integrated_data = data_for_integration.integrate(coord)

    # Add label modifier to dataset and all potential data arrays
    integrated_data = add_label_modifier_xarray(integrated_data, 'integrated')

    return integrated_data


def get_normalized_xarray(
        xarray_data: XRObject,
        norm_axis_val: EnhancedNumeric | xr.DataArray | np.ndarray | None = None,
        norm_axis: str | None = None,
        norm_var: str | None = None,
        mode: str = 'nearest',
        subtract_min=True
) -> XRObject:
    """
    Get normalized data based on specified parameters.

    Make sure that norm_axis is sorted in ascending order (default when using xarray_data.sortby(norm_axis)).

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
    mode : str, {'nearest', 'linear', 'quadratic', 'cubic'}, optional
        The interpolation data_aggregation_method for finding norm_axis values.
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
    ValueOutOfOptionsError,
        If `norm_axis` or `norm_var` is not found in the data.
        If the data_aggregation_method is not one of the allowed methods.
    """
    # TODO: Add warning that norm axis must be sorted in ascending order!

    assert_options(
        str(norm_axis),
        list(xarray_data.coords.keys()) + [str(None)],
        'norm_axis',
        ValueOutOfOptionsError,
    )
    if norm_axis is None:
        # Set norm_axis to the last axis (deepest level) if not provided
        norm_axis = list(xarray_data.dims.keys())[-1]

    if norm_axis not in xarray_data.dims.keys():
        # If norm_axis is not a dimension but a coordinate,
        # assume it's a coordinate with only one dimension
        norm_dim = xarray_data[norm_axis].dims[0]
    else:
        norm_dim = norm_axis

    assert_options(
        str(norm_var),
        list(xarray_data.data_vars.keys()) + [str(None)],
        'norm_var',
        ValueOutOfOptionsError,
    )
    if norm_var is None:
        # Set norm_var to the first variable if not provided
        norm_var = list(xarray_data.data_vars.keys())[0]

    assert_options(
        mode,
        {'nearest', 'linear', 'quadratic', 'cubic'},
        'mode',
        ValueOutOfOptionsError,
    )

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
    else:  # data_aggregation_method == 'linear', 'quadratic', 'cubic'
        data_norm_axis_units = extract_units(data.coords[norm_axis])
        norm_axis_val = norm_axis_val.pint.to(data_norm_axis_units)
        norm_axis_val = strip_units(norm_axis_val)
        norm_dim_val = convert_coord_to_dim(data.coords[norm_axis], norm_axis_val.data)

    norm_value = data.pint.interp({norm_dim: norm_dim_val}, mode)

    norm_data = data / norm_value
    norm_data.update(data.coords)

    # Add label modifier to dataset and all potential data arrays
    norm_data = add_label_modifier_xarray(norm_data, 'normalized')

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
    # Get label_modifiers (e.g. 'integrated' or 'normalized')
    label_modifier_str = ''
    if 'label_modifier' in xarray_data.attrs:
        label_modifier_str = '_'.join(np.flip(xarray_data.attrs['label_modifier']))
        label_modifier_str = varname_to_title_string(label_modifier_str)

    # Get variable names
    if isinstance(xarray_data, xr.Dataset):
        key_list = list(xarray_data.variables.keys())
    else:  # if xr.DataArray
        key_list = [xarray_data.name] + list(xarray_data.coords.keys())

    # Get Title-Styled Labels
    label_dict = {}
    for key in key_list:  # parse variables
        key = str(key)
        label = varname_to_title_string(key.replace('_per_', '/'), '/')  # get title-styled string

        # Find if label modifier needs to be added
        if isinstance(xarray_data, xr.Dataset):
            add_label_modifier = key in list(xarray_data.data_vars.keys())  # if key is a data variable
        else:
            add_label_modifier = key == xarray_data.name  # if the key is the name of the xr.DataArray
        if add_label_modifier:
            label = label_modifier_str + ' ' + label

        # Get values to check for units
        if key != getattr(xarray_data, 'name', None):
            values = xarray_data[key].data
        else:
            values = xarray_data.data

        # Get variable's units, if they exist.
        if isinstance(values, Qty):
            unit_str = f'[{values.units:~P}]'
            if unit_str == '[]':
                unit_str = '[A.U.]'
            label = label + f' {unit_str}'

        label_dict[key] = label

    return label_dict


if HAS_VISUALIZATION_DEP:
    def quick_plot_xarray(
            xarray_data: xr.Dataset | xr.DataArray,
            var: str | None = None,
            coord1: str | None = None,
            coord2: str | None = None,
            var_units: str | None = None,
            coord1_units: str | None = None,
            coord2_units: str | None = None,
            plot_method: str | None = None,
            **plot_kwargs
    ) -> PLTArtist:
        """
        Create a quick plot of an xarray data array/set.

        Parameters
        ----------
        xarray_data : xr.Dataset | xr.DataArray
            Either a dataset with a variable name `var` or a data array.
        var : str | None, optional
            In the case of a dataset given in `xarray_data`, it is the name of the data variable to use
            as the y-axis for 2D data and z-axis for 3D data. If None, the first variable in the dataset is used.
            If the xarray_data is a DataArray, `var` is ignored.
        coord1 : str | None, optional
            Name of the coordinate/dimension to use as the x-axis. If None, defaults to deepest dimension.
            Must be 1D coordinate.
        coord2 : str | None, optional
            Name of the coordinate/dimension to use as the y-axis (only for 3D data).
            If None, defaults to deepest dimension, after coord1-related dimension is removed.
            Must be 1D coordinate.
        var_units : str | None, optional
            The units to change the plotted variable to, if applicable. If None, defaults to predefined units.
        coord1_units : str | None, optional
            The units to change the x-axis to, if applicable. If None, defaults to predefined units.
        coord2_units : str | None, optional
            The units to change the y-axis to (only for 3D data), if applicable. If None, defaults to predefined units.
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
        ValueOutOfOptionsError
            - If var not in available variables.
            - If coord1 or coord2 not in available coordinates.
            - If plot_method not in available plot methods.
        ValueOutOfBoundsError
            - If coord1 or coord2 arrays are not 1D.
            - If var array is not either 1D or 2D.
        ValueError
            - If coord1 and coord2 arrays are dependent on the same dimension.
        """
        if isinstance(xarray_data, xr.Dataset):
            if var is None:  # Set var to the first variable if not provided
                var = list(xarray_data.data_vars.keys())[0]
            assert_options(str(var), set(xarray_data.variables.keys()), 'var', ValueOutOfOptionsError)
            xarray_data = xarray_data[var]

        # Define the data
        data = xarray_data.copy(deep=True)

        # Check coordinate input validity
        assert_options(str(coord1), list(xarray_data.coords.keys()) + [str(None)], 'coord1', ValueOutOfOptionsError)
        assert_options(str(coord2), list(xarray_data.coords.keys()) + [str(None)], 'coord2', ValueOutOfOptionsError)

        if coord1 is not None:
            assert_options(len(data[coord1].dims), {1}, 'len(data[coord1].dims)', ArrayDimensionNumberError)
        if coord2 is not None:
            assert_options(len(data[coord2].dims), {1}, 'len(data[coord2].dims)', ArrayDimensionNumberError)
        if coord1 is not None and coord2 is not None:
            if data[coord1].dims[0] == data[coord2].dims[0]:
                raise ValueError('coord1 and coord2 must not depend on the same dimension.')

        # Get amount of data dimensions
        ndims = len(data.dims)
        assert_options(ndims, {1, 2}, 'len(data.dims)', ArrayDimensionNumberError)

        # Check the user input method is correct, or set default methods
        if plot_method is not None:
            available_plot_methods = [method for method in dir(data.plot) if not method.startswith('_')]
            if ndims == 2:
                available_plot_methods += ['linecollection']
            assert_options(plot_method, available_plot_methods, 'plot_method', ValueOutOfOptionsError)

        else:
            # set default plot_method
            if ndims == 1:
                plot_method = 'line'
            elif ndims == 2:
                plot_method = 'pcolormesh'

        # set default coords
        dim_names = [None] + list(data.dims)
        if coord1 is None and coord2 is None:
            coord1 = dim_names[-1]
            coord2 = dim_names[-2]
        elif coord1 is not None and coord2 is None:
            dim1 = data[coord1].dims[0]
            other_dim_names = dim_names.copy()
            other_dim_names.remove(dim1)
            coord2 = other_dim_names[-1]
        elif coord2 is not None and coord1 is None:
            dim2 = data[coord2].dims[0]
            other_dim_names = dim_names.copy()
            other_dim_names.remove(dim2)
            coord1 = other_dim_names[-1]

        # Make own method (-ish)
        set_colors = False
        if plot_method == 'linecollection':
            plot_method = 'line'
            set_colors = True
            plot_kwargs.setdefault('add_legend', False)

        # Change units if necessary
        if var_units is not None:
            data = data.pint.to({var: var_units})
        if coord1 is not None and coord1_units is not None:
            data = data.pint.to({coord1: coord1_units})
        if coord2 is not None and coord2_units is not None:
            data = data.pint.to({coord2: coord2_units})

        # Get plot function with either the default method, or the user's preferred method
        plot = getattr(data.plot, plot_method) if plot_method is not None else data.plot

        # set plot coordinate keys for x, y, and hue
        plot_kwargs['x'] = coord1
        if plot_method == 'line':
            plot_kwargs['hue'] = coord2
        else:
            plot_kwargs['y'] = coord2

        # Set some default kwargs if applicable
        plot_args = getfullargspec(getattr(dataarray_plot, plot_method)).kwonlyargs
        default_kwargs = {'center': False, 'cmap': 'Spectral', 'norm': None}
        for key in default_kwargs.keys():
            if key in plot_kwargs.keys():
                default_kwargs[key] = plot_kwargs.pop(key)
            if key in plot_args:
                plot_kwargs.setdefault(key, default_kwargs[key])

        # Plot data!
        result = plot(**plot_kwargs)

        ax: plt.Axes = result[0].axes if isinstance(result, typing.Sequence) else result.axes

        if set_colors:
            # set some default plot keyword arguments specific to 'linecollection' plot_method
            for key, value in default_kwargs.items():
                plot_kwargs.setdefault(key, value)

            cmap_args = getfullargspec(set_linecollection_cmap).args
            cmap_args += getfullargspec(set_linecollection_cmap).kwonlyargs
            cmap_kwargs = {a: plot_kwargs[a] for a in plot_kwargs if a in cmap_args}
            cbar = set_linecollection_cmap(xarray_data=data, ax=ax, lines=result, **cmap_kwargs)
        else:
            cbar = None

        # Set axis labels if necessary
        if plot_kwargs.get('add_labels', True):
            quick_plot_labels = get_quick_plot_labels(data)

            x_axis = re.sub(r'([ \n])\[[^)]*]', '', ax.get_xlabel())
            x_label = quick_plot_labels[x_axis]
            ax.set_xlabel(x_label)

            y_axis = re.sub(r'([ \n])\[[^)]*]', '', ax.get_ylabel())
            y_label = quick_plot_labels[y_axis]
            ax.set_ylabel(y_label)

            if hasattr(ax, 'get_zlabel'):
                z_axis = re.sub(r'([ \n])\[[^)]*]', '', ax.get_zlabel())
                z_label = quick_plot_labels[z_axis]
                ax.set_zlabel(z_label)

            cbar = getattr(result, 'colorbar', cbar)
            if cbar is not None:
                cbar_ax: plt.Axes = cbar.ax
                color_axis = re.sub(r'([ \n])\[[^)]*]', '', cbar_ax.get_ylabel())
                color_label = quick_plot_labels[color_axis]
                cbar_ax.set_ylabel(color_label)

        return result


    def set_linecollection_cmap(
            xarray_data: xr.Dataset | xr.DataArray,
            hue: str,
            lines: list[plt.Line2D],
            ax: plt.Axes,
            cmap: str | mpl.colors.Colormap | None = None,
            vmin: EnhancedNumeric | None = None,
            vmax: EnhancedNumeric | None = None,
            norm: mpl.colors.Normalize = None,
            extend: ExtendOptions = 'neither',
            levels: Sequence = None,
            center: float | bool = False,
            robust: bool = False,
            colors: str | None = None,
            add_colorbar: bool = True,
            cbar_ax: plt.Axes = None,
            cbar_kwargs: dict[str, Any] | None = None,
    ):
        """
        Parameters
        ----------
        xarray_data
        hue
        lines: list[plt.Line2D]
            A list of Line2D objects.
        ax: plt.Axes
            The axes of the line plot
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
        vmin : pint.Quantity, float or None, optional
            Lower value to anchor the colormap, otherwise it is inferred from the
            data and other keyword arguments. When a diverging dataset is inferred,
            setting `vmin` or `vmax` will fix the other by symmetry around
            ``center``. Setting both values prevents use of a diverging colormap.
            If discrete levels are provided as an explicit list, both of these
            values are ignored.
        vmax : pint.Quantity, float or None, optional
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
        add_colorbar: bool
            If ``True``, the colorbar will appear in the figure.
        cbar_ax : matplotlib axes object, optional
            Axes in which to draw the colorbar.
        cbar_kwargs : dict, optional
            Dictionary of keyword arguments to pass to the colorbar
            (see :meth:`matplotlib:matplotlib.figure.Figure.colorbar`).

        Raises
        ------
        ValueError
            If both `cmap` and `colors` are defined.
        """

        if cmap and colors:
            raise ValueError("Can't specify both cmap and colors.")

        cbar_kwargs = {} if cbar_kwargs is None else dict(cbar_kwargs)

        plot_data = xarray_data[hue].data
        plot_data_magnitude = plot_data.m

        if vmin is not None and isinstance(plot_data, Qty):
            if isinstance(vmin, Qty):
                vmin.m_as(plot_data.u)
        if vmax is not None and isinstance(plot_data, Qty):
            if isinstance(vmax, Qty):
                vmax.m_as(plot_data.u)

        cmap_params = _determine_cmap_params(
            plot_data=plot_data_magnitude,
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
        colors = cm.to_rgba(plot_data_magnitude)
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
