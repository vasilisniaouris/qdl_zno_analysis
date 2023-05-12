"""
Module containing various utility functions for this package.
"""
import dataclasses
import re
import types
import typing
from dataclasses import dataclass, fields
from operator import attrgetter
from typing import List, Dict, Type, Iterable

import numpy as np
import pint

from qdl_zno_analysis import ureg, Qty
from qdl_zno_analysis.constants import default_units
from qdl_zno_analysis.typevars import EnhancedNumeric


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
    else:
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
        first_value = tuple(first_value) if isinstance(first_value, Iterable) else first_value  # make it a tuple.
        value_list = [first_value]

        for d in list_of_dicts[1:]:
            test_element = getattr(d[key], 'magnitude', d[key])  # if Qty get magnitude, else get value.
            test_element = tuple(test_element) if isinstance(test_element, Iterable) else test_element

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
