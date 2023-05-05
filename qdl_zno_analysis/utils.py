"""
Module containing various utility functions for this package.
"""

import re
import types
import typing
from typing import List, Dict, Type, Iterable

import numpy as np
import pint

from qdl_zno_analysis import ureg, Qty


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

