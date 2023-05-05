"""
This module provides functionality for parsing filename strings containing subsets separated by predefined separator characters.

Notes
-----
- This module is meant to be used internally by other modules in the qdl_zno_analysis package, and should not be
  imported or used directly by external code.
- The `_secondary_filename_separators` attribute is meant to be used internally by the module, and should not be modified
  by external code, unless you know what you are doing. Changing the value of this attribute will change the expected
  filename separator convention.
"""


import re
import warnings
from typing import List, Dict, Tuple, Any

from pathlib import Path

from qdl_zno_analysis import ureg, Qty
from qdl_zno_analysis.constants import ureg_unit_prefixes, default_units
from qdl_zno_analysis.errors import MethodInputError, InvalidUnitStringError, InvalidDerivedUnitStringError, \
    UnsupportedUnitError, IncompatibleUnitWarning
from qdl_zno_analysis.typevars import AnyString
from qdl_zno_analysis.utils import is_unit_valid

_secondary_filename_separators = ['-', ';', ',']
"""
A list of secondary separators for parsing individual subsets in a filename string.
It is used in the `parse_subset_items` function to parse the subsets in a filename string.

This list of separators is used to separate different sub-parameters in the filename. 
The function `parse_subset_items` checks if any of the secondary separators are present
in the subset_str string, and if so, uses them to split the subsets into smaller chunks 
that can be parsed recursively.

This attribute is meant to be used internally by the module, and should not be modified
by external code, unless you know what you are doing. Changing the value of this 
attribute will change the expected filename separator convention.
"""


def _get_next_separator(separator) -> str | None:
    """
    Returns the next separator in the list of valid separators after `separator`, if it exists.
    The list of valid seperators is in `_secondary_filename_separators`.

    Parameters
    ----------
    separator : str
        The separator character to be used in the subset_str string.

    Returns
    -------
    str or None
        The next separator in the list of valid separators after `separator`, or None if `separator`
        is not in the list of valid separators.

    Examples
    --------
    >>> _get_next_separator('-')
    ';'

    >>> _get_next_separator(',')

    """

    separators = _secondary_filename_separators
    if separator in separators:
        sep_idx = separators.index(separator)
        next_separator = separators[sep_idx + 1] if sep_idx < len(separators) - 1 else None
        return next_separator

    return None


def _get_filename_subset_string_format_regex(separator, sep_below_str=None) -> Tuple[str, str]:
    """
    Returns a tuple containing the regular expression (regex) patterns for matching subset_str strings in a filename.

    Parameters
    ----------
    separator : str
        The separator character that is used subset_str string. If not in the list of valid
        separators `_secondary_filename_separators`, a MethodInputError will be raised.
    sep_below_str : str, optional
        A string representing all the separators that come after `separator` in the list of valid
        separators. If not provided, the function will determine `sep_below_str` from `separator` and
        `_secondary_filename_separators`.

    Returns
    -------
    Tuple[str, str]
        A tuple containing two regex patterns: the first pattern matches a single key-value pair in
        a subset_str string, and the second pattern tests if a string contains one or more valid subset_str
        strings.

    Examples
    --------
    >>> _get_filename_subset_string_format_regex('-')
    ('([a-zA-Z0-9]+)~([a-zA-Z0-9~;,]+)-?', '^(([a-zA-Z0-9]+)~([a-zA-Z0-9~;,]+)-?)+$')

    """

    separators = _secondary_filename_separators
    if sep_below_str is None:
        if separator not in separators:
            raise MethodInputError('separator', separator, separators, '_get_filename_subset_string_format_regex')
        else:
            sep_idx = separators.index(separator)
            separators_below = separators[sep_idx + 1:]
            sep_below_str = ''.join(separators_below)

    filename_subset_string_format_regex = f"([a-zA-Z0-9]+)~([a-zA-Z0-9~{sep_below_str}]+){separator}?"
    filename_subset_string_format_test_regex = f"^({filename_subset_string_format_regex})+$"

    return filename_subset_string_format_regex, filename_subset_string_format_test_regex


def parse_value(string: str) -> str:
    """
    Parses a string representing a numerical value with an optional unit and sign and returns it
    in a standardized format.

    Parameters
    ----------
    string: str
        A string representing a numerical value with an optional unit and sign. The value can be a positive ('p') or
        negative ('n') number and can contain a decimal point represented by the letter 'p' followed by one or more
        digits. The unit can be any combination of characters.

    Returns
    -------
    str
        A standardized string representing the numerical value with an optional unit. The sign is represented by a
        '-' for negative, and '' for positive. The decimal point is represented by a '.', and the value is NOT separated
        from the unit by any spaces. If the input string is invalid, it returns the input string.

    Raises
    ------
    UserWarning
        If the input string does not match the expected format, a warning is raised and None is returned.

    Notes
    -----
    This method is based on the regular expression "^([pn]?)([0-9]+)(p[0-9]+|)(.*)$".
    The regex match-groups are:
        - Sign: 0 or 1 occurrences of p (for positive '+') or n (for megative '-'). Matches empty string.
        - Integer part: 1 or more digits.
        - Decimal part: 0 or 1 occurrences of p (for decimal point '.') followed by 1 or more digits. Matches empty string.
        - Unit: any amount (0 included) of any characters.
    If the groups do not match the expected format, a warning is raised and None is returned.
    If the groups match the expected format, the groups are converted to standardized strings and returned.
    In the standardized string:
        - the sign is represented by a '-' for negative, and '' for positive.
        - the decimal point is represented by a '.', and the value is
        - the unit is represented by any combination of characters, same as the input.

    Examples
    --------
    >>> parse_value("730p5nm")
    '730.5nm'
    >>> parse_value("n730p35nm")
    '-730.35nm'
    >>> parse_value("p730n")
    '730n'
    >>> parse_value("730p")
    '730p'
    >>> parse_value("730")
    '730'
    >>> parse_value("n730")
    '-730'
    >>> parse_value("n730p")
    '-730p'
    >>> parse_value("n730p5")
    '-730.5'
    >>> parse_value("Potato")
    'Potato'
    """

    regex_sign = "([pn]?)"  # 0 or 1 occurrences of p or n. matches empty string
    regex_int_part = "([0-9]+)"  # 1 or more digits.
    regex_decimal_part = "(p[0-9]+|)?"  # 0 or 1 occurrences of p followed by 1 or more digits. will match empty string.
    unit_part = "(.*)"  # any amount (0 included) of any characters.
    regex = '^' + regex_sign + regex_int_part + regex_decimal_part + unit_part + '$'
    matches = re.match(regex, string)

    if matches is None:
        return string
    if not matches.group(0) == string:  # if only part of the input string matches the regex
        return string

    sign = matches.group(1).replace('p', '').replace('n', '-')
    int_part = matches.group(2)
    decimal_part = matches.group(3).replace('p', '.')
    unit_part = matches.group(4)
    modified_string = f'{sign}{int_part}{decimal_part}{unit_part}'

    return modified_string


def parse_subset_str(subset_str: str, separator: str = '-') -> Dict | List | str:
    """
    Parse a subset of a filename string into a nested dictionary or a list of values.
    This function calls recursively on itself to read the sunset string in different separator levels.
    Other separators in descending level order are in the module attribute `_secondary_filename_separators`.

    Parameters
    ----------
    subset_str : str
        The subset string to be parsed.
    separator : str, optional
        The separator character used to split the string into substrings, by default '-'.

    Returns
    -------
    Dict | List | str
        If the subset string has a dictionary-like format (e.g. 'Wvl~From~369p1n;To~340p1n;Step~1-Pwr~10uW'),
        it returns a nested dictionary where each key is a header and each value is the parsed result of its
        corresponding substring. If the subset string has a list-like format
        (e.g. 'Matisse-From~369p1n;To~340p1n;Step~1-10uW'), it returns a list of parsed
        values. If the subset string has a single value, it returns the parsed value as a string.

    Examples
    --------
    >>> parse_subset_str('Wvl~From~369p1n;To~340p1n;Step~1-Pwr~10uW')
    {'Wvl': {'From': '369.1n', 'To': '340.1n', 'Step': '1'}, 'Pwr': '10uW'}

    >>> parse_subset_str('Matisse-From~369p1n;To~340p1n;Step~1-10uW')
    ['Matisse', {'From': '369.1n', 'To': '340.1n', 'Step': '1'}, '10uW']

    >>> parse_subset_str('Potato')
    'Potato'
    """

    next_separator = _get_next_separator(separator)
    if separator is not None:
        filename_subset_string_format_regex, filename_subset_string_format_test_regex = \
            _get_filename_subset_string_format_regex(separator)
        match_subset_string_format = re.match(filename_subset_string_format_test_regex, subset_str)

        if match_subset_string_format:
            finds = re.findall(filename_subset_string_format_regex, subset_str)
            return {header: parse_subset_str(value, next_separator) for header, value in finds}

    finds = subset_str.split(separator)
    if len(finds) > 1:
        return [parse_subset_str(find, next_separator) for find in finds]
    else:
        return parse_value(finds[0])


def parse_filename(filename: AnyString) -> Dict[str, Any]:
    """
    Parse a filename string into a dictionary of key-value pairs.

    This function takes a filename string and parses it into a dictionary of key-value pairs
    using the underscore character as a primary separator + the separators defined in the
    `_secondary_filename_separators`. Each subset_str in the filename is processed by the `parse_subset_str` function.
    The resulting dictionary contains all the key-value pairs extracted from the filename subsets.

    Parameters
    ----------
    filename : str | Path
        The input filename string to parse.

    Returns
    -------
    Dict[str, str | List[str] | Dict[str, str | Dict[str | str]]]
        A dictionary containing information parsed from the filename. The dictionary may include
        the following keys: 'FNo', 'Misused', and other keys derived from the subset_str strings in the
        filename. The first numeric string will be parsed as the file number 'FNo'
        and will be stored in the 'FNo' key of the resulting dictionary.
        The 'Misused' key contains a list of substrings that could not be parsed. Other keys represent the headers
        derived from the subset_str strings, and their values are either a string, a list of strings, or a
        nested dictionary with key-value pairs representing different types of information.

    Examples
    --------
    >>> parse_filename('003_Lsr~Wvl~From~369p1n;To~340p1n;Step~1-Pwr~10uW_Tmp~6p1K_MgF~5p1_apples_potatoes')
    {'FNo': '003', 'Lsr': {'Wvl': {'From': '369.1n', 'To': '340.1n', 'Step': '1'}, 'Pwr': '10uW'}, 'Tmp': '6.1K', 'MgF': '5.1', 'Misused': ['apples', 'potatoes']}

    >>> parse_filename('003_Lsr~Matisse-From~369p1n;To~340p1n;Step~1-10uW_Tmp~6p1K_MgF~5p1_apples_potatoes')
    {'FNo': '003', 'Lsr': ['Matisse', {'From': '369.1n', 'To': '340.1n', 'Step': '1'}, '10uW'], 'Tmp': '6.1K', 'MgF': '5.1', 'Misused': ['apples', 'potatoes']}

    """

    filename_stem = Path(filename).stem
    subset_strings = filename_stem.split('_')

    info_dict = {}
    user_misuse = []

    for subset_str in subset_strings:
        filename_subset_string_format_regex, filename_subset_string_format_test_regex = \
            _get_filename_subset_string_format_regex('', ''.join(reversed(_secondary_filename_separators)))
        match_subset_string_format = re.match(filename_subset_string_format_test_regex, subset_str)

        if match_subset_string_format:
            subset_header = match_subset_string_format.group(2)
            subset_value = match_subset_string_format.group(3)
            parsed_subset_value = parse_subset_str(subset_value, '-')
            info_dict[subset_header] = parsed_subset_value
        else:
            parsed_subset = parse_value(subset_str)
            if isinstance(parsed_subset, str):
                if parsed_subset.isnumeric() and 'FNo' not in info_dict.keys():
                    info_dict['FNo'] = parsed_subset
                else:
                    user_misuse.append(parsed_subset)

    if len(user_misuse) > 0:
        info_dict['Misused'] = user_misuse

    return info_dict


def get_unit_from_str(unit_str: str, primary_physical_type: str = None, context=None):
    """
    Get a unit from a string.

    This function takes a string containing a unit and returns a unit object.
    The string can be either a full unit (e.g. 'um') or the prefix of the default core unit
    (e.g. for length with core unit 'm') a string of 'u' will return 'um'), or nothing, in which case it the unit will
    become the default unit for the primary physical type (e.g. if length, the default unit is 'nm').
    Context allows for unit conversion between units not usually compatible (e.g. in 'spectroscopy' context, length
    can be converted to frequency or energy). If the primary physical type is None, then the unit is returned as is.

    Caveat: The following units are also valid unit prefixes:
    ['G', 'M', 'P', 'T', 'a', 'c', 'd', 'da', 'h', 'k', 'm', 'u', 'µ', 'μ'].
    Since this method checks for prefixes first, it is advisable to not include units at all if they are not prefixed.

    Parameters
    ----------
    unit_str : str
        A string representation of the physical unit.
    primary_physical_type : str, optional
        An optional primary physical type. If provided, the returned unit will be checked for
        compatibility with the default unit of the primary physical type, and a warning will
        be raised if the units are not compatible. If not provided, the unit will be returned
        as is.
    context : str, optional
        An optional context string. If provided, the pint.UnitRegistry will be enabled to use the
        specified context.

    Returns
    -------
    pint.Unit
        A pint.Unit object representing the physical unit.

    Raises
    ------
    InvalidUnitStringError
        If the string is not a valid unit.
    InvalidDerivedUnitStringError
        If the string is a valid unit but not compatible with the primary physical type.
    IncompatibleUnitWarning
        If the string is a valid unit and is not compatible with the primary physical type, a warning
        is raised instead of an error.
    UnsupportedUnitError
        If the unit is not supported.
    """
    if primary_physical_type is None:
        if is_unit_valid(unit_str):
            return ureg.Unit(unit_str)
        else:
            raise InvalidUnitStringError(unit_str)

    if context is not None:
        ureg.enable_contexts(context)

    if len(unit_str) == 0:
        return ureg.Unit(default_units[primary_physical_type]['main'])
    elif unit_str in ureg_unit_prefixes:
        new_unit_str = unit_str + default_units[primary_physical_type]['core']
        if is_unit_valid(new_unit_str):
            return ureg.Unit(new_unit_str)
        else:
            raise InvalidDerivedUnitStringError(unit_str, primary_physical_type, default_units)
    elif is_unit_valid(unit_str):
        unit = ureg.Unit(unit_str)
        if not unit.is_compatible_with(default_units[primary_physical_type]['main']):
            warnings.warn(IncompatibleUnitWarning(unit_str, primary_physical_type, context, default_units))
        return unit
    else:
        raise UnsupportedUnitError(unit_str, primary_physical_type, context)


def parse_string_with_units(s, primary_physical_type, context=None):
    """
    Parse a string containing a number with units.
    The units can be either a full unit (e.g. 'um') or the prefix of the defualt core unit
    (e.g. for length with core unit 'm') a string of 'u' will return 'um'), or nothing, in which case it the unit will
    become the default unit for the primary physical type (e.g. if length, the default unit is 'nm').
    Context allows for unit conversion between units not usually compatible (e.g. in 'spectroscopy' context, length
    can be converted to energy or frequency).

    This function takes a string with units and parses it into a pint.Quantity object.
    The string is parsed by splitting it into a number and a unit string.
    The number is parsed into a float and the unit is parsed into a pint.Unit object.
    The number and unit are then combined into a pint.Quantity object.

    Caveat: The following units are also valid unit prefixes:
    ['G', 'M', 'P', 'T', 'a', 'c', 'd', 'da', 'h', 'k', 'm', 'u', 'µ', 'μ'].
    Since this method checks for prefixes first, it is advisable to not include units at all if they are not prefixed.

    Parameters
    ----------
    s : str
        The input string to parse.
    primary_physical_type : str
        The primary physical type of the quantity. Allowed values are the keys of the `.constants.default_units`
        dictionary
    context : str | None
        The context of the pint.Quantity. The only one we will need is 'sp' or 'spectroscopy' which allows for the
        conversion between length, frequency and energy.

    Returns
    -------
    pint.Quantity | str
        The quantity object created from the input string. If the string does not match the regular expression,
         it returns the input string.

    Examples
    --------
    >>> parse_string_with_units('369.1n', 'length')
    <Quantity(369.1, 'nanometer')>

    """

    match = re.match(r'([-+]?[0-9]*\.?[0-9]+)([a-zA-Z]*)', s)
    if match is None:
        return s
    number_str, unit_str = match.groups()
    number = float(number_str)
    unit = get_unit_from_str(unit_str, primary_physical_type, context)
    return Qty(number, unit)


