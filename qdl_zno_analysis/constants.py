"""
This module contains the constants used in throughout this package.
"""
from typing import Iterable

import numpy as np
from pandas import DataFrame
from scipy.constants import physical_constants

from qdl_zno_analysis import ureg, Qty
from qdl_zno_analysis.errors import assert_options, ValueOutOfOptionsError, assert_unit_on_value
from qdl_zno_analysis.external_data import df_from_ext_data


def _get_ureg_unit_prefixes() -> list:
    """
    Returns a list of unique prefix symbols used by the pint.UnitRegistry instance.

    Returns
    -------
    list:
        A list of unique prefix symbols used by the pint.UnitRegistry instance.

    Examples
    --------
    >>> _get_ureg_unit_prefixes()
    ['E', 'Ei', 'G', 'Gi', 'Ki', 'M', 'Mi', 'P', 'Pi', 'T', 'Ti', 'Y', 'Yi', 'Z', 'Zi', 'a', 'atto', 'c', 'centi', 'd', 'da', 'deca', 'deci', 'deka', 'demi', 'exa', 'exbi', 'f', 'femto', 'gibi', 'giga', 'h', 'hecto', 'k', 'kibi', 'kilo', 'm', 'mebi', 'mega', 'micro', 'milli', 'n', 'nano', 'p', 'pebi', 'peta', 'pico', 'semi', 'sesqui', 'tebi', 'tera', 'u', 'y', 'yobi', 'yocto', 'yotta', 'z', 'zebi', 'zepto', 'zetta', 'µ', 'μ']
    """
    ureg_prefix_symbols = [value.defined_symbol for _, value in ureg._prefixes.items()]  # get all the symbols
    ureg_prefix_symbols += [value.name for _, value in ureg._prefixes.items()]  # get the full names as well
    ureg_prefix_symbols += [alias for _, value in ureg._prefixes.items() for alias in value.aliases
                            if len(value.aliases) > 0]  # getting all aliases (e.g. μ and u both stand for micro)
    filtered_list = list(filter(None, ureg_prefix_symbols))
    unique_prefix_symbols = list(np.unique(filtered_list))

    return unique_prefix_symbols


ureg_unit_prefixes = _get_ureg_unit_prefixes()
""" List of unique prefix symbols used by the pint.UnitRegistry instance. """

default_units = {
    'dimensionless': {'main': '', 'core': ''},
    'length': {'main': 'nm', 'core': 'm'},
    'time': {'main': 'us', 'core': 's'},
    'energy': {'main': 'eV', 'core': 'eV'},
    'frequency': {'main': 'GHz', 'core': 'Hz'},
    'power': {'main': 'uW', 'core': 'W'},
    'temperature': {'main': 'K', 'core': 'K'},
    'magnetic field': {'main': 'T', 'core': 'T'},
    'counts': {'main': 'count', 'core': 'count'},
    'counts/time': {'main': 'count/s', 'core': 'count/s'},
    'counts/time/power': {'main': 'count/s/uW', 'core': 'count/s/W'},
    'voltage': {'main': 'mV', 'core': 'V'},
    'current': {'main': 'mA', 'core': 'A'},
    'angle': {'main': 'deg', 'core': 'deg'},
    }
""" Dictionary containing the default physical types used in this package.
    They can be changed by the user by directly accessing this dictionary. """


c: Qty = Qty(*physical_constants['speed of light in vacuum'][:2]).to('nm*GHz')
""" Speed of light in vacuum in nanometers * gigahertz. """

h: Qty = Qty(*physical_constants['Planck constant'][:2]).to('eV/GHz')
""" Planck constant in eV / gigahertz. """

hc: Qty = (h * c).to('eV*nm')
""" Planck constant times speed of light in vacuum in eV * nanometers. """


_refractive_index_databases = {
    'air': 'refractive_index_air.csv',
}
""" 
Dictionary containing the refractive index databases. The keys are the names of the mediums and the values are the 
filenames of the databases. 

Keys
----
air
    The air database contains data for vacuum wavelengths from 0.23 to 1.69 um. 
"""


def get_n_air(wfe_value: Qty | float | Iterable, input_medium='air') -> float | np.ndarray:
    """
    Get the refractive index of air for a given wavelength, frequency or energy.
    The refractive index is interpolated from the database of refractive indices.

    Parameters
    ----------
    wfe_value:  float | Iterable | pint.Quantity
        Value(s) of the wavelength to be calculated. If no units are provided, 'nm' are assumed.
    input_medium: str
        The medium the `wfe_value` is given in, e.g. 'air', 'vacuum' or other user-defined material.
        Defaults to 'air'.

    Returns
    -------
    float | np.ndarray
        Refractive index of output medium in the provided wfe_values. If a single value was provided, a float is
        returned. If multiple values were provided, a numpy array is returned.

    Raises
    ------
    ValueOutOfOptionsError
        If input medium is not in list of allowed media.

    Examples
    --------
    >>> get_n_air(369 * ureg.nm)
    1.0002846755358596
    >>> get_n_air(369 * ureg.nm, 'vacuum')
    1.0002846829178083
    >>> get_n_air([3.35, 3.36] * ureg.eV, 'vacuum')
    array([1.00028461, 1.00028468])
    """
    return get_refractive_index(wfe_value, output_medium='air', input_medium=input_medium)


def get_refractive_index(wfe_value: Qty | float | Iterable, output_medium='air',
                         input_medium=None) -> float | np.ndarray:
    """
    Get the refractive index of air, vacuum or other materials if properly defined in the _refractive_index_databases.
    The refractive index is interpolated from the database of refractive indices.

    Parameters
    ----------
    wfe_value:  float | Iterable | pint.Quantity
        Value(s) of the wavelength to be calculated. If no units are provided, 'nm' are assumed.
    output_medium: str
        The medium for the requested refractive index, e.g. 'air', 'vacuum' or other user-defined material.
        Defaults to 'air'.
    input_medium: str
        The medium the `wfe_value` is given in, e.g. 'air', 'vacuum' or other user-defined material.
        Defaults to the `output_medium` value. Only used for wavelength input (which is the default).

    Returns
    -------
    float | np.ndarray
        Refractive index of output medium in the provided wfe_values. If a single value was provided, a float is
        returned. If multiple values were provided, a numpy array is returned.

    Raises
    ------
    ValueOutOfOptionsError
        If input or output media are not in list of allowed media.
    IncompatibleUnitError
        If wfe_value not in units compatible with the spectroscopy content.

    Examples
    --------
    >>> get_refractive_index(369 * ureg.nm)
    1.0002846755358596
    >>> get_refractive_index(369 * ureg.nm, 'air', 'vacuum')
    1.0002846829178083
    >>> get_refractive_index([3.35, 3.36] * ureg.eV, 'air', 'vacuum')
    array([1.00028461, 1.00028468])

    """
    wfe_value: Qty = wfe_value if isinstance(wfe_value, Qty) else Qty(wfe_value, default_units['length']['main'])
    input_medium: str = input_medium if input_medium is not None else output_medium

    _allowed_mediums = list(_refractive_index_databases.keys()) + ['vacuum']

    assert_unit_on_value(wfe_value, 'nm', 'spectroscopy')
    assert_options(output_medium, _allowed_mediums, 'output_medium', ValueOutOfOptionsError)
    assert_options(input_medium, _allowed_mediums, 'input_medium', ValueOutOfOptionsError)

    def _get_refractive_index(value: Qty | float | Iterable, medium: str, is_input_vac=True):
        if medium == 'vacuum':
            return 1.

        n_database = df_from_ext_data(_refractive_index_databases[medium])

        if value.check('[energy]'):
            with ureg.context('spectroscopy'):
                value = value.m_as('um')
            value_column = 'Wavelength Vacuum (um)'
        elif value.check('[frequency]'):
            with ureg.context('spectroscopy'):
                value = value.m_as('um')
            value_column = 'Wavelength Vacuum (um)'
        else:  # if [length], it will proceed, if not, Pint will raise a unit dimensionality error
            value = value.m_as('um')
            value_column = 'Wavelength Vacuum (um)' if is_input_vac else 'Wavelength Medium (um)'

        n_database = n_database.sort_values(by=value_column, ignore_index=True)

        return np.interp(value, np.asarray(n_database[value_column]), np.asarray(n_database['Refractive Index']))

    input_frequency = wfe_value.to(
        'THz', 'sp', n=_get_refractive_index(wfe_value, input_medium, is_input_vac=bool(input_medium == 'vacuum')))

    return _get_refractive_index(input_frequency, output_medium)


def _get_extended_refractive_index_database(original_file) -> DataFrame:
    """
    Add 'Wavelength Medium (um)', 'Frequency (THz)' and Energy (eV)' column to refractive index csv file that already
    contains two columns: 'Wavelength Vacuum (um)' and 'Refractive Index'.

    Parameters
    ----------
    original_file: str
        The filename of the original refractive index database.

    Returns
    -------
    DataFrame
        The extended refractive index database.
    """
    database: DataFrame = df_from_ext_data(original_file)
    wavelength_vacuum = database['Wavelength Vacuum (um)']
    refractive_index = database['Refractive Index']
    database['Wavelength Medium (um)'] = wavelength_vacuum / refractive_index
    wavelength_vacuum = Qty(wavelength_vacuum.tolist(), 'um')
    database['Energy (eV)'] = wavelength_vacuum.to('eV', 'sp', n=1).m
    database['Frequency (THz)'] = wavelength_vacuum.to('THz', 'sp', n=1).m

    return database
