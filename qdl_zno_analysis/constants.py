"""
This module contains the constants used in throughout this package.
"""

from typing import Iterable, List

import numpy as np
from scipy.constants import physical_constants

from qdl_zno_analysis import ureg, Qty
from qdl_zno_analysis.errors import MethodInputError
from qdl_zno_analysis.external_data import df_from_ext_data


def _get_ureg_unit_prefixes() -> List:
    """
    Returns a list of unique prefix symbols used by the pint.UnitRegistry instance.

    Returns
    -------
    List:
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


# @ureg.with_context('spectroscopy')
# @ureg.wraps(None, ('nm', None), False)
def get_n_air(value: Qty | float | Iterable, medium='air') -> float | np.ndarray:
    """
    Get the refractive index of air or vacuum.
    The refractive index of air or vacuum is interpolated from the database of refractive indices.
    The database contains data for vacuum wavelengths from 0.23 to 1.69 um.

    Parameters
    ----------
    value:  float | Iterable | pint.Quantity
        Value(s) of the refractive index to be calculated. If no units are provided, 'nm' are assumed.
    medium: str
        'air' or 'vacuum'. The default is 'air'. Only applies for length input (which is the default).

    Returns
    -------
    float | np.ndarray
        Refractive index of air in the provided values. If a single value was provided, a float is returned.
        If multiple values were provided, a numpy array is returned.

    Raises
    ------
    MethodInputError
        If medium is not 'air' or 'vacuum'.

    Notes
    -----
    The database was downloaded from `[here](https://refractiveindex.info/?shelf=other&book=air&page=Ciddor)`.

    Examples
    --------
    >>> get_n_air(369 * ureg.nm)
    1.0002846755361399
    >>> get_n_air(369 * ureg.nm, medium='vacuum')
    1.0002846829178083
    >>> get_n_air([3.35, 3.36] * ureg.eV, medium='vacuum')
    array([1.00027316, 1.00027316])

    """
    value: Qty = value if isinstance(value, Qty) else Qty(value, 'nm')

    allowed_media = ['vacuum', 'air']
    if medium not in allowed_media:
        raise MethodInputError('medium', medium, allowed_media, 'get_n_air')

    n_air_database = df_from_ext_data("refractive_index_air.csv")

    if value.check('[energy]'):
        value = value.m_as('eV')
        value_column = 'Energy (eV)'
    elif value.check('[frequency]'):
        value = value.m_as('THz')
        value_column = 'Frequency (THz)'
    else:  # if [length], it will proceed, if not, Pint will raise a unit dimensionality error
        value = value.m_as('um')
        value_column = 'Wavelength Air (um)' if medium == 'air' else 'Wavelength Vacuum (um)'

    return np.interp(value, n_air_database[value_column], n_air_database['Refractive Index'])
