"""
This module provides physics-related utilities (e.g. spectroscopy).
"""

from dataclasses import dataclass, field

import numpy as np

from qdl_zno_analysis import Qty, ureg
from qdl_zno_analysis.constants import _refractive_index_databases, get_refractive_index
from qdl_zno_analysis.errors import assert_options, ValueOutOfOptionsError, assert_unit_on_value
from qdl_zno_analysis.typevars import EnhancedNumeric
from qdl_zno_analysis.utils import to_qty_force_units, to_qty, Dataclass

from typing import ClassVar


@dataclass(repr=False)
class WFE(Dataclass):
    """
    A class to hold information for a set of wavelength, frequency, and energy values.
    One of the three needs to be inputted, the others are calculated automatically.

    Parameters
    ----------
    wfe : int | float | pint.Quantity | np.ndarray[int | float] | list[int | float]
        wavelength, frequency or energy input. If not pint.Quantity, defaults to length units.
    input_medium : str, optional
        Input medium, by default 'air'.
    medium : str, optional
        Stored medium, by default 'air'.

    Examples
    --------
    >>> from qdl_zno_analysis import ureg
    >>> WFE(369 * ureg.nm)
    WFE(medium='air', wavelength_medium=<Quantity(369.0, 'nanometer')>, wavelength_vacuum=<Quantity(369.105045, 'nanometer')>, frequency=<Quantity(812214.468, 'gigahertz')>, energy=<Quantity(3.35904914, 'electron_volt')>, refractive_index=1.0002846755358596)
    >>> WFE(369 * ureg.nm).to_dict()
    {'wfe': <Quantity(369, 'nanometer')>, 'input_medium': 'air', 'medium': 'air', 'wavelength_medium': <Quantity(369.0, 'nanometer')>, 'wavelength_vacuum': <Quantity(369.105045, 'nanometer')>, 'frequency': <Quantity(812214.468, 'gigahertz')>, 'energy': <Quantity(3.35904914, 'electron_volt')>, 'refractive_index': 1.0002846755358596}

    Raises
    ------
    ValueOutOfOptionsError
        When the medium related inputs are not included in the default options.
    IncompatibleUnitError
        When input wfe units are not compatible with the spectroscopy unit context (length, frequency, speed).

    """

    wfe: EnhancedNumeric | np.ndarray[int | float] | list[int | float] = field(repr=False)
    """ 
    Wavelength, frequency or energy input. If not pint.Quantity, defaults to length units. 
    Can be either a single value or an assortment of values. 
    """
    input_medium: str = field(repr=False, default='air')
    """ Input medium, by default 'air'. """
    medium: str = 'air'
    """ Stored medium, by default 'air'. """
    diffraction_order: int = 1
    """ Diffraction order, by default 1. Affects output wfe, but not refractive index. """

    wavelength_medium: Qty = field(init=False)
    """ Wavelength in medium specified in input parameters. """
    wavelength_vacuum: Qty = field(init=False)
    """ Wavelength in vacuum. """
    frequency: Qty = field(init=False)
    """ Frequency. """
    energy: Qty = field(init=False)
    """ Energy. """
    refractive_index: float | np.ndarray = field(init=False)
    """ Refractive index in `medium` at the given wavelength/frequency/energy value. """

    _allowed_mediums: ClassVar[list[str]] = list(_refractive_index_databases.keys()) + ['vacuum']
    """ List of allowed mediums. """

    def __post_init__(self):
        self._check_input()
        self.wfe: Qty = to_qty(self.wfe, 'length')
        self.refractive_index = get_refractive_index(self.wfe, self.medium, self.input_medium)
        input_ref_index = get_refractive_index(self.wfe, self.input_medium, self.input_medium)

        self.frequency = to_qty_force_units(self.wfe, 'frequency', 'sp', n=input_ref_index) * self.diffraction_order
        self.energy = to_qty_force_units(self.frequency, 'energy', 'sp')
        self.wavelength_vacuum = to_qty_force_units(self.frequency, 'length', 'sp')
        self.wavelength_medium = to_qty_force_units(self.frequency, 'length', 'sp', n=self.refractive_index)

    def _check_input(self):
        """ Check if the input values are valid. """
        assert_unit_on_value(self.wfe, 'nm', 'spectroscopy')
        assert_options(self.input_medium, self._allowed_mediums, 'input_medium', ValueOutOfOptionsError)
        assert_options(self.medium, self._allowed_mediums, 'medium', ValueOutOfOptionsError)

    @property
    def freq(self):
        """ Return frequency. """
        return self.frequency

    @property
    def eng(self):
        """ Return energy. """
        return self.energy

    @property
    def wvl_mdm(self):
        """ Return wavelength in medium. """
        return self.wavelength_medium

    @property
    def wvl_vac(self):
        """ Return wavelength in vacuum. """
        return self.wavelength_vacuum


@ureg.context('spectroscopy')
def convert_spectroscopic_delta(delta=None, center_v=None, min_v=None, max_v=None, input_medium='air',
                                output_medium='vacuum', output_units='nm', returned_values='auto'):
    """
    Convert a spectroscopic delta from one medium to another and between wavelength, frequency and energy (wfe).

    Parameters
    ----------
    delta : int | float | pint.Quantity, optional
        Bandwidth in wfe, by default None.
    center_v : int | float | pint.Quantity, optional
        Center of bandwidth in wfe, by default None.
    min_v : int | float | pint.Quantity, optional
        Minimum value of bandwidth in wfe, by default None.
    max_v : int | float | pint.Quantity, optional
        Maximum value of bandwidth in wfe, by default None.
    input_medium : str, optional
        Input medium, by default 'air' (only important if input is in length units).
    output_medium : str, optional
        Output medium, by default 'vacuum' (only important if output is in length units).
    output_units : str, optional
        Output units, by default 'nm'.
    returned_values : str, {'auto', 'all'}, optional
        Returned values, by default 'auto'. If 'auto' returns only the values that were provided as input. If 'all',
        returns all four values, delta, center, min and max.

    Returns
    -------
    tuple(pint.Quantity, ...)
        If `return_values` input is 'auto', returns only the converted values for which an input was provided
        (e.g. for delta and center_v input, it returns a tuple of the converted delta and center_v).
        If 'all', returns all four values after the conversion, i.e. delta, center_v, min_v and max_v.

    Raises
    ------
    ValueError
        - If returned_values is not allowed.
        - If less than two inputs were provided.
    ValueOutOfOptionsError
        If input_medium or output_medium is not allowed.
        If return values neither 'all' or 'auto'.

    Notes
    -----
    The input_medium and output_medium must be a medium from the refractive index databases or 'vacuum'.

    Examples
    --------
    >>> from qdl_zno_analysis import ureg
    >>> convert_spectroscopic_delta(delta=1 * ureg.nm, center_v=369 * ureg.nm, input_medium='vacuum', output_medium='air')
    (<Quantity(0.999741314, 'nanometer')>, <Quantity(368.894982, 'nanometer')>)
    >>> convert_spectroscopic_delta(delta=1 * ureg.nm, center_v=369 * ureg.nm, input_medium='vacuum', output_medium='air', output_units='THz')
    (<Quantity(2.20175387, 'terahertz')>, <Quantity(812.445686, 'terahertz')>)
    >>> convert_spectroscopic_delta(delta=1 * ureg.nm, center_v=369 * ureg.nm, input_medium='vacuum', output_medium='air', output_units='eV')
    (<Quantity(0.00910572235, 'electron_volt')>, <Quantity(3.36000538, 'electron_volt')>)
    >>> convert_spectroscopic_delta(delta=1 * ureg.nm, center_v=369 * ureg.nm, input_medium='air', output_medium='vacuum')
    (<Quantity(1.00025874, 'nanometer')>, <Quantity(369.105045, 'nanometer')>)

    >>> convert_spectroscopic_delta(min_v=368.5 * ureg.nm, max_v=369.5 * ureg.nm, input_medium='air', output_medium='vacuum', returned_values='all')
    (<Quantity(1.00025874, 'nanometer')>, <Quantity(369.105045, 'nanometer')>, <Quantity(368.604916, 'nanometer')>, <Quantity(369.605175, 'nanometer')>)
    >>> convert_spectroscopic_delta(delta=1 * ureg.nm, min_v=368.5 * ureg.nm, input_medium='air', output_medium='vacuum', returned_values='all')
    (<Quantity(1.00025874, 'nanometer')>, <Quantity(369.105045, 'nanometer')>, <Quantity(368.604916, 'nanometer')>, <Quantity(369.605175, 'nanometer')>)
    >>> convert_spectroscopic_delta(delta=1 * ureg.nm, max_v=369.5 * ureg.nm, input_medium='air', output_medium='vacuum', returned_values='all')
    (<Quantity(1.00025874, 'nanometer')>, <Quantity(369.105045, 'nanometer')>, <Quantity(368.604916, 'nanometer')>, <Quantity(369.605175, 'nanometer')>)
    >>> convert_spectroscopic_delta(center_v=369 * ureg.nm, max_v=369.5 * ureg.nm, input_medium='air', output_medium='vacuum', returned_values='all')
    (<Quantity(1.00025874, 'nanometer')>, <Quantity(369.105045, 'nanometer')>, <Quantity(368.604916, 'nanometer')>, <Quantity(369.605175, 'nanometer')>)

    """

    delta: Qty = to_qty(delta, 'length')
    center_v: Qty = to_qty(center_v, 'length')
    min_v: Qty = to_qty(min_v, 'length')
    max_v: Qty = to_qty(max_v, 'length')

    allowed_return_values = {'auto', 'all'}
    assert_options(returned_values.lower(), allowed_return_values, 'returned_values', ValueOutOfOptionsError)

    allowed_mediums = list(_refractive_index_databases.keys()) + ['vacuum']
    assert_options(input_medium, allowed_mediums, 'input_medium', ValueOutOfOptionsError)
    assert_options(output_medium, allowed_mediums, 'output_medium', ValueOutOfOptionsError)

    input_keys = []
    if delta is not None:
        input_keys.append('delta')
        if center_v is not None:
            input_keys.append('center_v')
            min_v = center_v - delta / 2.
            max_v = center_v + delta / 2.
        elif min_v is not None:
            input_keys.append('min_v')
            center_v = min_v + delta / 2.
            max_v = min_v + delta
        elif max_v is not None:
            input_keys.append('max_v')
            center_v = max_v - delta / 2.
            min_v = max_v - delta
        else:
            raise ValueError("Two of center_v, min_v, or max_v must be specified.")
    elif center_v is not None:
        input_keys.append('center_v')
        if min_v is not None:
            input_keys.append('min_v')
            delta = (center_v - min_v) * 2.
            max_v = min_v + delta
        elif max_v is not None:
            input_keys.append('max_v')
            delta = (max_v - center_v) * 2.
            min_v = max_v - delta
        else:
            raise ValueError("Two of delta, center_v, min_v, or max_v must be specified.")
    elif min_v is not None and max_v is not None:
        input_keys.append('min_v')
        input_keys.append('max_v')
        center_v = (min_v + max_v) / 2.
        delta = (max_v - min_v) * 2.
    else:
        raise ValueError("Two of delta, center_v, min_v, or max_v must be specified.")

    freq_min = to_qty_force_units(min_v, 'frequency', 'sp', **{'n': get_refractive_index(min_v, input_medium)})
    freq_center = to_qty_force_units(center_v, 'frequency', 'sp', **{'n': get_refractive_index(center_v, input_medium)})
    freq_max = to_qty_force_units(max_v, 'frequency', 'sp', **{'n': get_refractive_index(max_v, input_medium)})

    return_min = freq_min.to(output_units, 'sp', **{'n': get_refractive_index(freq_min, output_medium)})
    return_center = freq_center.to(output_units, 'sp', **{'n': get_refractive_index(freq_center, output_medium)})
    return_max = freq_max.to(output_units, 'sp', **{'n': get_refractive_index(freq_max, output_medium)})
    return_delta = return_max - return_min
    if return_delta < 0:
        return_delta *= -1
        # return_max, return_min = return_min, return_max

    if returned_values.lower() == 'all':
        return return_delta, return_center, return_min, return_max
    else:
        return_list = []
        if 'delta' in input_keys:
            return_list.append(return_delta)
        if 'center_v' in input_keys:
            return_list.append(return_center)
        if 'min_v' in input_keys:
            return_list.append(return_min)
        if 'max_v' in input_keys:
            return_list.append(return_max)

        return tuple(return_list)
