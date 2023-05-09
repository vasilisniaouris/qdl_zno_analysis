from dataclasses import dataclass, field

import numpy as np

from qdl_zno_analysis import Qty, ureg
from qdl_zno_analysis.constants import _refractive_index_databases, get_refractive_index, default_units
from qdl_zno_analysis.errors import MethodInputError
from qdl_zno_analysis.typevars import EnhancedNumeric
from qdl_zno_analysis.utils import to_qty_force_units, to_qty

from typing import ClassVar


@dataclass
class WFE:
    wfe: EnhancedNumeric = field(repr=False)
    input_medium: str = field(repr=False, default='air')
    medium: str = 'air'

    wavelength_medium: Qty = field(init=False)
    wavelength_vacuum: Qty = field(init=False)
    frequency: Qty = field(init=False)
    energy: Qty = field(init=False)
    refractive_index: float | np.ndarray = field(init=False)

    _allowed_mediums: ClassVar[list[str]] = list(_refractive_index_databases.keys()) + ['vacuum']

    def __post_init__(self):
        self._check_input()
        self.wfe: Qty = to_qty(self.wfe, 'length')
        self.refractive_index = get_refractive_index(self.wfe, self.medium, self.input_medium)
        input_ref_index = get_refractive_index(self.wfe, self.input_medium, self.input_medium)

        self.frequency = to_qty_force_units(self.wfe, 'frequency', 'sp', n=input_ref_index)
        self.energy = to_qty_force_units(self.frequency, 'energy', 'sp')
        self.wavelength_vacuum = to_qty_force_units(self.frequency, 'length', 'sp')
        self.wavelength_medium = to_qty_force_units(self.frequency, 'length', 'sp', n=self.refractive_index)

    def _check_input(self):
        with ureg.context('sp'):
            if not self.wfe.is_compatible_with('nm'):
                raise ValueError('some message')  # TODO: change message
        if self.input_medium not in self._allowed_mediums:
            raise MethodInputError('input_medium', self.input_medium, self._allowed_mediums, 'WFE')
        if self.medium not in self._allowed_mediums:
            raise MethodInputError('medium', self.medium, self._allowed_mediums, 'WFE')

    @property
    def freq(self):
        return self.frequency

    @property
    def eng(self):
        return self.energy

    @property
    def wvl_mdm(self):
        return self.wavelength_medium

    @property
    def wvl_vac(self):
        return self.wavelength_vacuum


@ureg.context('spectroscopy')
def convert_spectroscopic_delta(delta=None, center_v=None, min_v=None, max_v=None, input_medium='air',
                                output_medium='vacuum', output_units='nm', returned_values='auto'):
    default_length_units = default_units['length']['main']
    delta: Qty = to_qty(delta, 'length')
    center_v: Qty = to_qty(center_v, 'length')
    min_v: Qty = to_qty(min_v, 'length')
    max_v: Qty = to_qty(max_v, 'length')

    allowed_return_values = ['auto', 'all']
    if returned_values.lower() not in allowed_return_values:
        raise ValueError(f"returned_values must be one of {allowed_return_values}.")

    allowed_mediums = list(_refractive_index_databases.keys()) + ['vacuum']
    if input_medium not in allowed_mediums:
        raise MethodInputError('input_medium', input_medium, allowed_mediums, 'convert_spectroscopic_delta')
    if output_medium not in allowed_mediums:
        raise MethodInputError('input_medium', output_medium, allowed_mediums, 'convert_spectroscopic_delta')

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
            raise ValueError("Two off center_v, min_v, or max_v must be specified.")
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
            raise ValueError("Two off delta, center_v, min_v, or max_v must be specified.")
    elif min_v is not None and max_v is not None:
        input_keys.append('min_v')
        input_keys.append('max_v')
        center_v = (min_v + max_v) / 2.
        delta = (max_v - min_v) * 2.
    else:
        raise ValueError("Two off delta, center_v, min_v, or max_v must be specified.")

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
