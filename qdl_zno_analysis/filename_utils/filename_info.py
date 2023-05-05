"""
This module contains a classes that extracts information from filenames.
The base class `Info` is used as a template for child classes that extract specific information from filenames.
The `FilenameInfo` class is the only class that the user may interact with, since it interfaces
with the `FilenameManager` class.
"""
import dataclasses
import inspect
from dataclasses import dataclass, field, fields
from operator import attrgetter
from typing import Dict, List, Tuple, ClassVar

import numpy as np

from qdl_zno_analysis import Qty, ureg
from qdl_zno_analysis.constants import default_units, get_n_air
from qdl_zno_analysis.errors import InfoSubclassArgumentNumberError
from qdl_zno_analysis.filename_utils.filename_parsing import parse_string_with_units, parse_filename
from qdl_zno_analysis.typevars import AnyString
from qdl_zno_analysis.utils import normalize_dict, str_to_valid_varname, get_class_arg_type_hints


def _value_str_to_attr_physical_type(arg, value_str, attr_physical_type_dict):
    """ Converts a string to its corresponding quantity value. """
    if arg in attr_physical_type_dict:
        physical_type = attr_physical_type_dict[arg]
        if isinstance(physical_type, tuple):
            physical_type, context = physical_type
        else:
            context = None
        if isinstance(value_str, list):
            return Qty.from_list([parse_string_with_units(v, physical_type, context) for v in value_str])
        else:
            return parse_string_with_units(value_str, physical_type, context)
    else:
        return value_str


@dataclass(repr=False)
class Info:
    """
    A base class that defines a few classmethods and class variables that can be used in child classes for
    extracting information from filename strings.
    """

    _attr_physical_type_dict: Dict[str, str] = field(repr=False, default_factory=dict)
    """ A dictionary that maps attribute names to their unit physical type in a parsed filename. """

    _header_code_to_attr_name_conversion_dict: ClassVar[Dict[str, str]] = {}
    """ A dictionary that maps the header code of a parsed filename to the corresponding attribute name of a `Info`
    object. """

    _info_based_parameters: ClassVar[Dict] = {}
    """ A dictionary that maps attribute names to the class of the object that the attribute represents if the
     attribute is an `Info` subclass. """

    @classmethod
    def get_hc2arg(cls) -> Dict[str, str]:
        """ Returns a dictionary that maps the header code of a parsed filename to the corresponding attribute name of a
        `Info` object. """
        return cls._header_code_to_attr_name_conversion_dict

    @classmethod
    def from_parsed_filename(cls, parsed_object, attr_physical_type_dict=None):
        """ Returns an `Info` object with attributes initialized from a parsed filename.
        The argument `parsed_object` can be a list, a dictionary or a subclass of `Info`. """

        if attr_physical_type_dict is None:
            attr_physical_type_dict = cls._attr_physical_type_dict

        if isinstance(parsed_object, list):
            keys = inspect.getfullargspec(cls).args[2:]  # omit self and _attr_physical_type_dict
            if len(keys) < len(parsed_object):
                raise InfoSubclassArgumentNumberError(parsed_object, keys)
            parsed_object = dict(zip(keys, parsed_object))
        elif isinstance(parsed_object, dict):
            parsed_object = {cls.get_hc2arg().get(key, key): value
                             for key, value in parsed_object.items()}

        converted_dict = {'_attr_physical_type_dict': attr_physical_type_dict}

        for attr, value in parsed_object.items():
            if isinstance(value, str):
                converted_dict[attr] = _value_str_to_attr_physical_type(
                    attr, value, attr_physical_type_dict)
            else:
                if attr in cls._info_based_parameters:
                    info_class = cls._info_based_parameters[attr]
                    converted_dict[attr] = info_class.from_parsed_filename(value)
                else:
                    if isinstance(value, dict) \
                            and ScanInfo in get_class_arg_type_hints(cls, attr):
                        parent_physical_type = attr_physical_type_dict.get(attr, None)
                        if isinstance(parent_physical_type, tuple):
                            parent_physical_type, parent_context = parent_physical_type
                            converted_dict[attr] = ScanInfo.from_parsed_filename(
                                value, parent_physical_type, parent_context)
                        else:
                            converted_dict[attr] = ScanInfo.from_parsed_filename(
                                value, parent_physical_type)
                    else:
                        converted_dict[attr] = _value_str_to_attr_physical_type(
                            attr, value, attr_physical_type_dict)

        return cls(**converted_dict)

    def to_dict(self):
        """
        Returns a dictionary representation of the `Info` object.
        Utilizes the `dataclasses._asdict_inner` method to cover inner values to dict as well.
        """
        result = []
        for f in fields(self):
            if '_attr_physical_type_dict' not in f.name:
                value = getattr(self, f.name)
                value = value.to_dict() if isinstance(value, Info) else dataclasses._asdict_inner(value, dict)
                result.append((f.name, value))
        return dict(result)

    def to_normalized_dict(self, separator='.'):
        """ Returns a normalized dictionary representation of the `Info` object.
        The argument `separator` is used to separate keys of nested dictionaries. """
        return normalize_dict(self.to_dict(), separator=separator)

    def __str__(self):
        """
        The string conversion of the `Info` object. It deliberately omits all None values so that the print is
        not crowded.
         """
        # https://stackoverflow.com/questions/72161257/exclude-default-fields-from-python-dataclass-repr
        not_none_fields = ((f.name, attrgetter(f.name)(self))
                           for f in fields(self) if attrgetter(f.name)(self) is not None and f.repr)

        not_none_fields_repr = ", ".join(f"{name}={value!r}" for name, value in not_none_fields)
        return f"{self.__class__.__name__}({not_none_fields_repr})"

    def __repr__(self):
        return str(self)


@dataclass(repr=False)
class ScanInfo(Info):
    """
    A class that defines the attributes of a scan.

    The header conventions of the `ScanInfo` class are defined in the `_header_code_to_attr_name_conversion_dict`
    dictionary, which maps the codes used in the file header to the corresponding attribute names in the class.
    Here are the header codes and their corresponding attribute names:

    - 'Start' or 'From' or 'Initial' or 'Init': `start`
    - 'Stop' or 'To' or 'Final': `stop`
    - 'Step' or 'Resolution' or 'Res': `step`
    - 'StepNo': `step_no`
    - 'Rate': `rate`
    - 'Duration' or 'Dur': `duration`
    - 'Mode': `mode`
    - 'Msc': `miscellaneous`
    """
    start: Qty | None = None
    """ The starting point of the scan. May be a list of values. """
    stop: Qty | None = None
    """ The ending point of the scan.  May be a list of values. """
    step: Qty | None = None
    """ The step size of the scan. May be a list of values, same size as start and stop. """
    step_no: int | np.ndarray[int] | Qty | None = None
    """ The number of steps of the scan. May be a list of values, same size as start and stop. """
    direction: str | List[str] | np.ndarray[int] | None = field(default=None, init=False)
    """ The scanning direction. May be a list of values. """

    rate: str | List[str] | None = None
    """ The scanning speed. May be a list of values. """
    duration: Qty | None = None
    """ The total duration of the scan. """
    mode: str | None = None
    """ The scanning mode. """
    miscellaneous: str | Qty | List | Dict | None = None
    """ Miscellaneous, user-defined information. """

    parent_physical_type: str | None = field(default=None, repr=False)
    """ The unit types of the parent object. """
    parent_context: str | None = field(default=None, repr=False)
    """ The unit context of the parent object. """

    _header_code_to_attr_name_conversion_dict = {
        'Start': 'start',
        'From': 'start',
        'Initial': 'start',
        'Init': 'start',
        'Stop': 'stop',
        'To': 'stop',
        'Final': 'stop',
        'Step': 'step',
        'Resolution': 'step',
        'Res': 'step',
        'StepNo': 'step_no',
        'Rate': 'rate',
        'Duration': 'duration',
        'Dur': 'duration',
        'Mode': 'mode',
        'Msc': 'miscellaneous',
    }

    _attr_physical_type_dict = {
        'start': None,
        'stop': None,
        'step': None,
        'step_no': 'dimensionless',
        'duration': 'time',
        'resolution': None,
    }

    def __post_init__(self):
        if self.step is None and self.step_no is not None:
            self.step = np.abs(Qty((self.stop - self.start) / (self.step_no - 1)).to(self.start.u))
        elif self.step_no is None and self.step is not None:
            self.step_no = 1 + np.abs(Qty((self.stop - self.start) / self.step).to('dimensionless'))

        if isinstance(self.step_no, Qty):
            self.step_no = self.step_no.m
        if isinstance(self.step_no, np.ndarray):
            if len(self.step_no) == 1:
                self.step_no = int(self.step_no[0])
            else:
                self.step_no = np.rint(self.step_no).astype(int)
        elif self.step_no is not None:
            self.step_no = int(self.step_no)

        self.direction = np.rint(np.sign(self.stop - self.start)).astype(int)

    @property
    def initial(self):
        """ Alias for `start`. """
        return self.start

    @property
    def final(self):
        """ Alias for `stop`. """
        return self.stop

    @property
    def range(self) -> Tuple[Qty, Qty] | Tuple[Tuple[Qty, Qty], Tuple[Qty, Qty]]:
        """ Returns the range of the scan range=(start, stop). """
        return self.start, self.stop

    def __add__(self, other):
        """ Imitates the `+` operator. """
        if isinstance(other, (int, float, Qty)):
            return ScanInfo(self._attr_physical_type_dict, self.start + other, self.stop + other, self.step,
                            self.step_no, self.rate, self.duration, self.mode, self.miscellaneous,
                            self.parent_physical_type, self.parent_context)
        else:
            raise TypeError(f'Cannot add {type(other)} to ScanInfo.')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """ Imitates the `-` operator. """
        if isinstance(other, (int, float, Qty)):
            return ScanInfo(self._attr_physical_type_dict, self.start - other, self.stop - other, self.step,
                            self.step_no, self.rate, self.duration, self.mode, self.miscellaneous,
                            self.parent_physical_type, self.parent_context)
        else:
            raise TypeError(f'Cannot subtract {type(other)} from ScanInfo.')

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        """ Imitates the `*` operator. """
        if isinstance(other, (int, float, Qty)):
            return ScanInfo(self._attr_physical_type_dict, self.start * other, self.stop * other, self.step * other,
                            self.step_no, self.rate, self.duration, self.mode, self.miscellaneous,
                            self.parent_physical_type, self.parent_context)
        else:
            raise TypeError(f'Cannot multiply {type(other)} from ScanInfo.')

    def __truediv__(self, other):
        """ Imitates the `/` operator. """
        if isinstance(other, (int, float, Qty)):
            return ScanInfo(self._attr_physical_type_dict, self.start / other, self.stop / other, self.step / other,
                            self.step_no, self.rate, self.duration, self.mode, self.miscellaneous,
                            self.parent_physical_type, self.parent_context)
        else:
            raise TypeError(f'Cannot divide {type(other)} from ScanInfo.')

    def to(self, units, context=None, **context_kwargs):
        """
        Returns a new ScanInfo object with specified unites. Imitates the `pint.Quantity.to` method.

        Parameters
        ----------
        units : str
            The units to convert to.
        context : str, optional
            The unit context to use. If not provided, the current context is used.

        Returns
        -------
        ScanInfo
            The new ScanInfo object with the specified units.

        """
        if context in ['sp', 'spectroscopy'] and 'n' in context_kwargs.keys():
            n = context_kwargs['n']
            with ureg.context(context, n=n[0]):
                start = self.start.to(units) if self.start is not None else None
            with ureg.context(context, n=n[1]):
                stop = self.stop.to(units) if self.stop is not None else None
        else:
            if context is not None:
                ureg.enable_contexts(context, **context_kwargs)
            start = self.start.to(units) if self.start is not None else None
            stop = self.stop.to(units) if self.stop is not None else None

        step_no = self.step_no
        step = np.abs(Qty((stop - start) / (step_no - 1)).to(units))
        rate = self.rate
        duration = self.duration
        mode = self.mode
        miscellaneous = self.miscellaneous
        return ScanInfo(self._attr_physical_type_dict, start, stop, step, step_no, rate, duration, mode,
                        miscellaneous, self.parent_physical_type, context)

    @classmethod
    def from_parsed_filename(cls, parsed_object: Dict | List, parent_physical_type=None, parent_context=None):

        if isinstance(parsed_object, list):
            keys = inspect.getfullargspec(cls).args[2:]  # omit self and _attr_physical_type_dict
            if len(keys) < len(parsed_object):
                raise InfoSubclassArgumentNumberError(parsed_object, keys)
            parsed_object = dict(zip(keys, parsed_object))

        parsed_object['parent_physical_type'] = parent_physical_type
        parsed_object['parent_context'] = parent_context

        if parent_context is not None:
            parent_physical_type = (parent_physical_type, parent_context)

        attr_physical_type_dict = {key: parent_physical_type if value is None else value
                                   for key, value in cls._attr_physical_type_dict.items()}

        return super().from_parsed_filename(parsed_object, attr_physical_type_dict)

    def __getitem__(self, item):
        """ Imitates the __getitem__ functionality of a list.
        Returns the n-th item of all the iterable attributes. Non-iterable attributes are returned as is. """
        if len(self.start) > 1:
            start = self.start[item] if self.start is not None else None
            stop = self.stop[item] if self.stop is not None else None
            step = self.step[item] if self.step is not None else None
            step_no = self.step_no[item] if self.step_no is not None else None
            rate = self.rate[item] if self.rate is not None else None
            duration = self.duration
            mode = self.mode
            miscellaneous = self.miscellaneous
            return ScanInfo(start, stop, step, step_no, rate, duration, mode, miscellaneous,
                            self.parent_physical_type, self.parent_context)
        raise TypeError(f"'ScanInfo' object is not subscriptable")

    def to_dict(self):
        """ Returns a dictionary representation of the `Info` object. """
        return_dict = super().to_dict()
        return_dict.pop('parent_physical_type', None)
        return_dict.pop('parent_context', None)
        return return_dict


@dataclass(repr=False)
class SourceInfo(Info):
    """
    A class that defines the attributes of a source.

    The header conventions of the `SourceInfo` class are defined in the `_header_code_to_attr_name_conversion_dict`
    dictionary, which maps the codes used in the file header to the corresponding attribute names in the class.
    Here are the header codes and their corresponding attribute names:

    - 'Name' : `name`
    - 'Wvl', 'Frq', 'Eng': `wfe`
    - 'Pwr' : `power`
    - 'Ord' : `order`
    - 'Mdm' : `medium`
    - 'Msc' : `miscellaneous`

    """

    name: str | None = None
    """ The name of the source. """
    wfe: Qty | ScanInfo | None = None
    """ The wavelength/frequency/energy of the source. """

    wavelength_vacuum: Qty | None = field(init=False)
    """ The wavelength of the source in vacuum. Determined after initialization. """
    wavelength_air: Qty | None = field(init=False)
    """ The wavelength of the source in air. Determined after initialization. """
    frequency: Qty | None = field(init=False)
    """ The frequency of the source. Determined after initialization. """
    energy: Qty | None = field(init=False)
    """ The energy of the source. Determined after initialization. """

    power: Qty | ScanInfo | None = None
    """ The power of the source. """
    order: int = 1
    """ In case of frequency conversion, the order of the conversion. """
    medium: str = 'Air'
    """ The medium of the source. Only affects the input if wavelength is provided. """
    miscellaneous: str | Qty | List | Dict | None = None
    """ Any miscellaneous information about the source. """

    _attr_physical_type_dict = {
        'wfe': ('length', 'spectroscopy'),
        'power': 'power',
    }

    _header_code_to_attr_name_conversion_dict = {
        'Name': 'name',
        'Wvl': 'wfe',
        'Frq': 'wfe',
        'Eng': 'wfe',
        'Pwr': 'power',
        'Ord': 'order',
        'Mdm': 'medium',
    }

    def __post_init__(self):
        if self._attr_physical_type_dict['wfe'][0] in ['frequency', 'energy'] or \
                (self._attr_physical_type_dict['wfe'][0] == 'length' and self.medium.lower() in ['vacuum', 'vac', 'v']):
            if self._attr_physical_type_dict['wfe'][0] == 'frequency':
                frequency = self.wfe
                energy = self.wfe.to(default_units['energy']['main'], 'spectroscopy')
                wavelength_vacuum = self.wfe.to(default_units['length']['main'], 'spectroscopy')
            elif self._attr_physical_type_dict['wfe'][0] == 'energy':
                energy = self.wfe
                frequency = self.wfe.to(default_units['frequency']['main'], 'spectroscopy')
                wavelength_vacuum = self.wfe.to(default_units['length']['main'], 'spectroscopy')
            else:
                wavelength_vacuum = self.wfe
                frequency = self.wfe.to(default_units['frequency']['main'], 'spectroscopy')
                energy = self.wfe.to(default_units['energy']['main'], 'spectroscopy')

            n_air = get_n_air(Qty.from_list(frequency.range)) \
                if isinstance(frequency, ScanInfo) else get_n_air(frequency)
            wavelength_air = frequency.to(default_units['length']['main'], 'spectroscopy', **{'n': n_air})

        elif self._attr_physical_type_dict['wfe'][0] == 'length' and self.medium.lower() in ['air', 'a']:
            wavelength_air = self.wfe
            n_air = get_n_air((Qty.from_list(wavelength_air.range)), 'air') \
                if isinstance(wavelength_air, ScanInfo) else get_n_air(wavelength_air, 'air')
            frequency = self.wfe.to(default_units['frequency']['main'], 'spectroscopy', **{'n': n_air})
            energy = self.wfe.to(default_units['energy']['main'], 'spectroscopy', **{'n': n_air})
            wavelength_vacuum = frequency.to(default_units['length']['main'], 'spectroscopy')
        else:
            raise ValueError(f"Unknown medium: {self.medium}")

        self.order = int(self.order)

        self.wavelength_vacuum = wavelength_vacuum / self.order
        self.wavelength_air = wavelength_air / self.order
        self.frequency = frequency * self.order
        self.energy = energy * self.order

    @property
    def wvl_air(self):
        """ Alias for `wavelength`. """""
        return self.wavelength_air

    @property
    def wvl_vac(self):
        """ Alias for `wavelength`. """""
        return self.wavelength_vacuum

    @property
    def freq(self):
        """ Alias for `frequency`. """""
        return self.frequency

    @property
    def en(self):
        """ Alias for `energy`. """""
        return self.energy

    @property
    def pwr(self):
        """ Alias for `power`. """""
        return self.power

    @classmethod
    def from_parsed_filename(cls, parsed_object):
        """
        Adds a case where a dict of multiple sources is provided.
        Changes the _attr_physical_type_dict 'wfe' key to match the input dict.
        """
        attr_physical_type_dict = cls._attr_physical_type_dict

        if isinstance(parsed_object, dict):
            if 'Frq' in parsed_object:
                attr_physical_type_dict['wfe'] = ('frequency', 'spectroscopy')
            elif 'Eng' in parsed_object:
                attr_physical_type_dict['wfe'] = ('energy', 'spectroscopy')

        if isinstance(parsed_object, dict):
            if all([isinstance(v, dict) for v in parsed_object.values()]):
                return_dict: Dict[str, SourceInfo] = {str_to_valid_varname(k): cls.from_parsed_filename(v)
                                                      for k, v in parsed_object.items()}
                for key, value in return_dict.items():
                    if value.name is None:
                        value.name = key
                return return_dict

        return super().from_parsed_filename(parsed_object, attr_physical_type_dict)

    # def to_dict(self):
    #     """ Returns a dictionary representation of the `Info` object. """
    #     return_dict = super().to_dict()
    #     # return_dict.pop('wve', None)
    #     return return_dict


@dataclass(repr=False)
class OpticsInfo(Info):
    """
    A class that defines attributes of an optical path.

    The header conventions of the `OpticsInfo` class are defined in the `_header_code_to_attr_name_conversion_dict`
     dictionary, which maps the codes used in the file header to the corresponding attribute names in the class.
     Here are the header codes and their corresponding attribute names:

    - 'HWP' or 'WP2': `half_waveplate_angle`
    - 'QWP' or 'WP4': `quarter_waveplate_angle`
    - 'Plr': `polarizer`
    - 'PnH': `pinhole`
    - 'Flt': `filters`
    - 'Msc': `miscellaneous`
    """

    half_waveplate_angle: Qty | ScanInfo | None = None
    """ The angle of the half waveplate. """
    quarter_waveplate_angle: Qty | ScanInfo | None = None
    """ The angle of the quarter waveplate. """
    polarizer: Qty | str | ScanInfo | None = None
    """ The angle of the polarizer. Can be string like 'H' or 'V'."""
    pinhole: Qty | str = None
    """ The diameter of the pinhole. Can be string like 'In' or 'Out'"""
    filters: str | List = None
    """ The filters used on the optical path. Can be string like 'F1' or ['F1', 'F2'] """
    miscellaneous: str | Qty | List | Dict = None
    """ Any miscellaneous information about the optical path. """

    _attr_physical_type_dict = {
        'half_waveplate_angle': 'angle',
        'quarter_waveplate_angle': 'angle',
        'polarizer': 'angle',
        'pinhole': 'length',
    }

    _header_code_to_attr_name_conversion_dict = {
        'HWP': 'half_waveplate_angle',
        'WP2': 'half_waveplate_angle',
        'QWP': 'quarter_waveplate_angle',
        'WP4': 'quarter_waveplate_angle',
        'Plr': 'polarizer',
        'PnH': 'pinhole',
        'Flt': 'filters',
        'Msc': 'miscellaneous',
    }

    @property
    def hwp(self):
        """ Alias for `half_waveplate_angle`. """""
        return self.half_waveplate_angle

    @property
    def qwp(self):
        """ Alias for `quarter_waveplate_angle`. """""
        return self.quarter_waveplate_angle

    @property
    def wp2(self):
        """ Alias for `half_waveplate_angle`. """""
        return self.half_waveplate_angle

    @property
    def wp4(self):
        """ Alias for `quarter_waveplate_angle`. """""
        return self.quarter_waveplate_angle

    @property
    def plr(self):
        """ Alias for `polarizer`. """""
        return self.polarizer

    @property
    def pnh(self):
        """ Alias for `pinhole`. """""
        return self.pinhole

    @property
    def flt(self):
        """ Alias for `filters`. """""
        return self.filters


@dataclass(repr=False)
class RFLineInfo(Info):
    """ A class that defines attributes of a RF line. """
    pass


@dataclass(repr=False)
class FilenameInfo(Info):
    """
    A class that defines attributes of a filename.
    The `FilenameInfo` class is designed to facilitate the handling and manipulation of
    information related to a data file's filename,
    making it easier to work with and extract meaningful information from the filename.

    The header conventions of the `FilenameInfo` class are the mapping of the class attributes
    to their corresponding header codes used in the filename.
    The header codes and their corresponding attribute names are defined in the
    _header_code_to_attr_name_conversion_dict class variable.
    The header conventions for FilenameInfo are as follows:

    - 'FNo': `file_number`
    - 'Smp': `sample_name`
    - 'Lsr': `lasers`
    - 'RFS': `rf_sources`
    - 'MgF': `magnetic_field`
    - 'Tmp': `temperature`
    - 'Exc': `excitation_path_optics`
    - 'Col': `collection_path_optics`
    - 'EnC': `exc_and_col_path_optics`
    - 'RFL': `rf_lines`
    - 'MsT': `measurement_type`
    - 'Msc': `miscellaneous`
    - 'Spt': `spot`
    - 'Misused': `other`

    These conventions are used to encode and decode the filename headers, which are often used to
    store metadata about the data in a file. The `from_filename()` method of `FilenameInfo`
    uses these conventions to extract the attributes from a filename.
    """

    file_number: int | None = None
    """ The file number of the file. """
    sample_name: str | None = None
    """ The sample name relevant to the data. """
    lasers: SourceInfo | Dict[str, SourceInfo] | None = None
    """ The lasers used for the data acquisition. """
    rf_sources: SourceInfo | Dict[str, SourceInfo] | None = None
    """ The RF sources used for the data acquisition. """
    magnetic_field: Qty | ScanInfo | None = None
    """ The magnetic field used for the data acquisition. """
    temperature: Qty | ScanInfo | None = None
    """ The temperature used for the data acquisition. """
    excitation_path_optics: OpticsInfo | None = None
    """ The excitation path optics used for the data acquisition. """
    collection_path_optics: OpticsInfo | None = None
    """ The collection path optics used for the data acquisition. """
    exc_and_col_path_optics: OpticsInfo | None = None
    """ The excitation-collection common path optics used for the data acquisition. """
    rf_lines: RFLineInfo | None = None
    """ The RF lines used for the data acquisition. """
    measurement_type: str | None = None
    """ The measurement type of the data acquisition."""
    miscellaneous: str | Qty | List | Dict | None = None
    """ Any miscellaneous information about the data acquisition. """
    spot: Qty | ScanInfo | None = None
    """ The sample location at which the data was acquired. """
    other: str | Qty | List | Dict | None = None
    """ Any other information about the data acquisition. """

    _info_based_parameters = {
        'lasers': SourceInfo,
        'rf_sources': SourceInfo,
        'excitation_path_optics': OpticsInfo,
        'collection_path_optics': OpticsInfo,
        'exc_and_col_path_optics': OpticsInfo,
        'rf_lines': RFLineInfo,
    }

    _attr_physical_type_dict = {
        'file_number': 'dimensionless',
        'magnetic_field': 'magnetic field',
        'temperature': 'temperature',
        'spot': 'length',
    }

    _header_code_to_attr_name_conversion_dict = {
        'FNo': 'file_number',
        'Smp': 'sample_name',
        'Lsr': 'lasers',
        'RFS': 'rf_sources',
        'MgF': 'magnetic_field',
        'Tmp': 'temperature',
        'Exc': 'excitation_path_optics',
        'Col': 'collection_path_optics',
        'EnC': 'exc_and_col_path_optics',
        'RFL': 'rf_lines:',
        'MsT': 'measurement_type',
        'Msc': 'miscellaneous',
        'Spt': 'spot',
        'Misused': 'other',
    }

    def __post_init__(self):
        self.file_number = int(self.file_number) if self.file_number is not None else None

    @property
    def fno(self):
        """ Alias for `file_number`. """
        return self.file_number

    @property
    def smp(self):
        """ Alias for `sample_name`. """
        return self.sample_name

    @property
    def lsr(self):
        """ Alias for `lasers`. """
        return self.lasers

    @property
    def rfs(self):
        """ Alias for `rf_sources`. """
        return self.rf_sources

    @property
    def mgf(self):
        """ Alias for `magnetic_field`. """
        return self.magnetic_field

    @property
    def tmp(self):
        """ Alias for `temperature`. """
        return self.temperature

    @property
    def exc(self):
        """ Alias for `excitation_path_optics`. """
        return self.excitation_path_optics

    @property
    def col(self):
        """ Alias for `collection_path_optics`. """
        return self.collection_path_optics

    @property
    def enc(self):
        """ Alias for `exc_and_col_path_optics`. """
        return self.exc_and_col_path_optics

    @property
    def rfl(self):
        """ Alias for `rf_lines`. """
        return self.rf_lines

    @property
    def mst(self):
        """ Alias for `measurement_type`. """
        return self.measurement_type

    @property
    def msc(self):
        """ Alias for `miscellaneous`. """
        return self.miscellaneous

    @property
    def spt(self):
        """ Alias for `spot`. """
        return self.spot

    @property
    def spot_x(self):
        """ Returns the first element of the spot list of values. """
        return self.spot[0]

    @property
    def spot_y(self):
        """ Returns the second element of the spot list of values. """
        return self.spot[1]

    @property
    def spt_x(self):
        """ Alias for `spot_x`. """
        return self.spot[0]

    @property
    def spt_y(self):
        """ Alias for `spot_y`. """
        return self.spot[1]

    @classmethod
    def from_filename(cls, filename: AnyString):
        """
        Creates a `FilenameInfo` object from a filename.

        Examples
        --------
        A simple filename example, input of laser metadata is via a "list":

        >>> fn = '001_Lsr~Matisse-737p8n-10n-2_Tmp~6p1K_MgF~5p1_Col~PnH~40u_Spt~n2p1u-4p4u.csv'
        >>> FilenameInfo.from_filename(fn)
        FilenameInfo(file_number=1, lasers=SourceInfo(name='Matisse', wfe=<Quantity(737.8, 'nanometer')>, wavelength_vacuum=<Quantity(369.001625, 'nanometer')>, wavelength_air=<Quantity(368.9, 'nanometer')>, frequency=<Quantity(812442.109, 'gigahertz')>, energy=<Quantity(3.35999058, 'electron_volt')>, power=<Quantity(10.0, 'nanowatt')>, order=2, medium='Air'), magnetic_field=<Quantity(5.1, 'tesla')>, temperature=<Quantity(6.1, 'kelvin')>, collection_path_optics=OpticsInfo(pinhole=<Quantity(40.0, 'micrometer')>), spot=<Quantity([-2.1  4.4], 'micrometer')>)

        Input of the laser metadata is via a "dict":

        >>> fn = '001_Lsr~Name~Matisse-Wvl~737p8n-Pwr~10n-Ord~2_Tmp~6p1_MgF~5p1_Col~PnH~40u'
        >>> FilenameInfo.from_filename(fn).lasers
        SourceInfo(name='Matisse', wfe=<Quantity(737.8, 'nanometer')>, wavelength_vacuum=<Quantity(369.001625, 'nanometer')>, wavelength_air=<Quantity(368.9, 'nanometer')>, frequency=<Quantity(812442.109, 'gigahertz')>, energy=<Quantity(3.35999058, 'electron_volt')>, power=<Quantity(10.0, 'nanowatt')>, order=2, medium='Air')

        We can store multiple information within the same subparameter with the (embeded) list/dict format:

        >>> fn = '001_Lsr~Name~Matisse-Wvl~737p8n-Pwr~10n-Ord~2_Tmp~6p1_MgF~5p1_Col~PnH~40u-Flt~LP380;BP370,6.csv'
        >>> FilenameInfo.from_filename(fn).col.flt
        ['LP380', ['BP370', '6']]

        We can setup scan with the dict format

        >>> fn = '001_Lsr~Matisse-From~737p7;To~737p9;Step~50p-10n-2_Tmp~6p1_MgF~5p1_Col~PnH~40u'
        >>> FilenameInfo.from_filename(fn).lasers
        SourceInfo(name='Matisse', wfe=ScanInfo(start=<Quantity(737.7, 'nanometer')>, stop=<Quantity(737.9, 'nanometer')>, step=<Quantity(50.0, 'picometer')>, step_no=4, direction=1), wavelength_vacuum=ScanInfo(start=<Quantity(368.951611, 'nanometer')>, stop=<Quantity(369.051638, 'nanometer')>, step=<Quantity(0.0333423189, 'nanometer')>, step_no=4, direction=1), wavelength_air=ScanInfo(start=<Quantity(368.85, 'nanometer')>, stop=<Quantity(368.95, 'nanometer')>, step=<Quantity(25.0, 'picometer')>, step_no=4, direction=1), frequency=ScanInfo(start=<Quantity(812552.24, 'gigahertz')>, stop=<Quantity(812332.007, 'gigahertz')>, step=<Quantity(73.4107997, 'gigahertz')>, step_no=4, direction=-1), energy=ScanInfo(start=<Quantity(3.36044605, 'electron_volt')>, stop=<Quantity(3.35953524, 'electron_volt')>, step=<Quantity(0.000303602673, 'electron_volt')>, step_no=4, direction=-1), power=<Quantity(10.0, 'nanowatt')>, order=2, medium='Air')

        We cannot setup a scan with the list format! A list initialization is only reserved for first-level input.

        >>> fn = '001_Lsr~Matisse-737p7;737p9;50p-10n-2_Tmp~6p1_MgF~5p1_Col~PnH~40u'
        >>> FilenameInfo.from_filename(fn).lasers
        SourceInfo(name='Matisse', wfe=<Quantity([7.377e+02 7.379e+02 5.000e-02], 'nanometer')>, wavelength_vacuum=<Quantity([3.68951611e+02 3.69051638e+02 2.50077001e-02], 'nanometer')>, wavelength_air=<Quantity([3.6885e+02 3.6895e+02 2.5000e-02], 'nanometer')>, frequency=<Quantity([8.12552240e+05 8.12332007e+05 1.19880060e+10], 'gigahertz')>, energy=<Quantity([3.36044605e+00 3.35953524e+00 4.95784091e+04], 'electron_volt')>, power=<Quantity(10.0, 'nanowatt')>, order=2, medium='Air')

        We can include two or more sources with the dict format!

        >>> fn = '001_Lsr~Matisse~Wvl~737p8;Pwr~10n;Ord~2-Toptica~Wvl~368p1;Pwr~20n_Tmp~6p1_MgF~5p1_Col~PnH~40u'
        >>> FilenameInfo.from_filename(fn).lasers
        {'matisse': SourceInfo(name='matisse', wfe=<Quantity(737.8, 'nanometer')>, wavelength_vacuum=<Quantity(369.001625, 'nanometer')>, wavelength_air=<Quantity(368.9, 'nanometer')>, frequency=<Quantity(812442.109, 'gigahertz')>, energy=<Quantity(3.35999058, 'electron_volt')>, power=<Quantity(10.0, 'nanowatt')>, order=2, medium='Air'), 'toptica': SourceInfo(name='toptica', wfe=<Quantity(368.1, 'nanometer')>, wavelength_vacuum=<Quantity(368.204812, 'nanometer')>, wavelength_air=<Quantity(368.1, 'nanometer')>, frequency=<Quantity(814200.271, 'gigahertz')>, energy=<Quantity(3.36726176, 'electron_volt')>, power=<Quantity(20.0, 'nanowatt')>, order=1, medium='Air')}

        We can choose to scan one or more resources

        >>> fn = '001_Lsr~Matisse~Wvl~From~737p7,To~737p9,Step~50p;Pwr~10n;Ord~2-Toptica~Wvl~368p1;Pwr~20n_Tmp~6p1_MgF~5p1_Col~PnH~40u'
        >>> FilenameInfo.from_filename(fn).lasers
        {'matisse': SourceInfo(name='matisse', wfe=ScanInfo(start=<Quantity(737.7, 'nanometer')>, stop=<Quantity(737.9, 'nanometer')>, step=<Quantity(50.0, 'picometer')>, step_no=4, direction=1), wavelength_vacuum=ScanInfo(start=<Quantity(368.951611, 'nanometer')>, stop=<Quantity(369.051638, 'nanometer')>, step=<Quantity(0.0333423189, 'nanometer')>, step_no=4, direction=1), wavelength_air=ScanInfo(start=<Quantity(368.85, 'nanometer')>, stop=<Quantity(368.95, 'nanometer')>, step=<Quantity(25.0, 'picometer')>, step_no=4, direction=1), frequency=ScanInfo(start=<Quantity(812552.24, 'gigahertz')>, stop=<Quantity(812332.007, 'gigahertz')>, step=<Quantity(73.4107997, 'gigahertz')>, step_no=4, direction=-1), energy=ScanInfo(start=<Quantity(3.36044605, 'electron_volt')>, stop=<Quantity(3.35953524, 'electron_volt')>, step=<Quantity(0.000303602673, 'electron_volt')>, step_no=4, direction=-1), power=<Quantity(10.0, 'nanowatt')>, order=2, medium='Air'), 'toptica': SourceInfo(name='toptica', wfe=<Quantity(368.1, 'nanometer')>, wavelength_vacuum=<Quantity(368.204812, 'nanometer')>, wavelength_air=<Quantity(368.1, 'nanometer')>, frequency=<Quantity(814200.271, 'gigahertz')>, energy=<Quantity(3.36726176, 'electron_volt')>, power=<Quantity(20.0, 'nanowatt')>, order=1, medium='Air')}

        We can scan in more than one dimensions, too!

        >>> fn = '001_Lsr~Matisse-737p8n-10n-2_Tmp~6p1K_MgF~5p1_Col~PnH~40u_Spt~From~n10u;n10u-To~10u;10u-StepNo~50;50.csv'
        >>> FilenameInfo.from_filename(fn).spot
        ScanInfo(start=<Quantity([-10. -10.], 'micrometer')>, stop=<Quantity([10. 10.], 'micrometer')>, step=<Quantity([0.40816327 0.40816327], 'micrometer')>, step_no=array([50, 50]), direction=array([1, 1]))

        """
        parsed_filename_dict = parse_filename(filename)
        return cls.from_parsed_filename(parsed_filename_dict)
