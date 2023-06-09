"""
This module includes the basis classes for storing and processing metadata from different files and experiments.
"""

from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
import pint_xarray
import xarray as xr

from qdl_zno_analysis import Qty
from qdl_zno_analysis.physics import WFE
from qdl_zno_analysis.typevars import EnhancedNumeric
from qdl_zno_analysis.utils import to_qty_force_units, to_qty, Dataclass


@dataclass(repr=False)
class Metadata(Dataclass):
    """ A base class to store and process metadata from a file. Meant to be subclassed. """

    def post_init(self):
        return None


@dataclass(repr=False)
class MetadataSpectrum(Metadata):
    """
    Class representing metadata for spectrum data.
    """

    wfe_offset: EnhancedNumeric = field(default=Qty(0, 'nm'), repr=False)
    """ Wavelength/Frequency/Energy. Default is Qty(0, 'nm'). """
    pixel_offset: float = 0
    """ Pixel offset. Default is 0. """
    background_per_cycle: EnhancedNumeric | np.ndarray | list[EnhancedNumeric] | xr.DataArray = Qty(0, 'counts')
    """ Background counts per cycle. Default is Qty(0, 'counts'). A sequence can be given. Deepest axis will be assumed 
    to be pixels, and the other axis will be file_index. Frames can not have different backgrounds. """
    second_order: bool = False
    """ True if the spectra x-axis should be converted to m=2 diffraction order (wavelength is halved, 
    others are doubled). Refractive index should not change. Default is False. """

    exposure_time: list[EnhancedNumeric] | EnhancedNumeric = field(default=Qty([], 's'))
    """ Shutter open / exposure time. Default is Qty([], 's'). """
    cycles: np.ndarray[int] | int = field(default=np.array([]))
    """ Number of cycles of exposure. Default is np.ndarray([]). """
    calibration_data: EnhancedNumeric = None  # must be same as wfe offset units
    """ Calibration data. Must be in same units as `wfe_offset`. Default is None. If not provided, 
    `calibrated_values` is used. """
    input_medium: str = 'air'
    """ Input medium of the wfe and calibration values. Matters if wavelength is provided. Defaults to 'air'. """
    pixel_no: int = None
    """ Total pixel number of detector. Defaults to None. """
    pixels: list[int] = None
    """ List of pixels. Defaults to None. """
    calibrated_values: Qty = None
    """ Calibrated values. Should be in length, frequency, or energy units. default unit is length. 
    Defaults to None. """
    file_no: int = None
    """ The number of files the metadata object represents. Defaults to 1. """

    background: Qty = field(init=False, default=None)
    """ Background counts. Calculated after complete initialization is confirmed. """
    background_per_time: Qty = field(init=False, default=None)
    """ Background counts per time. Calculated after complete initialization is confirmed. """
    background_per_time_per_power: dict[str, Qty] = field(init=False, default=None)
    """ Background counts per time per power. Calculated after complete initialization is confirmed. """

    metadata_xarray: xr.Dataset = field(init=False, default=None, repr=False)

    # post init runs when user calls it, not after __init__ as __post_init__ would.
    def post_init(self, power_dict: dict[str, EnhancedNumeric] | None = None):
        self.wfe_offset: Qty = to_qty(self.wfe_offset, 'length')
        self.background_per_cycle: Qty = to_qty_force_units(self.background_per_cycle, 'counts')
        self.exposure_time: Qty = to_qty_force_units(self.exposure_time, 'time').to('s')
        self.calibration_data = to_qty(self.calibration_data, 'length')
        if self.calibrated_values is None and self.calibration_data is not None:
            self.calibrated_values = self._apply_calibration()

        self.file_no = len(self.cycles) if self.file_no is None else self.file_no

        if power_dict is None:
            power_dict = {}
        self.set_backgrounds(power_dict)

        # set all metadata that may depend on a data coordinate ( i.e. 'file_index' or 'pixel')
        self._set_metadata_xarray()

        # If there was only one file provided, drop file_index dimensions!
        if self.file_no == 1:
            self.cycles = self.cycles[0]
            self.exposure_time = self.exposure_time[0]

            bg_shape = np.shape(self.background_per_cycle)
            if len(bg_shape) == 0:
                self.background = self.background[0]
                self.background_per_time = self.background_per_time[0]
                for laser_name, value in self.background_per_time_per_power.items():
                    self.background_per_time_per_power[laser_name] = value[0]
            else:
                self.background = np.reshape(self.background, self.background_per_cycle)
                self.background_per_time = np.reshape(self.background_per_time, self.background_per_cycle)
                for laser_name, value in self.background_per_time_per_power.items():
                    self.background_per_time_per_power[laser_name] = np.reshape(value, self.background_per_cycle)
        else:
            if np.all(self.cycles[0] == self.cycles):
                self.cycles = self.cycles[0]
            if np.all(self.exposure_time[0] == self.exposure_time):
                self.exposure_time = self.exposure_time[0]

    def set_backgrounds(self, power_dict: dict[str, EnhancedNumeric]):
        """Set the background counts and background counts per time."""

        self.background = self.background_per_cycle * self.cycles
        self.background_per_time = self.background_per_cycle / self.exposure_time

        self._set_background_per_time_per_power(power_dict)

    def _set_background_per_time_per_power(self, power_dict: dict[str, EnhancedNumeric]):
        """
        Set background counts per unit time per unit power for each laser.

        Parameters
        ----------
        power_dict : dict[str, int| float | pint.Quantity]
            A dictionary mapping laser names to their respective power values. An empty string is used for the total
            power.
        """

        self.background_per_time_per_power = {}
        for laser_name, power in power_dict.items():
            self.background_per_time_per_power[laser_name] = to_qty_force_units(
                self.background_per_time / power, 'counts/time/power')

    def _set_metadata_xarray(self):
        self.metadata_xarray = xr.Dataset(coords={'file_index': range(self.file_no), 'pixel': self.pixels})

        bg_variable_names = \
            ['cycles', 'exposure_time', 'background_per_cycle', 'background', 'background_per_time'] + \
            list(self.background_per_time_per_power.keys())

        for variable_name in bg_variable_names:
            self._set_metadata_xarray_variable(variable_name)

        if self.file_no > 1:
            if np.all(self.metadata_xarray['cycles'] == self.metadata_xarray['cycles'][0]):
                self.metadata_xarray['cycles'] = (), self.metadata_xarray['cycles'].data[0]
            if np.all(self.metadata_xarray['exposure_time'] == self.metadata_xarray['exposure_time'][0]):
                self.metadata_xarray['exposure_time'] = (), self.metadata_xarray['exposure_time'].data[0]

        # If only one file was provided, remove file-related coordinates
        if len(self.metadata_xarray['file_index']) == 1:
            self.metadata_xarray = self.metadata_xarray.isel(file_index=0, drop=True)

    def _set_metadata_xarray_variable(self, variable_name: str):
        data = getattr(self, variable_name, None)
        if data is None:
            data = self.background_per_time_per_power[variable_name]
            if variable_name == '':
                variable_name = 'background_per_time_per_power'
            else:
                variable_name = f'background_per_time_per_{variable_name}_power'

        data_shape = np.shape(data)
        if len(data_shape) == 0:
            coord_names = ()
        elif len(data_shape) == 1 and data_shape[0] == self.file_no:
            coord_names = ('file_index',)
        elif len(data_shape) == 1 and data_shape[0] == self.pixel_no:
            coord_names = ('pixel',)
        elif len(data_shape) == 2:
            coord_names = ('file_index', 'pixel')
        else:
            raise ValueError('...')  # TODO: Change error message

        self.metadata_xarray[variable_name] = coord_names, data

    def _apply_calibration(self) -> Qty:
        """ Apply the calibration data to the pixel list in order to obtain calibrated values
        (in wavelength, frequency or energy). """

        pixels = np.asarray(self.pixels) + self.pixel_offset

        result = Qty(np.zeros(pixels.shape), self.calibration_data.u)
        for i, cal_coefficient in enumerate(self.calibration_data):
            result += cal_coefficient * pixels ** i

        return result + self.wfe_offset

    @property
    def calibrated_wfe(self) -> WFE:
        """ Returns the calibrated wavelength/frequency/energy values (WFE). """

        calibrated_values = self.calibrated_values.copy()
        if self.second_order:
            calibrated_values *= 0.5 if calibrated_values.check('[length]') else 2
        return WFE(calibrated_values, self.input_medium, 'air')


T_Metadata = TypeVar("T_Metadata", bound=Metadata)
