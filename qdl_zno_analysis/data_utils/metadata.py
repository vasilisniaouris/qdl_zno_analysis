"""
This module includes the basis classes for storing and processing metadata from different files and experiments.
"""

from dataclasses import dataclass, field

import numpy as np

from qdl_zno_analysis import Qty
from qdl_zno_analysis.physics import WFE
from qdl_zno_analysis.typevars import EnhancedNumeric
from qdl_zno_analysis.utils import to_qty_force_units, to_qty, Dataclass


@dataclass(repr=False)
class Metadata(Dataclass):
    """ A base class to store and process metadata from a file. Meant to be subclassed. """

    def set_all_values_after_init(self):
        return None


@dataclass(repr=False)
class MetadataSpectrum(Metadata):
    """
    Class representing metadata for spectrum data.
    """

    wfe_offset: EnhancedNumeric = field(default=Qty(0, 'nm'),  repr=False)
    """ Wavelength/Frequency/Energy. Default is Qty(0, 'nm'). """
    pixel_offset: float = 0
    """ Pixel offset. Default is 0. """
    background_per_cycle: EnhancedNumeric = Qty(0, 'counts')
    """ Background counts per cycle. Default is Qty(0, 'counts'). """
    second_order: bool = False
    """ True if the spectra x-axis should be converted to m=2 diffraction order (wavelength is halved, 
    others are doubled). Refractive index should not change. Default is False. """
    exposure_time: EnhancedNumeric = None
    """ Shutter open / exposure time. Default is None. """
    cycles: int = None
    """ Number of cycles of exposure. Default is None. """
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

    background: Qty = field(init=False, default=None)
    """ Background counts. Calculated after complete initialization is confirmed. """
    background_per_time: Qty = field(init=False, default=None)
    """ Background counts per time. Calculated after complete initialization is confirmed. """
    background_per_time_per_power: dict[str, Qty] = field(init=False, default=None)
    """ Background counts per time per power. Calculated after complete initialization is confirmed. """

    def set_all_values_after_init(self):
        self.wfe_offset: Qty = to_qty(self.wfe_offset, 'length')
        self.background_per_cycle: Qty = to_qty_force_units(self.background_per_cycle, 'counts')
        self.exposure_time: Qty = to_qty_force_units(self.exposure_time, 'time').to('s')
        self.calibration_data = to_qty(self.calibration_data, 'length')
        if self.calibrated_values is None and self.calibration_data is not None:
            self.calibrated_values = self._apply_calibration()

        self.set_backgrounds()

    def set_backgrounds(self):
        """Set the background counts and background counts per time."""
        self.background = self.background_per_cycle * self.cycles
        self.background_per_time = self.background_per_cycle / self.exposure_time

    def set_background_per_time_per_power(self, power_dict: dict[str, EnhancedNumeric]):
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

