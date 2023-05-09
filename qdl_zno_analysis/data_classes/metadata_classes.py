from dataclasses import dataclass, field
from typing import List

import numpy as np

from qdl_zno_analysis import Qty, ureg
from qdl_zno_analysis.physics import WFE
from qdl_zno_analysis.typevars import EnhancedNumeric
from qdl_zno_analysis.utils import to_qty_force_units, to_qty


@dataclass
class Metadata:
    def set_all_values_after_init(self):
        return None


@dataclass
class MetadataSpectrum(Metadata):
    wfe_offset: EnhancedNumeric = field(default=Qty(0, 'nm'),  repr=False)
    pixel_offset: float = 0
    background_per_cycle: EnhancedNumeric = Qty(0, 'counts')
    second_order: bool = False
    exposure_time: EnhancedNumeric = None
    cycles: int = None
    calibration_data: EnhancedNumeric = None  # must be same as wfe offset units
    input_medium: str = 'air'
    pixel_no: int = None
    pixels: list[int] = None
    calibrated_values: Qty = None

    background: Qty = field(init=False, default=None)
    background_per_time: Qty = field(init=False, default=None)
    background_per_time_per_power: dict[str, Qty] = field(init=False, default=None)

    def set_all_values_after_init(self):
        self.wfe_offset: Qty = to_qty(self.wfe_offset, 'length')
        self.background_per_cycle: Qty = to_qty_force_units(self.background_per_cycle, 'counts')
        self.exposure_time: Qty = to_qty_force_units(self.exposure_time, 'time').to('s')
        self.calibration_data = to_qty(self.calibration_data, 'length')
        if self.calibrated_values is None and self.calibration_data is not None:
            self.calibrated_values = self._apply_calibration()

        self.set_backgrounds()

    def set_backgrounds(self):
        self.background = self.background_per_cycle * self.cycles
        self.background_per_time = self.background_per_cycle / self.exposure_time

    def set_background_per_time_per_power(self, power_dict: dict[str, EnhancedNumeric]):
        self.background_per_time_per_power = {}
        for laser_name, power in power_dict.items():
            self.background_per_time_per_power[laser_name] = to_qty_force_units(
                self.background_per_time / power, 'counts/time/power')

    def _apply_calibration(self) -> Qty:
        pixels = np.asarray(self.pixels) + self.pixel_offset

        result = Qty(np.zeros(pixels.shape), self.calibration_data.u)
        for i, cal_coefficient in enumerate(self.calibration_data):
            result += cal_coefficient * pixels ** i

        return result + self.wfe_offset

    @property
    def calibrated_wfe(self) -> WFE:
        calibrated_values = self.calibrated_values.copy()
        if self.second_order:
            calibrated_values *= 0.5 if calibrated_values.check('[length]') else 2
        return WFE(calibrated_values, self.input_medium, 'air')

