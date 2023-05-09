import warnings
from pathlib import Path
from typing import Iterable, Any, Type

import matplotlib.pyplot as plt
import numpy as np
import spe_loader as sl
import xmltodict
from sif_parser import np_open as read_sif

from qdl_zno_analysis import Qty
from qdl_zno_analysis.data_classes.metadata_classes import Metadata, MetadataSpectrum
from qdl_zno_analysis.data_classes.quantity_dataframes import QuantityDataFrame
from qdl_zno_analysis.errors import MethodInputError
from qdl_zno_analysis.filename_utils.filename_info import FilenameInfo
from qdl_zno_analysis.filename_utils.filename_manager import FilenameManager
from qdl_zno_analysis.typevars import AnyString, EnhancedNumeric
from qdl_zno_analysis.utils import to_qty_force_units, str_to_valid_varname, interpolated_integration, find_nearest, \
    normalize_dict, varname_to_title_string


class Data:
    _allowed_file_extensions: list[str] = None
    _qdlf_datatype: str = None
    _data_keys: list[str] = None
    _quick_plot_labels: dict[str, str] = None

    def __init__(self, filename: AnyString, metadata):
        if isinstance(filename, Iterable) and not isinstance(filename, str):
            raise TypeError("Filename must be a string or path, not an iterable (e.g. list).")

        self.filename_manager = FilenameManager(filename)

        if not self.filename_manager.valid_paths:
            raise FileNotFoundError(f"No such file or directory: {self.filename_manager.filenames[0]}")

        self._filename = self.filename_manager.filenames[0]
        self._filename_info = self.filename_manager.filename_info_list[0]
        self._filename_info_dict = self.filename_manager.filename_info_dicts[0]
        self._filename_info_norm_dict = self.filename_manager.filename_info_norm_dicts[0]

        self._data = QuantityDataFrame(columns=self._data_keys)
        self._metadata = metadata

        self._read_file()
        self.__post_init__()

    def __post_init__(self):
        self._set_metadata_after_init()
        self._set_data_after_init()
        self._set_quick_plot_labels()

    def _read_file(self):
        warnings.warn('Define your own get_data() function')

    def _set_metadata_after_init(self):
        self._metadata.set_all_values_after_init()

    def _set_data_after_init(self):
        pass

    def _set_quick_plot_labels(self):
        if not hasattr(self, '_data') or self._quick_plot_labels is not None:
            return

        label_dict = {}
        for key in self._data.keys():
            label = varname_to_title_string(key.replace('_per_', '/'), '/')
            values = self._data[key]
            if isinstance(values, Qty):
                label = label + f' ({values.units:~P})'
            label_dict[key] = label

        self._quick_plot_labels = label_dict

    def _check_column_validity(self, string):
        if string not in self.data.columns:
            raise ValueError(f"x_axis must be one of {self.data.columns}, not {string}")

    @property
    def data(self) -> QuantityDataFrame:
        return self._data

    @property
    def metadata(self) -> Type[Metadata]:
        return self._metadata

    @property
    def quick_plot_labels(self) -> dict[str, str]:
        return self._quick_plot_labels

    @property
    def filename_info(self) -> FilenameInfo:
        return self._filename_info

    @property
    def filename_info_dict(self) -> dict[str, Any]:
        return self._filename_info_dict

    @property
    def filename_info_norm_dict(self) -> dict[str, Any]:
        return self._filename_info_norm_dict

    @property
    def filename(self) -> Path:
        return self._filename

    def quick_plot(self, x_axis: str, y_axis: str, fig: plt.Figure = None, ax: plt.Axes = None,
                   *plot_args, **plot_kwargs) -> plt.Line2D:

        self._check_column_validity(x_axis)
        self._check_column_validity(y_axis)

        if fig is None and ax is None:
            fig, ax = plt.subplots()
        elif fig is None:
            fig: plt.Figure = ax.get_figure()
        elif ax is None:
            axes: list[plt.Axes] = fig.get_axes()
            ax = fig.add_subplot(1, 1, 1) if len(axes) == 0 else axes[0]

        x_data = self.data[x_axis]
        y_data = self.data[y_axis]

        lines = ax.plot(x_data, y_data, *plot_args, **plot_kwargs)
        line: plt.Line2D = lines[0]

        if line.get_label().startswith('_child'):
            fno = self.filename_info.file_number
            label = fno if fno is not None else str(self.filename)
            line.set_label(label)

        ax.set_xlabel(self.quick_plot_labels[x_axis])
        ax.set_ylabel(self.quick_plot_labels[y_axis])

        return line


class DataSpectrum(Data):
    _allowed_file_extensions: list[str] = None
    _qdlf_datatype: str = "spectrum"
    _data_keys = ['pixel', 'wavelength_air', 'wavelength_vacuum', 'frequency', 'energy',
                  'counts', 'counts_per_cycle', 'counts_per_time', 'counts_per_time_per_power',
                  'nobg_counts', 'nobg_counts_per_cycle', 'nobg_counts_per_time', 'nobg_counts_per_time_per_power']

    _default_background_per_cycle = Qty(0, 'counts')

    def __init__(self, filename: AnyString, wavelength_offset: EnhancedNumeric = Qty(0, 'nm'), pixel_offset: float = 0,
                 background_per_cycle: EnhancedNumeric = None, second_order: bool = False):
        if background_per_cycle is None:
            background_per_cycle = self._default_background_per_cycle

        metadata = MetadataSpectrum(wfe_offset=wavelength_offset,
                                    pixel_offset=pixel_offset,
                                    background_per_cycle=background_per_cycle,
                                    second_order=second_order)

        super().__init__(filename=filename, metadata=metadata)

    def _set_metadata_after_init(self):
        super()._set_metadata_after_init()
        self._metadata.set_background_per_time_per_power(self._get_excitation_power())

    def _set_data_after_init(self):
        self._set_all_x_data()
        self._set_all_y_data()

    def _set_all_x_data(self):
        wfe = self.metadata.calibrated_wfe
        self._data['pixel'] = self.metadata.pixels
        self._data['energy'] = wfe.energy
        self._data['frequency'] = wfe.frequency
        self._data['wavelength_vacuum'] = wfe.wavelength_vacuum
        self._data['wavelength_air'] = wfe.wavelength_medium

    def _set_all_y_data(self):
        self._data['nobg_counts'] = to_qty_force_units(self.data['counts'] - self.metadata.background, 'counts')

        self._data['counts_per_cycle'] = to_qty_force_units(
            self.data['counts'] / self.metadata.cycles, 'counts')
        self._data['counts_per_time'] = to_qty_force_units(
            self.data['counts_per_cycle'] / self.metadata.exposure_time, 'counts/time')

        self._data['nobg_counts_per_cycle'] = to_qty_force_units(
            self._data['counts_per_cycle'] - self.metadata.background_per_cycle, 'counts')
        self._data['nobg_counts_per_time'] = to_qty_force_units(
            self._data['counts_per_time'] - self.metadata.background_per_time, 'counts/time')

        power_dict = self._get_excitation_power()
        for laser_name, power in power_dict.items():
            data_key = f"counts_per_time_per_{laser_name}_power" if laser_name else "counts_per_time_per_power"

            self._data[data_key] = to_qty_force_units(
                self.data['counts_per_time'] / power, 'counts/time/power')

            self._data[f"nobg_{data_key}"] = to_qty_force_units(
                self._data[data_key] - self.metadata.background_per_time_per_power[laser_name], 'counts/time/power')

    def _get_excitation_power(self) -> dict[str, EnhancedNumeric]:
        lasers = self.filename_info.lsr

        if isinstance(lasers, dict):
            power_dict = {str_to_valid_varname(key): to_qty_force_units(value.power, 'power')
                          for key, value in lasers.items()}
            power_dict[''] = np.sum([value for value in lasers.values()])
        else:
            power_dict = {str_to_valid_varname(lasers.name): to_qty_force_units(lasers.power, 'power')} \
                if lasers.name is not None else {}
            power_dict[''] = to_qty_force_units(lasers.power, 'power')

        for key in power_dict.keys():
            if power_dict[key] is None or power_dict[key].m == 0:
                power_dict[key] = np.nan

        return power_dict

    @property
    def metadata(self) -> MetadataSpectrum:
        return self._metadata

    def quick_plot(self, x_axis: str = 'pixel', y_axis: str = 'counts', fig: plt.Figure = None, ax: plt.Axes = None,
                   *plot_args, **plot_kwargs) -> plt.Line2D:
        return super().quick_plot(x_axis, y_axis, fig, ax, *plot_args, **plot_kwargs)

    def integrate_in_region(self, start, end, x_axis='pixel', y_axis='counts'):
        self._check_column_validity(x_axis)
        self._check_column_validity(y_axis)

        x_data = self.data[x_axis]
        y_data = self.data[y_axis]

        start = start if isinstance(start, Qty) else Qty(start, x_data.units) if hasattr(x_data, 'units') else start
        end = end if isinstance(end, Qty) else Qty(end, x_data.units) if hasattr(x_data, 'units') else end

        return interpolated_integration(start, end, x_data, y_data)

    def integrate_all(self, x_axis='pixel', y_axis='counts'):
        return self.integrate_in_region(self.data[x_axis][0], self.data[x_axis][-1], x_axis, y_axis)

    def get_normalized_data(self, x_value=None, x_axis='pixel', mode='nearest', subtract_min=True):
        allowed_modes = ['nearest', 'linear']
        if mode not in allowed_modes:
            raise MethodInputError('mode', mode, allowed_modes, 'get_normalized_data')

        data = self.data.copy()
        if subtract_min:
            for key in data.keys():
                if 'count' in key:
                    data[key] = data[key] - data[key].min()

        if x_value is None:
            x_value = data[x_axis][np.argmax(data['counts'])]

        if mode == 'nearest':
            x_value = find_nearest(data[x_axis], x_value)

        for key in data.keys():
            if 'count' in key:
                y_value = np.interp(x_value, data[x_axis], data[key])
                data[key] = data[key] / y_value

        return data


class DataSIF(DataSpectrum):
    _default_background_per_cycle = Qty(300, 'counts')

    def _read_file(self):
        counts_info, acquisition_info = read_sif(self.filename)

        self.data['counts'] = Qty(counts_info.tolist()[0][0], 'count')

        self._metadata.exposure_time = Qty(acquisition_info['ExposureTime'], 's')
        self._metadata.cycles = int(acquisition_info['StackCycleTime'] / acquisition_info['ExposureTime'])

        self._metadata.calibration_data = Qty(acquisition_info['Calibration_data'], 'nm')
        self._metadata.pixel_no = len(self.data['counts'])
        self._metadata.pixels = np.array(range(1, self._metadata.pixel_no + 1))


class DataSPE(DataSpectrum):
    _default_background_per_cycle = Qty(0, 'counts')

    def _read_file(self):
        spe_info: sl.SpeFile = sl.load_from_files([str(self.filename)])
        norm_footer_dict = self._get_spe_footer_dict()

        def find_partial_key(text: str):
            return [key for key in norm_footer_dict if text in key]

        self.data['counts'] = Qty(spe_info.data[0][0][0], 'count')

        exposure_text = 'exposuretime._text'
        value = float(norm_footer_dict[find_partial_key(exposure_text)[0]])
        self._metadata.exposure_time = Qty(value, 'ms').to('s')

        cycles_text = 'cyclecount._text'
        value = int(norm_footer_dict[find_partial_key(cycles_text)[0]])
        self._metadata.cycles = value

        self._metadata.pixel_no = len(self.data['counts'])
        self._metadata.pixels = np.array(range(self._metadata.pixel_no))
        self._metadata.calibrated_values = Qty(spe_info.wavelength, 'nm')

    def _get_spe_footer_dict(self) -> dict:
        with open(self.filename) as f:
            footer_pos = sl.read_at(f, 678, 8, np.uint64)[0]

            f.seek(footer_pos)
            xmltext = f.read()

            footer_dict = xmltodict.parse(xmltext)

        return normalize_dict(footer_dict)

