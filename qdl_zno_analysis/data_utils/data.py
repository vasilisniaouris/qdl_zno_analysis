"""
This module includes the basis classes for reading, storing and processing data and metadata from different files and
experiments.
"""
import re
import warnings
from pathlib import Path
from typing import Iterable, Any, Type

import matplotlib.pyplot as plt
import numpy as np
import spe_loader as sl
import xmltodict
from sif_parser import np_open as read_sif

from qdl_zno_analysis import Qty
from qdl_zno_analysis.data_utils.metadata import Metadata, MetadataSpectrum
from qdl_zno_analysis.data_utils.quantity_dataframes import QuantityDataFrame
from qdl_zno_analysis.errors import MethodInputError
from qdl_zno_analysis.filename_utils.filename_info import FilenameInfo
from qdl_zno_analysis.filename_utils.filename_manager import FilenameManager
from qdl_zno_analysis.typevars import AnyString, EnhancedNumeric
from qdl_zno_analysis.utils import to_qty_force_units, str_to_valid_varname, interpolated_integration, find_nearest, \
    normalize_dict, varname_to_title_string


class Data:
    """
    A base class to read, store and process data and metadata from a file. Meant to be subclassed.
    """

    _allowed_file_extensions: list[str] = None
    """ A list of the expected file extensions. """
    _qdlf_datatype: str = None
    """ A predetermined string for separating different types of data for in-lab file type QDLF. """
    _data_keys: list[str] = None
    """ The names of the data-columns used to initialize the dataframe holding all the file and processed data. """
    _quick_plot_labels: dict[str, str] = None
    """ A dictionary with title-styled strings associated with data-columns. To be used as labels in plots. """

    def __init__(self, filename: AnyString, metadata):
        """
        Initialize a Data object with a filename and metadata.

        Parameters
        ----------
        filename: str or Path
            The name of the file to read.
        metadata:
            Metadata associated with the data.
        """
        # test input validity
        if isinstance(filename, Iterable) and not isinstance(filename, str):
            raise TypeError("Filename must be a string or path, not an iterable (e.g. list).")

        self.filename_manager = FilenameManager(filename)

        if not self.filename_manager.valid_paths:
            raise FileNotFoundError(f"No such file or directory: {self.filename_manager.filenames[0]}")

        # set filename-related attributes
        self._filename = self.filename_manager.filenames[0]

        filetype = self.filename.suffix[1:]
        if filetype not in self._allowed_file_extensions:
            raise ValueError(f"Filetype {filetype} is not in the allowed list of "
                             f"filetypes {self._allowed_file_extensions}")

        self._filename_info = self.filename_manager.filename_info_list[0]
        self._filename_info_dict = self.filename_manager.filename_info_dicts[0]
        self._filename_info_norm_dict = self.filename_manager.filename_info_norm_dicts[0]

        # initialize data and metadata
        self._data = QuantityDataFrame(columns=self._data_keys)
        self._metadata = metadata

        self._read_file()
        self.__post_init__()

    def __post_init__(self):
        self._set_metadata_after_init()
        self._set_data_after_init()
        self._set_quick_plot_labels()

    def _read_file(self):
        """ A method to read data from the file. Must be overloaded in subclass. """
        warnings.warn('Define your own get_data() function')

    def _set_metadata_after_init(self):
        """ A method to set metadata after initialization. May be overloaded in subclass. """
        self._metadata.set_all_values_after_init()

    def _set_data_after_init(self):
        """ A method to set data after initialization/file-reading. May be overloaded in subclass. """
        pass

    def _set_quick_plot_labels(self):
        """
        A method to set quick plot labels for the data columns.
        Takes column names and turns them to title-styled strings with units when necessary.
        """

        # make sure `data` has been initialized.
        if not hasattr(self, '_data') or self._quick_plot_labels is not None:
            return

        label_dict = {}
        for key in self._data.keys():  # parse data columns
            label = varname_to_title_string(key.replace('_per_', '/'), '/')  # get title-styled string
            values = self._data[key]
            if isinstance(values, Qty):  # get column units if they exist.
                label = label + f' ({values.units:~P})'
            label_dict[key] = label

        self._quick_plot_labels = label_dict

    def _check_column_validity(self, string):
        """ A method to check if a data column exists. """
        if string not in self.data.columns:
            raise ValueError(f"x_axis must be one of {self.data.columns}, not {string}")

    @property
    def data(self) -> QuantityDataFrame:
        """ Return the dataframe holding all the file and processed data. """
        return self._data

    @property
    def metadata(self) -> Type[Metadata]:
        """ Returns the Metadata dataclass holding all the file and processed metadata. """
        return self._metadata

    @property
    def quick_plot_labels(self) -> dict[str, str]:
        """ Returns plotting label dictionary. """
        return self._quick_plot_labels

    @property
    def filename_info(self) -> FilenameInfo:
        """ Returns the input filename FilenameInfo dataclass. """
        return self._filename_info

    @property
    def filename_info_dict(self) -> dict[str, Any]:
        """ Returns the input filename FilenameInfo dictionary. """
        return self._filename_info_dict

    @property
    def filename_info_norm_dict(self) -> dict[str, Any]:
        """ Returns the input filename FilenameInfo normalized dictionary. """
        return self._filename_info_norm_dict

    @property
    def filename(self) -> Path:
        """ Returns the input filename. """
        return self._filename

    def integrate_in_region(self, start, end, x_axis, y_axis):
        """
        Integrates the area under the curve of the spectrum within a given range.

        Parameters
        ----------
        start: int | float | pint.Quantity
            The start of the integration range. Must be within the range of the `x_axis` data-column.
        end: int | float | pint.Quantity
            The end of the integration range. Must be within the range of the `x_axis` data-column.
        x_axis:
            The label of the x-axis used for the integration.
        y_axis:
            The label of the y-axis used for the integration.

        Returns
        -------
        pint.Quantity
            The value of the integrated area as a pint.Quantity object.

        Notes
        -----
        If the start or end points are not values in the x-axis data-column, this method uses a linear interpolation to
        find the y-axis values at the given start or end points!
        """

        self._check_column_validity(x_axis)
        self._check_column_validity(y_axis)

        x_data = self.data[x_axis]
        y_data = self.data[y_axis]

        start = start if isinstance(start, Qty) else Qty(start, x_data.units) if hasattr(x_data, 'units') else start
        end = end if isinstance(end, Qty) else Qty(end, x_data.units) if hasattr(x_data, 'units') else end

        return interpolated_integration(start, end, x_data, y_data)

    def integrate_all(self, x_axis, y_axis):
        return self.integrate_in_region(self.data[x_axis][0], self.data[x_axis][-1], x_axis, y_axis)

    def get_normalized_data(self, x_value, x_axis, y_axis, shared_y_string_key,
                            mode='nearest', subtract_min=True):
        """
        Returns a copy of the data with all columns normalized by dividing by the maximum value at the given
        `x_value` of `y_axis` data-column or the position of the maximum value of the `y_axis` data-column
        if `x_value` is not provided.

        E.g. if `x_axis` is 'pixel' and `y_axis` is 'counts', if the `x_value` is defined,
        this method will find the 'counts' at, or near, the x-value of 'pixel'
        ('linear' interpolation or 'nearest', depending on `mode`) and normalize everything to this value.
        If x_value is None, the method will find the maximum value of the 'counts' and normalize to this value.

        Parameters
        ----------
        x_value : int | float | pint.Quantity | None
            Value or pint.Quantity object corresponding to the desired `x-axis` value at which to perform normalization.
            If None, the maximum value of the `y_axis` will be used.
        x_axis : str
            The column of data to use as the x-axis
        y_axis : str
            The column of data to use as the y-axis to find the maximum value argument
        shared_y_string_key : str
            A string indicating the shared substring in the keys of the y-axis columns to be normalized.
        mode : str, optional
            Interpolation mode to use when finding the nearest x-axis value to `x_value`.
            Valid options are 'nearest' (gets exact nearest to x_value x_axis-value) and 'linear' (interpolates).
            Default is 'nearest'.
        subtract_min : bool, optional
            If True, subtract the minimum y-axis value from all y-axis columns before normalization. Default is True.

        Returns
        -------
        QuantityDataFrame
            A copy of the data with all columns normalized.

        Raises
        ------
        MethodInputError
            If `mode` is not one of the allowed modes.
        """

        allowed_modes = ['nearest', 'linear']
        if mode not in allowed_modes:
            raise MethodInputError('mode', mode, allowed_modes, 'get_normalized_data')

        data = QuantityDataFrame(self.data.copy())
        if subtract_min:
            for key in data.keys():
                if shared_y_string_key in key:
                    data[key] = data[key] - data[key].min()

        if x_value is None:
            x_value = data[x_axis][np.argmax(data[y_axis])]

        if mode == 'nearest':
            x_value = find_nearest(data[x_axis], x_value)

        for key in data.keys():
            if shared_y_string_key in key:
                y_value = np.interp(x_value, data[x_axis], data[key])
                data[key] = data[key] / y_value

        return data

    def quick_plot(self, x_axis: str, y_axis: str, x_units: str = None, y_units: str = None,
                   fig: plt.Figure = None, ax: plt.Axes = None, *plot_args, **plot_kwargs) -> plt.Line2D:
        """
        Create a quick plot of the specified columns.

        Parameters
        ----------
        x_axis : str
            Name of the `data` column to use as the x-axis.
        y_axis : str
            Name of the `data` column to use as the y-axis.
        x_units : str
            The units to change the x-axis to. Defaults to None.
        y_units : str
            The units to change the y-axis to. Defaults to None.
        fig : plt.Figure, optional
            Figure object to use for the plot. If not provided, and ax is missing, a new figure will be created.
        ax : plt.Axes, optional
            Axes object to use for the plot. If not provided, a new subplot will be created.
        plot_args, plot_kwargs
            Additional arguments and keyword arguments to pass to the matplotlib.pyplot.plot function.

        Returns
        -------
        plt.Line2D
            The plotted line.

        Raises
        ------
        ValueError
            If the specified column names are not present in the data.

        Notes
        -----
        Between provided `fig` and `ax`, the `ax` argument takes precedence.
        The label for the line will be set to the filename or file number if it is not provided by the user.
        """

        # make sure x-axis and y-axis strings exist as columns in `data`.
        self._check_column_validity(x_axis)
        self._check_column_validity(y_axis)

        # make sure we are using the right figure/axes
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        elif fig is None:
            fig: plt.Figure = ax.get_figure()
        elif ax is None:
            axes: list[plt.Axes] = fig.get_axes()
            ax = fig.add_subplot(1, 1, 1) if len(axes) == 0 else axes[0]

        # retrieve the data from the dataframe.
        x_data = self.data[x_axis]
        y_data = self.data[y_axis]

        # change units if needed:
        x_data = x_data.to(x_units) if x_units is not None else x_data
        y_data = y_data.to(y_units) if y_units is not None else y_data

        # plot data on axes.
        lines = ax.plot(x_data, y_data, *plot_args, **plot_kwargs)
        line: plt.Line2D = lines[0]

        # Use file number as a label if label is not provided by the user.
        if line.get_label().startswith('_child'):
            fno = self.filename_info.file_number
            label = fno if fno is not None else str(self.filename)
            line.set_label(label)

        # set axes labels from the default quick plot label dictionary.
        x_label = self.quick_plot_labels[x_axis]
        if x_units is not None:
            x_label = re.sub(r'\([^)]*\)', '(' + x_units + ')', x_label)

        y_label = self.quick_plot_labels[y_axis]
        if y_units is not None:
            y_label = re.sub(r'\([^)]*\)', '(' + y_units + ')', y_label)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return line


class DataSpectrum(Data):
    """
    A class for handling spectral data.
    """
    _allowed_file_extensions: list[str] = ['sif', 'spe']
    """ List of allowed file extensions for spectral data. """
    _qdlf_datatype: str = "spectrum"
    """ QDLF datatype for spectral data. """
    _data_keys = ['pixel', 'wavelength_air', 'wavelength_vacuum', 'frequency', 'energy',
                  'counts', 'counts_per_cycle', 'counts_per_time', 'counts_per_time_per_power',
                  'nobg_counts', 'nobg_counts_per_cycle', 'nobg_counts_per_time', 'nobg_counts_per_time_per_power']
    """ List of keys for spectral data dataframe. """
    _default_background_per_cycle = {'sif': Qty(300, 'counts'), 'spe': Qty(0, 'counts')}
    """ Default background counts per cycle for each file extension. """

    def __init__(self, filename: AnyString, wavelength_offset: EnhancedNumeric = Qty(0, 'nm'), pixel_offset: float = 0,
                 background_per_cycle: EnhancedNumeric = None, second_order: bool = False):

        if background_per_cycle is None:
            background_per_cycle = self._default_background_per_cycle[Path(filename).suffix[1:]]

        metadata = MetadataSpectrum(wfe_offset=wavelength_offset,
                                    pixel_offset=pixel_offset,
                                    background_per_cycle=background_per_cycle,
                                    second_order=second_order)

        super().__init__(filename=filename, metadata=metadata)

    def _read_file(self):
        """ Reads the spectral data file. """
        if self.filename.suffix == '.sif':
            self._read_sif_file()
        elif self.filename.suffix == '.spe':
            self._read_spe_file()

    def _read_sif_file(self):
        """ Reads the spectral data from an SIF file. """

        # get all data with common SIF file parser
        counts_info, acquisition_info = read_sif(self.filename)

        # get primary data
        self.data['counts'] = Qty(counts_info.tolist()[0][0], 'count')

        # get exposure time and cycle-count metadata
        self._metadata.exposure_time = Qty(acquisition_info['ExposureTime'], 's')
        self._metadata.cycles = int(acquisition_info['StackCycleTime'] / acquisition_info['ExposureTime'])

        # get pixel and calibration related metadata
        self._metadata.calibration_data = Qty(acquisition_info['Calibration_data'], 'nm')
        self._metadata.pixel_no = len(self.data['counts'])
        self._metadata.pixels = np.array(range(1, self._metadata.pixel_no + 1))

    def _read_spe_file(self):
        """ Reads the spectral data from an SPE file. """
        # get data with common SPE file reader
        spe_info: sl.SpeFile = sl.load_from_files([str(self.filename)])

        # get normalized footer
        with open(self.filename) as f:
            footer_pos = sl.read_at(f, 678, 8, np.uint64)[0]
            f.seek(footer_pos)
            xmltext = f.read()
            footer_dict = xmltodict.parse(xmltext)

        norm_footer_dict = normalize_dict(footer_dict)

        def find_partial_key(text: str):
            return [key for key in norm_footer_dict if text in key]

        # get primary data
        self.data['counts'] = Qty(spe_info.data[0][0][0], 'count')

        # get exposure time metadata
        exposure_text = 'exposuretime._text'
        value = float(norm_footer_dict[find_partial_key(exposure_text)[0]])
        self._metadata.exposure_time = Qty(value, 'ms').to('s')

        # get cycle-count metadata
        cycles_text = 'cyclecount._text'
        value = int(norm_footer_dict[find_partial_key(cycles_text)[0]])
        self._metadata.cycles = value

        # get pixel and calibration-related data
        self._metadata.pixel_no = len(self.data['counts'])
        self._metadata.pixels = np.array(range(self._metadata.pixel_no))
        self._metadata.calibrated_values = Qty(spe_info.wavelength, 'nm')

    def _set_metadata_after_init(self):
        super()._set_metadata_after_init()
        self._metadata.set_background_per_time_per_power(self._get_excitation_power())

    def _set_data_after_init(self):
        self._set_all_x_data()
        self._set_all_y_data()

    def _set_all_x_data(self):
        """ Sets all the x-axis data for the spectral data. """
        wfe = self.metadata.calibrated_wfe
        self._data['pixel'] = self.metadata.pixels
        self._data['energy'] = wfe.energy
        self._data['frequency'] = wfe.frequency
        self._data['wavelength_vacuum'] = wfe.wavelength_vacuum
        self._data['wavelength_air'] = wfe.wavelength_medium

    def _set_all_y_data(self):
        """ Sets all the y-axis data for the spectral data. """
        self._data['nobg_counts'] = to_qty_force_units(self.data['counts'] - self.metadata.background, 'counts')

        self._data['counts_per_cycle'] = to_qty_force_units(
            self.data['counts'] / self.metadata.cycles, 'counts')
        self._data['counts_per_time'] = to_qty_force_units(
            self.data['counts_per_cycle'] / self.metadata.exposure_time, 'counts/time')

        self._data['nobg_counts_per_cycle'] = to_qty_force_units(
            self._data['counts_per_cycle'] - self.metadata.background_per_cycle, 'counts')
        self._data['nobg_counts_per_time'] = to_qty_force_units(
            self._data['counts_per_time'] - self.metadata.background_per_time, 'counts/time')

        # define power-related data, if power is provided.
        power_dict = self._get_excitation_power()
        for laser_name, power in power_dict.items():  # for each laser, we get a different power-related data-column.
            data_key = f"counts_per_time_per_{laser_name}_power" if laser_name else "counts_per_time_per_power"

            self._data[data_key] = to_qty_force_units(
                self.data['counts_per_time'] / power, 'counts/time/power')

            self._data[f"nobg_{data_key}"] = to_qty_force_units(
                self._data[data_key] - self.metadata.background_per_time_per_power[laser_name], 'counts/time/power')

    def _get_excitation_power(self) -> dict[str, EnhancedNumeric]:
        """
        Finds the excitation power(s) of the laser(s) used to collect the spectrum.
        Finds the total laser power as well.

        Returns
        -------
        A dictionary of the form {laser_name: excitation_power} for each laser used, and a key with an empty string for
        the total excitation power.
        """
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

    def quick_plot(self, x_axis: str = 'pixel', y_axis: str = 'counts', x_units: str = None, y_units: str = None,
                   fig: plt.Figure = None, ax: plt.Axes = None, *plot_args, **plot_kwargs) -> plt.Line2D:
        return super().quick_plot(x_axis, y_axis, x_units, y_units, fig, ax, *plot_args, **plot_kwargs)

    def integrate_in_region(self, start, end, x_axis='pixel', y_axis='counts'):
        return super().integrate_in_region(start, end, x_axis, y_axis)

    def integrate_all(self, x_axis='pixel', y_axis='counts'):
        return super().integrate_all(x_axis, y_axis)

    def get_normalized_data(self, x_value=None, x_axis='pixel', y_axis='counts',
                            shared_y_string_key='count', mode='nearest', subtract_min=True):
        return super().get_normalized_data(x_value, x_axis, y_axis, shared_y_string_key, mode, subtract_min)
