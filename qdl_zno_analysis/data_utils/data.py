"""
This module includes the basis classes for reading, storing and processing data and metadata from different files and
experiments.
"""
import re
import warnings
from pathlib import Path
from typing import Any, Type, Sequence

import numpy as np
import pint_xarray
import xarray as xr
from pint_xarray.conversion import extract_units, strip_units

try:  # visualization dependencies
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    pass

try:  # spectroscopy dependencies
    import spe_loader as sl
    import xmltodict
    from sif_parser import np_open as read_sif
except ModuleNotFoundError:
    pass

from qdl_zno_analysis import Qty, ureg
from qdl_zno_analysis.data_utils.metadata import Metadata, MetadataSpectrum
from qdl_zno_analysis.errors import MethodInputError
from qdl_zno_analysis.filename_utils.filename_info import FilenameInfo
from qdl_zno_analysis.filename_utils.filename_manager import FilenameManager
from qdl_zno_analysis.typevars import AnyString, EnhancedNumeric
from qdl_zno_analysis.utils import to_qty_force_units, str_to_valid_varname, \
    normalize_dict, varname_to_title_string, convert_coord_to_dim, integrate_xarray, get_normalized_xarray, \
    quick_plot_xarray


class Data:
    """
    A base class to read, store and process data and metadata from a file. Meant to be subclassed.
    """

    _allowed_file_extensions: list[str] = None
    """ A list of the expected file extensions. """
    _qdlf_datatype: str = None
    """ A predetermined string for separating different types of data for in-lab file type QDLF. """

    _data_dim_names: list[str] = None
    """ The names of the dataset dimensions used to initialize
    the xarray dataset holding all the file and processed data. """
    _data_coords: dict[str, tuple[str, ...]] = None
    """ The names of the dataset coordinates and the dimension they 
    are associated with in a form a tuple that are used to initialize 
    the xarray dataset holding all the file and processed data. """
    _data_variables: dict[str, tuple[str, ...]] = None
    """ The names of the dataset variables and the dimension they are 
    associated with in a form of a tuple that are used to initialize 
    the xarray dataset holding all the file and processed data. """

    _quick_plot_labels: dict[str, str] = None
    """ A dictionary with title-styled strings associated with data-columns. To be used as labels in plots. """

    def __init__(
            self,
            filename_input: AnyString | FilenameManager,
            metadata=None,
    ):
        """
        Initialize a Data object with a filename and metadata.

        Parameters
        ----------
        filename_input: str or Path or FilenameManager
            The name of the file to read, or the filename manager containing the filename of interest..
        metadata:
            Metadata associated with the data.
        """

        if isinstance(filename_input, FilenameManager):
            self.filename_manager = filename_input
        else:
            FilenameManager(filename_input)

        self._check_filename_manager_input()

        # setting filename info attributes
        self._filename_info = self.filename_manager.filename_info_list[0]
        self._filename_info_dict = self.filename_manager.filename_info_dicts[0]
        self._filename_info_norm_dict = self.filename_manager.filename_info_norm_dicts[0]

        # initialize data and metadata
        # make empty xarray Dataset with predetermined dimensions.
        self._data = xr.Dataset(coords={dim: [] for dim in self._data_dim_names})
        self._metadata = metadata

        self._read_file()
        self.__post_init__()

    def __post_init__(self):
        self._set_metadata_after_init()
        self._set_data_after_init()
        self._set_quick_plot_labels()

    def _check_filename_manager_input(self):
        if not self.filename_manager.valid_paths:
            raise FileNotFoundError(f"No such file(s) or directory(/ies): {self.filename_manager.filenames}")

        if len(self.filename_manager.valid_paths) > 1:
            raise ValueError(f"More than one filenames found: {self.filename_manager.valid_paths}")

        if len(self.filename_manager.available_filetypes) > 1:
            raise ValueError(f"More than one filetype found: {self.filename_manager.available_filetypes}")

        filetype = self.filename_manager.available_filetypes[0]
        if filetype not in self._allowed_file_extensions:
            raise ValueError(f"Filetype {filetype} is not in the allowed list of "
                             f"filetypes {self._allowed_file_extensions}")

    def _read_file(self):
        """
        A method to read data from the file. Must be overloaded in subclass.

        First initialize the data coordinates, and then the variables that depend on them.
        """
        warnings.warn('Define your own get_data() function')

    def _set_metadata_after_init(self):
        """ A method to set metadata after initialization. May be overloaded in subclass. """
        self._metadata.set_all_values_after_init()

    def _set_data_after_init(self):
        """ A method to set data after initialization/file-reading. May be overloaded in subclass. """
        self._set_all_data_coords()
        self._set_all_data_variables()

    def _set_all_data_coords(self):
        """ A method to set data coordinates after initialization/file-reading. May be overloaded in subclass. """
        pass

    def _set_all_data_variables(self):
        """ A method to set data variables after initialization/file-reading. May be overloaded in subclass. """
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
        for key in self._data.variables.keys():  # parse data columns
            key = str(key)
            label = varname_to_title_string(key.replace('_per_', '/'), '/')  # get title-styled string
            values = self._data[key].data
            if isinstance(values, Qty):  # get column units if they exist.
                label = label + f' [{values.units:~P}]'
            label_dict[key] = label

        self._quick_plot_labels = label_dict

    def _check_column_validity(self, string):
        """ A method to check if a data column exists. """
        if string not in self.data.variables.keys():
            raise ValueError(f"Axis must be one of {list(self.data.variables.keys())}, not {string}")

    @property
    def data(self) -> xr.Dataset:
        """ Returns the xarray dataset holding all the file and processed data. """
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
        return self.filename_manager.valid_paths[0]

    def integrate(
            self,
            start: EnhancedNumeric | None = None,
            end: EnhancedNumeric | None = None,
            coord: str | None = None,
            var: str | None = None,
    ):
        """
        Integrates the area under the curve of the data array/set within a given range.

        Parameters
        ----------
        start : int | float | pint.Quantity | None, optional
            The start of the integration range. Defaults to the first element of the coordinate array.
            Must be within the range of the coordinate.
        end : int | float | pint.Quantity | None, optional
            The end of the integration range. Defaults to the last element of the coordinate array.
            Must be within the range of the range of the coordinate.
        coord : str | None, optional
            The data coordinate that will be used as the x-axis of the integration.
            Defaults to the last (deepest level) dimension of the dataset.
        var : str | None, optional
            The data variable that will be used as the y-axis of the integration.
            Defaults to the entire dataset.

        Returns
        -------
        xr.Dataset | xr.DataArray
            The reduced data array/set of the integrated array values.

        Raises
        ------
        MethodInputError
            If the specified coordinate or variable is not found in the data.

        Notes
        -----
        If the start or end points are not values in the coordinate data array (x-axis),
        this method uses cubic interpolation to find the y-axis values at the given start or end points.
        """

        return integrate_xarray(self.data, start, end, coord, var)

    def get_normalized_data(
            self,
            norm_axis_val: EnhancedNumeric | xr.DataArray | np.ndarray | None = None,
            norm_axis: str | None = None,
            norm_var: str | None = None,
            mode='nearest',
            subtract_min=True
    ):
        """
        Get normalized data based on specified parameters.

        Parameters
        ----------
        norm_axis_val : int | float | Qty | xr.DataArray | np.ndarray | None, optional
            The value used for normalization along the `norm_axis`.
            If None, the maximum value of the normalization axis is used.
        norm_axis : str | None, optional
            The axis used for normalization. If None, the last axis (deepest level) is used.
        norm_var : str | None, optional
            The variable used for normalization. If None, the first variable in the dataset is used.
        mode : {'nearest', 'linear', 'quadratic', 'cubic'}, optional
            The interpolation mode for finding norm_axis values.
            'nearest' - Find the nearest `norm_axis` to the `norm_axis_val` .
            'linear' - Perform linear interpolation between adjacent `norm_axis` values.
            'quadratic' - Perform quadratic interpolation between adjacent `norm_axis` values.
            'cubic' - Perform cubic interpolation between adjacent `norm_axis` values.
            Default is 'nearest'.
        subtract_min : bool, optional
            If True, subtract the minimum values from the data before normalization.

        Returns
        -------
        xarray.Dataset
            The normalized data.

        Raises
        ------
        ValueError
            If `norm_axis` or `norm_var` is not found in the data.
        MethodInputError
            If the mode is not one of the allowed modes.
        """
        return get_normalized_xarray(self._data, norm_axis_val, norm_axis, norm_var, mode, subtract_min)

    def quick_plot(
            self,
            var: str | None = None,
            coord1: str | None = None,
            coord2: str | None = None,
            var_units: str | None = None,
            coord1_units: str | None = None,
            coord2_units: str | None = None,
            plot_method: str | None = None,
            **plot_kwargs
    ) -> Type[plt.Artist] | list[Type[plt.Artist]]:
        """
        Create a quick plot of the `data` dataset.

        Parameters
        ----------
        var : str | None, optional
            Name of the data variable to use as the y-axis for 2D data and z-axis for 3D data.
            If None, the first variable in the dataset is used.
        coord1 : str | None, optional
            Name of the coordinate to use as the x-axis.
        coord2 : str | None, optional
            Name of the coordinate to use as the y-axis (only for 3D data).
        var_units : str | None, optional
            The units to change the plotted variable to.
        coord1_units : str | None, optional
            The units to change the x-axis to.
        coord2_units : str | None, optional
            The units to change the y-axis to (only for 3D data).
        plot_method : str | None, optional
            The specific plotting method to use. See `xarray.DataArray.plot` for options.
            If None, the default method for the input variable will be used.
        **plot_kwargs
            Additional keyword arguments to pass to the matplotlib.pyplot.plot function.

        Returns
        -------
        plt.Artist
            The object returned by the `xarray.DataArray.plot` method.

        Raises
        ------
        ValueError
            If the specified column names are not present in the data.

        Notes
        -----
        The legend label for the line will be set to the filename or file number if it is not provided by the user.
        """

        result = quick_plot_xarray(self.data, var, coord1, coord2, var_units,
                                   coord1_units, coord2_units, plot_method, **plot_kwargs)

        # Use file number or filename as a legend label if it is not provided by the user.
        if not isinstance(result, Sequence):
            result_parse = [result]
        else:
            result_parse = result

        for res in result_parse:
            if res.get_label().startswith('_child'):
                fno = self.filename_info.file_number
                label = fno if fno is not None else str(self.filename)
                res.set_label(label)

        return result


class DataSpectrum(Data):
    """
    A class for handling spectral data.
    """
    _allowed_file_extensions: list[str] = ['sif', 'spe']
    """ List of allowed file extensions for spectral data. """
    _qdlf_datatype: str = "spectrum"
    """ QDLF datatype for spectral data. """

    _data_dim_names: list[str] = ['frame', 'pixel']
    """ The names of the dataset dimensions used to initialize
    the xarray dataset holding all the file and processed data. """
    _data_coords: dict[str, tuple[str, ...]] = {
        'time': ('frame',),
        'wavelength_air': ('pixel',),
        'wavelength_vacuum': ('pixel',),
        'frequency': ('pixel',),
        'energy': ('pixel',),
    }
    """ The names of the dataset coordinates and the dimension they 
    are associated with in a form a tuple that are used to initialize 
    the xarray dataset holding all the file and processed data. """
    _data_variables: dict[str, tuple[str, ...]] = {
        'counts': ('frame', 'pixel'),
        'counts_per_cycle': ('frame', 'pixel'),
        'counts_per_time': ('frame', 'pixel'),
        'counts_per_time_per_power': ('frame', 'pixel'),
        'nobg_counts': ('frame', 'pixel'),
        'nobg_counts_per_cycle': ('frame', 'pixel'),
        'nobg_counts_per_time': ('frame', 'pixel'),
        'nobg_counts_per_time_per_power': ('frame', 'pixel'),
    }
    """ The names of the dataset variables and the dimension they are 
    associated with in a form of a tuple that are used to initialize 
    the xarray dataset holding all the file and processed data. """

    _default_background_per_cycle = {'sif': Qty(300, 'counts'), 'spe': Qty(0, 'counts')}
    """ Default background counts per cycle for each file extension. """

    def __init__(
            self,
            filename_input: AnyString | FilenameManager,
            wavelength_offset: EnhancedNumeric = Qty(0, 'nm'),
            pixel_offset: float = 0,
            background_per_cycle: EnhancedNumeric = None,
            second_order: bool = False):

        metadata = MetadataSpectrum(wfe_offset=wavelength_offset,
                                    pixel_offset=pixel_offset,
                                    background_per_cycle=background_per_cycle,
                                    second_order=second_order)

        super().__init__(filename_input=filename_input, metadata=metadata)

    def _read_file(self):
        """ Reads the spectral data file. """
        if self._metadata.background_per_cycle is None:  # sets default background if None is given by user
            self._metadata.background_per_cycle = \
                self._default_background_per_cycle[self.filename.suffix[1:]]

        if self.filename.suffix == '.sif':
            self._read_sif_file()
        elif self.filename.suffix == '.spe':
            self._read_spe_file()

    def _read_sif_file(self):
        """ Reads the spectral data from an SIF file. """

        # get all data with common SIF file parser
        counts_info, acquisition_info = read_sif(self.filename)

        # set data dimensions
        frame_no = counts_info.shape[0]
        pixel_no = counts_info.shape[-1]
        self._data['frame'] = np.array(range(frame_no))
        self._data['pixel'] = np.array(range(1, pixel_no + 1))

        # get primary data
        self._data['counts'] = ('frame', 'pixel'), Qty(counts_info.reshape((frame_no, pixel_no)), 'count')

        # get exposure time and cycle-count metadata
        self._metadata.exposure_time = Qty(acquisition_info['ExposureTime'], 's')
        self._metadata.cycles = int(acquisition_info['StackCycleTime'] / acquisition_info['ExposureTime'])

        # get pixel and calibration related metadata
        self._metadata.calibration_data = Qty(acquisition_info['Calibration_data'], 'nm')
        self._metadata.pixel_no = pixel_no
        self._metadata.pixels = self._data['pixel']

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

        # set data dimensions
        counts_info = np.asarray(spe_info.data)
        frame_no = counts_info.shape[0]
        pixel_no = counts_info.shape[-1]
        self._data['frame'] = np.array(range(frame_no))
        self._data['pixel'] = np.array(range(pixel_no))

        # get primary data
        self._data['counts'] = ('frame', 'pixel'), Qty(counts_info.reshape((frame_no, pixel_no)), 'count')

        # get exposure time metadata
        exposure_text = 'exposuretime._text'
        value = float(norm_footer_dict[find_partial_key(exposure_text)[0]])
        self._metadata.exposure_time = Qty(value, 'ms').to('s')

        # get cycle-count metadata
        cycles_text = 'cyclecount._text'
        value = int(norm_footer_dict[find_partial_key(cycles_text)[0]])
        self._metadata.cycles = value

        # get pixel and calibration-related data
        self._metadata.pixel_no = pixel_no
        self._metadata.pixels = self._data['pixel']
        self._metadata.calibrated_values = Qty(spe_info.wavelength, 'nm')

    def _set_metadata_after_init(self):
        super()._set_metadata_after_init()
        self._metadata.set_background_per_time_per_power(self._get_excitation_power())

    def _set_data_after_init(self):
        super()._set_data_after_init()

        if len(self._data['frame']) == 1:  # If only one frame was taken, remove frame-related coordinates
            self._data = self._data.isel(frame=0, drop=True)

    def _set_all_data_coords(self):
        """ Sets all the coordinate-related data for the spectral data. """

        # frame-related coordinates
        self._data = self._data.assign_coords({
            'time': (('frame',), self._data['frame'].data * self.metadata.exposure_time),
        })

        # pixel-related coordinates
        wfe = self.metadata.calibrated_wfe
        self._data = self._data.assign_coords({
            'energy': (('pixel',), wfe.energy),
            'frequency': (('pixel',), wfe.frequency),
            'wavelength_vacuum': (('pixel',), wfe.wavelength_vacuum),
            'wavelength_air': (('pixel',), wfe.wavelength_medium),
        })

    def _set_all_data_variables(self):
        """ Sets all the y-axis data for the spectral data. """
        self._data['nobg_counts'] = ('frame', 'pixel'), \
            to_qty_force_units(self.data['counts'].data - self.metadata.background, 'counts')

        self._data['counts_per_cycle'] = ('frame', 'pixel'), \
            to_qty_force_units(self.data['counts'].data / self.metadata.cycles, 'counts')
        self._data['counts_per_time'] = ('frame', 'pixel'), \
            to_qty_force_units(self.data['counts_per_cycle'].data / self.metadata.exposure_time, 'counts/time')

        self._data['nobg_counts_per_cycle'] = ('frame', 'pixel'), \
            to_qty_force_units(self._data['counts_per_cycle'].data - self.metadata.background_per_cycle, 'counts')
        self._data['nobg_counts_per_time'] = ('frame', 'pixel'), \
            to_qty_force_units(self._data['counts_per_time'].data - self.metadata.background_per_time, 'counts/time')

        # define power-related data, if power is provided.
        power_dict = self._get_excitation_power()
        for laser_name, power in power_dict.items():  # for each laser, we get a different power-related data-column.
            data_key = f"counts_per_time_per_{laser_name}_power" if laser_name else "counts_per_time_per_power"

            self._data[data_key] = ('frame', 'pixel'), \
                to_qty_force_units(self.data['counts_per_time'].data / power, 'counts/time/power')

            self._data[f"nobg_{data_key}"] = ('frame', 'pixel'), \
                to_qty_force_units(self._data[data_key].data - self.metadata.background_per_time_per_power[laser_name],
                                   'counts/time/power')

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
        elif lasers is not None:
            power_dict = {str_to_valid_varname(lasers.name): to_qty_force_units(lasers.power, 'power')} \
                if lasers.name is not None else {}
            power_dict[''] = to_qty_force_units(lasers.power, 'power')
        else:
            power_dict = {}

        for key in power_dict.keys():
            if power_dict[key] is None or power_dict[key].m == 0:
                power_dict[key] = np.nan

        return power_dict

    @property
    def metadata(self) -> MetadataSpectrum:
        return self._metadata

