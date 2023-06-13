"""
This module includes the basis classes for reading, storing and processing data and metadata from different files and
experiments.
"""
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pint_xarray
import xarray as xr

from qdl_zno_analysis import Qty
from qdl_zno_analysis._extra_dependencies import plt, sl, xmltodict, read_sif
from qdl_zno_analysis.filename_utils.filename_info import FilenameInfo
from qdl_zno_analysis.filename_utils.filename_manager import FilenameManager
from qdl_zno_analysis.physics import WFE
from qdl_zno_analysis.typevars import AnyString, EnhancedNumeric, PLTArtist
from qdl_zno_analysis.utils import to_qty_force_units, str_to_valid_varname, \
    normalize_dict, varname_to_title_string, integrate_xarray, get_normalized_xarray, quick_plot_xarray, to_qty


class Data:
    """
    A base class to read, store and process data and metadata from a file. Meant to be subclassed.
    """

    _allowed_file_extensions: list[str] = ['qdlf']
    """ A list of the expected file extensions. """
    _qdlf_datatype: str = None
    """ A predetermined string for separating different types of data for in-lab file type QDLF. """

    _data_dim_names: list[str] = ['file_index']
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

    _metadata_dim_names: list[str] = ['file_index']
    """ The names of the dataset dimensions used to initialize
    the xarray dataset holding all the user, file and processed metadata. """
    _metadata_coords: dict[str, tuple[str, ...]] = None
    """ The names of the dataset coordinates and the dimension they 
    are associated with in a form a tuple that are used to initialize 
    the xarray dataset holding all the user, file and processed metadata. """
    _metadata_variables: dict[str, tuple[str, ...]] = None
    """ The names of the dataset variables and the dimension they are 
    associated with in a form of a tuple that are used to initialize 
    the xarray dataset holding all the user, file and processed metadata. """

    def __init__(self, filename_input: AnyString | FilenameManager):
        """
        Initialize a Data object with a filename and metadata.

        Parameters
        ----------
        filename_input: str or Path or FilenameManager
            The name of the file to read, or the filename manager containing the filename of interest.
        """
        if isinstance(filename_input, FilenameManager):
            self.filename_manager = filename_input
        else:
            FilenameManager(filename_input)

        self._check_filename_manager_input()

        # --- initialize data and metadata ---

        # get file_index array
        file_index = np.array(range(len(self.filename_manager.valid_paths)))
        # make empty xarray Dataset with predetermined dimensions.
        self._data: xr.Dataset = xr.Dataset(coords={dim: [] for dim in self._data_dim_names})
        # Set file index coord. will be dropped in post-initialization if there is only one file.
        self._data['file_index'] = file_index

        # Here, the metadata are initialized only from user-provided information.
        # We will add file-dependent information to the metadata object in the _read_file(s) methods,
        # and then update and run all necessary metadata-related routines in the __post_init__() method.
        self._metadata: xr.Dataset = xr.Dataset(coords={dim: [] for dim in self._metadata_dim_names})
        self._metadata['file_index'] = file_index

        self._read_files()
        self.__post_init__()

    def __post_init__(self):
        self._metadata_post_init()
        self._data_post_init()

    def _check_filename_manager_input(self):
        if not self.filename_manager.valid_paths:
            raise FileNotFoundError(f"No such file(s) or directory(/ies): {self.filename_manager.valid_paths}")

        wrong_filetypes = []
        for filetype in self.filename_manager.available_filetypes:
            if filetype not in self._allowed_file_extensions:
                wrong_filetypes.append(wrong_filetypes)
        if len(wrong_filetypes) > 0:
            raise ValueError(f"Filetypes {wrong_filetypes} not in the allowed list of "
                             f"filetypes: {self._allowed_file_extensions}")  # TODO: Change error

    def _read_files(self):
        """ A method to read data from multiple files. calls on `_read_file` multiple times and appends data to a parent
        `xrarray.Dataset`. """
        for file_index in range(len(self.filename_manager.valid_paths)):
            self._read_file(file_index)

    def _read_file(self, file_index: int):
        """
        A method to read data from the file. Must be overloaded in subclass.

        First initialize the data coordinates, and then the variables that depend on them.
        """
        warnings.warn('Define your own get_data() function')
        ...

    def _metadata_post_init(self):
        """ A method to set metadata after initialization. May be overloaded in subclass. """
        self._set_all_metadata_coords()
        self._set_all_metadata_variables()
        if len(self._metadata['file_index']) == 1:  # If only one file was provided, remove file-related coordinates
            self._metadata = self._metadata.isel(file_index=0, drop=True)

    def _set_all_metadata_coords(self):
        """ A method to set metadata coordinates after reading the file(s). May be overloaded in subclass. """
        pass

    def _set_all_metadata_variables(self):
        """ A method to set metadata variables after reading the file(s). May be overloaded in subclass. """
        pass

    def _data_post_init(self):
        """ A method to set more data variables after reading the file(s). May be overloaded in subclass. """
        self._set_all_data_coords()
        self._set_all_data_variables()
        if len(self._data['file_index']) == 1:  # If only one file was provided, remove file-related coordinates
            self._data = self._data.isel(file_index=0, drop=True)

    def _set_all_data_coords(self):
        """ A method to set data coordinates after reading the file(s). May be overloaded in subclass. """
        pass

    def _set_all_data_variables(self):
        """ A method to set data variables after reading the file(s). May be overloaded in subclass. """
        pass

    def _set_quick_plot_labels(self):
        """
        A method to set quick plot labels for the data columns.
        Takes column names and turns them to title-styled strings,
        and assigns them to the xarray dedicated attribute key that
        is used for easy plots.
        """
        for key in self._data.variables.keys():  # parse data columns
            key = str(key)
            label = varname_to_title_string(key.replace('_per_', '/'), '/')  # get title-styled string
            self._data[key].attrs['standard_name'] = label

    def _check_column_validity(self, string):
        """ A method to check if a data column exists. """
        if string not in self.data.variables.keys():
            raise ValueError(f"Axis must be one of {list(self.data.variables.keys())}, not {string}")

    @property
    def data(self) -> xr.Dataset:
        """ Returns the xarray dataset holding all the file and processed data. """
        return self._data

    @property
    def metadata(self) -> xr.Dataset:
        """ Returns the xarray dataset holding all the user, file and processed data. """
        return self._metadata

    @property
    def filename_info(self) -> FilenameInfo | list[FilenameInfo]:
        """ Returns the input filename FilenameInfo dataclass. """
        if len(self.filename_manager.filename_info_list) == 1:
            return self.filename_manager.filename_info_list[0]
        else:
            return self.filename_manager.filename_info_list

    @property
    def filename_info_dict(self) -> dict[str, Any] | list[dict[str, Any]]:
        """ Returns the input filename FilenameInfo dictionary. """
        if len(self.filename_manager.filename_info_dicts) == 1:
            return self.filename_manager.filename_info_dicts[0]
        else:
            return self.filename_manager.filename_info_dicts

    @property
    def filename_info_norm_dict(self) -> dict[str, Any] | list[dict[str, Any]]:
        """ Returns the input filename FilenameInfo normalized dictionary. """
        if len(self.filename_manager.filename_info_norm_dicts) == 1:
            return self.filename_manager.filename_info_norm_dicts[0]
        else:
            return self.filename_manager.filename_info_norm_dicts

    @property
    def filename(self) -> Path | list[Path]:
        """ Returns the input filename. """
        if len(self.filename_manager.valid_paths) == 1:
            return self.filename_manager.valid_paths[0]
        else:
            return self.filename_manager.valid_paths

    def integrate(
            self,
            start: EnhancedNumeric | None = None,
            end: EnhancedNumeric | None = None,
            coord: str | None = None,
            var: str | None = None,
    ) -> xr.Dataset:
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
    ) -> xr.Dataset:
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
    ) -> PLTArtist | list[PLTArtist]:
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
    _allowed_file_extensions: list[str] = Data._allowed_file_extensions + ['sif', 'spe']
    """ List of allowed file extensions for spectral data. """

    _qdlf_datatype: str = "spectrum"
    """ QDLF datatype for spectral data. """

    _data_dim_names: list[str] = Data._data_dim_names + ['frame', 'pixel']

    _data_coords: dict[str, tuple[str, ...]] = {
        'time': ('frame',),
        'wavelength_air': ('pixel',),
        'wavelength_vacuum': ('pixel',),
        'frequency': ('pixel',),
        'energy': ('pixel',),
    }

    _data_variables: dict[str, tuple[str, ...]] = {
        'counts': ('file_index', 'frame', 'pixel'),
        'counts_per_cycle': ('file_index', 'frame', 'pixel'),
        'counts_per_time': ('file_index', 'frame', 'pixel'),
        'counts_per_time_per_power': ('file_index', 'frame', 'pixel'),
        'nobg_counts': ('file_index', 'frame', 'pixel'),
        'nobg_counts_per_cycle': ('file_index', 'frame', 'pixel'),
        'nobg_counts_per_time': ('file_index', 'frame', 'pixel'),
        'nobg_counts_per_time_per_power': ('file_index', 'frame', 'pixel'),
    }

    _metadata_dim_names: list[str] = Data._metadata_dim_names + ['pixel']
    _metadata_coords: dict[str, tuple[str, ...]] = {}
    _metadata_variables: dict[str, tuple[str, ...]] = {
        'diffraction_order': (),
        'calibration_data': (),

        'wfe_offset': ('file_index',),
        'pixel_offset': ('file_index',),
        'exposure_time': ('file_index',),
        'cycles': ('file_index',),
        'calibrated_values': ('pixel',),

        'background': ('file_index', 'pixel'),
        'background_per_cycle': ('file_index', 'pixel'),
        'background_per_time': ('file_index', 'pixel'),
        'background_time_per_power': ('file_index', 'pixel'),
    }

    _default_background_per_cycle = {'sif': Qty(300, 'counts'), 'spe': Qty(0, 'counts')}
    """ Default background counts per cycle for each file extension. """

    @dataclass
    class __UserInfo:
        wfe_offset: EnhancedNumeric = Qty(0, 'nm')
        """ Wavelength/Frequency/Energy. Default is Qty(0, 'nm'). """
        pixel_offset: float = 0
        """ Pixel offset. Default is 0. """
        background_per_cycle: EnhancedNumeric | np.ndarray | list[EnhancedNumeric] | xr.DataArray = Qty(0, 'counts')
        """ Background counts per cycle. Default is Qty(0, 'counts'). The dimensions of the background can be up to 
        (file_index, pixel). Deepest axis will be assumed to be pixels, and the other axis will be file_index. 
        Frames can not have different backgrounds. """
        diffraction_order: int = 1
        """ The expected diffraction order of the observed spectrum. This will not change the refractive index. 
        Default is 1. """

    def __init__(
            self,
            filename_input: AnyString | FilenameManager,
            wfe_offset: EnhancedNumeric = Qty(0, 'nm'),
            pixel_offset: float = 0,
            background_per_cycle: EnhancedNumeric | np.ndarray | list = np.nan,
            diffraction_order: int = 1,
    ):
        # make a small user input dataclass from this.
        self._user_info = self.__UserInfo(
            to_qty(wfe_offset, 'length'),
            pixel_offset,
            to_qty_force_units(background_per_cycle, 'counts'),
            diffraction_order
        )

        super().__init__(filename_input)

    def _read_file(self, file_index: int):
        """ Reads the spectral data file. """

        file_path = self.filename_manager.valid_paths[file_index]
        # if self.metadata.background_per_cycle is None:  # sets default background if None is given by user
        #     self.metadata.background_per_cycle = \
        #         self._default_background_per_cycle[file_path.suffix[1:]]

        if file_path.suffix == '.sif':
            self._read_sif_file(file_index)
        elif file_path.suffix == '.spe':
            self._read_spe_file(file_index)

    def _read_sif_file(self, file_index: int):
        """ Reads the spectral data from a SIF file. """

        # get file path
        file_path = self.filename_manager.valid_paths[file_index]

        # get all data with common SIF file parser
        counts_info, acquisition_info = read_sif(file_path)

        # set data dimensions
        frame_no = counts_info.shape[0]
        pixel_no = counts_info.shape[-1]
        if len(self._data['frame']) == 0:  # if this is the first entry
            # get frame and pixel coordinates
            self._data['frame'] = np.array(range(frame_no))
            self._data['pixel'] = np.array(range(1, pixel_no + 1))

            # initialize empty data array
            self._data['counts'] = xr.DataArray(coords=self._data.coords)
            self._data['counts'] = self.data['counts'].pint.quantify('count')

            # set pixel coordinate in metadata
            self._metadata['pixel'] = self.data['pixel'].data

            # get calibration data -> We assume they are the same over all files!!
            self._metadata['calibration_data'] = (), Qty(set(acquisition_info['Calibration_data']), 'nm')

            # initialize empty exposure time and cycle metadata variables
            self._metadata['exposure_time'] = xr.DataArray(
                dims=['file_index'], coords={'file_index': self.metadata['file_index']})
            self._metadata['exposure_time'] = self.metadata['exposure_time'].pint.quantify('s')

            self._metadata['cycles'] = xr.DataArray(
                dims=['file_index'], coords={'file_index': self.metadata['file_index']})
        else:
            if self._data['frame'] != np.array(range(frame_no)):
                raise ValueError('...')  # TODO: change error

        # get primary data
        self._data['counts'][file_index] = Qty(counts_info.reshape((frame_no, pixel_no)), 'count')

        # get exposure time and cycles. These can vary between different files.
        self._metadata['exposure_time'][file_index] = Qty(acquisition_info['ExposureTime'], 's')
        self.metadata['cycles'][file_index] = int(acquisition_info['StackCycleTime'] / acquisition_info['ExposureTime'])

    def _read_spe_file(self, file_index: int):
        """ Reads the spectral data from an SPE file. """

        # get file path
        file_path = self.filename_manager.valid_paths[file_index]

        # get data with common SPE file reader
        spe_info: sl.SpeFile = sl.load_from_files([str(file_path)])

        # get normalized footer
        with open(self.filename) as f:
            footer_pos = sl.read_at(f, 678, 8, np.uint64)[0]
            f.seek(footer_pos)
            xml_text = f.read()
            footer_dict = xmltodict.parse(xml_text)

        norm_footer_dict = normalize_dict(footer_dict)

        def find_partial_key(text: str):
            return [key for key in norm_footer_dict if text in key]

        # set data dimensions
        counts_info = np.asarray(spe_info.data)
        frame_no = counts_info.shape[0]
        pixel_no = counts_info.shape[-1]
        if len(self._data['frame']) == 0:  # if this is the first entry
            # get frame and pixel coordinates
            self._data['frame'] = np.array(range(frame_no))
            self._data['pixel'] = np.array(range(pixel_no))

            # initialize empty data array
            self._data['counts'] = xr.DataArray(coords=self._data.coords)
            self.data['counts'] = self.data['counts'].pint.quantify('count')

            # get calibration values -> We assume they are the same over all files!!
            self.metadata.calibrated_values = Qty(spe_info.wavelength, 'nm')

            # set pixel-related metadata variables
            self.metadata.pixel_no = pixel_no
            self.metadata.pixels = self.data['pixel'].data

        else:
            if self._data['frame'] != np.array(range(frame_no)):
                raise ValueError('...')  # TODO: change error

        # get primary data
        self._data['counts'][file_index] = Qty(counts_info.reshape((frame_no, pixel_no)), 'count')

        # get exposure time. This can vary between different files.
        exposure_text = 'exposuretime._text'
        value = float(norm_footer_dict[find_partial_key(exposure_text)[0]])
        self.metadata.exposure_time = np.append(self.metadata.exposure_time, Qty(value, 'ms').to('s'))

        # get cycle-counts. This can vary between different files.
        cycles_text = 'cyclecount._text'
        value = int(norm_footer_dict[find_partial_key(cycles_text)[0]])
        self.metadata.cycles = np.append(self.metadata.cycles, value)

    def _set_all_metadata_coords(self):
        """ Sets all the coordinate-related metadata for the spectral data. """
        pass

    def _set_all_metadata_variables(self):
        """ Sets all the "y-axis" metadata for the spectral data. """

        if np.all(self.metadata['exposure_time'] == self.metadata['exposure_time'][0]):
            self._metadata['exposure_time'] = (), self.metadata['exposure_time'].data[0]

        if np.all(self.metadata['cycles'] == self.metadata['cycles'][0]):
            self._metadata['cycles'] = (), self.metadata['cycles'].data[0]

        self._metadata['wfe_offset'] = self._user_info.wfe_offset
        self._metadata['pixel_offset'] = self._user_info.pixel_offset
        self._metadata['diffraction_order'] = self._user_info.diffraction_order

        if 'calibrated_values' not in self.metadata.variables \
                and 'calibration_data' in self.metadata.variables:
            self._metadata['calibrated_values'] = ('pixel',), self._applied_calibration()

        bgpc = self._user_info.background_per_cycle
        if isinstance(bgpc, xr.DataArray):
            self._metadata['background_per_cycle'] = bgpc
        else:
            if np.all(np.isnan(bgpc)):
                if 'spe' in self.filename_manager.available_filetypes:
                    bgpc = self._default_background_per_cycle['spe']
                else:
                    bgpc = self._default_background_per_cycle['sif']

            coord_names = self._find_metadata_coord_names(bgpc)
            self._metadata['background_per_cycle'] = coord_names, bgpc

        self._metadata['background'] = \
            self._metadata['background_per_cycle'] * self.metadata['cycles']
        self._metadata['background_per_time'] = \
            self._metadata['background_per_cycle'] / self.metadata['exposure_time']

        # TODO: Add this to Data, before reading files!!
        for key, value in self.filename_manager.non_changing_filename_info_dict.items():
            if np.ndim(value) == 0:
                self._metadata[key] = (), value
            elif isinstance(value, Qty):
                self._metadata[key] = (), Qty(set(value.m), value.u)
            else:
                self._metadata[key] = (), np.array(set(value))

        for key, value in self.filename_manager.changing_filename_info_dict.items():
            if np.ndim(value) == 1:
                self._metadata[key] = ('file_index',), value
            else:
                self._metadata[key] = ('file_index',), set(value)  # TODO: fix, will not work.

        excitation_power_dict = self._get_excitation_power_dict()
        for laser_power_name, laser_power in excitation_power_dict.items():
            data_key = f"background_per_time_per_{laser_power_name}"
            self._metadata[data_key] = self.metadata['background_per_time'] / laser_power

    def _data_post_init(self):
        super()._data_post_init()

        if len(self._data['frame']) == 1:  # If only one frame was taken, remove frame-related coordinates
            self._data: xr.Dataset = self._data.isel(frame=0, drop=True)

    def _set_all_data_coords(self):
        """ Sets all the coordinate-related data for the spectral data. """

        self._data = self._data.assign_coords({
            'time': self._data['frame'] * self.metadata.exposure_time
        })

        # pixel-related coordinates
        # get calibrated WFE
        calibrated_values = self._metadata['calibrated_values'].data.copy()
        diff_ord = self._metadata['diffraction_order']
        if diff_ord != 1:
            calibrated_values *= 1 / diff_ord if calibrated_values.check('[length]') else diff_ord
        wfe = WFE(calibrated_values)  # assumes air as medium

        self._data = self._data.assign_coords({
            'energy': (('pixel',), wfe.energy),
            'frequency': (('pixel',), wfe.frequency),
            'wavelength_vacuum': (('pixel',), wfe.wavelength_vacuum),
            'wavelength_air': (('pixel',), wfe.wavelength_medium),
        })

    def _set_all_data_variables(self):
        """ Sets all the "y-axis" data for the spectral data. """
        self._data['counts_per_cycle'] = self.data['counts'] / self.metadata['cycles']
        self._data['counts_per_time'] = self.data['counts_per_cycle'] / self.metadata['exposure_time']

        excitation_power_dict = self._get_excitation_power_dict()
        for laser_power_name, laser_power in excitation_power_dict.items():
            data_key = f"counts_per_time_per_{laser_power_name}"
            self._data[data_key] = self.data['counts_per_time'] / laser_power

        self._data['nobg_counts'] = \
            self.data['counts'] - self.metadata['background']
        self._data['nobg_counts_per_cycle'] = \
            self.data['counts_per_cycle'] - self.metadata['background_per_cycle']
        self._data['nobg_counts_per_time'] = \
            self.data['counts_per_time'] - self.metadata['background_per_time']

        for laser_power_name in excitation_power_dict:
            data_key = f"counts_per_time_per_{laser_power_name}"
            bg_key = f"background_per_time_per_{laser_power_name}"
            self._data[f'nobg_{data_key}'] = self._data[data_key] - self.metadata[bg_key]

    def _get_excitation_power_dict(self) -> dict[str, xr.DataArray]:
        """
        Finds the excitation power(s) of the laser(s) used to collect the spectrum.
        Finds the total laser power as well.

        Returns
        -------
        A dictionary of the form {laser_name: excitation_power} for each laser used, and a key with an empty string for
        the total excitation power.
        """

        laser_power_names: list[str] = [str(key) for key in list(self._metadata.variables.keys())
                                        if str(key).startswith('lasers.') and str(key).endswith('.power')]

        if 'lasers.name' in self.metadata.variables.keys():
            excitation_power_dict = {'power': to_qty_force_units(self.metadata['lasers.power'], 'power')}
        else:
            excitation_power_dict = \
                {lpn.lstrip('lasers.').replace('.', '_'): to_qty_force_units(self.metadata[lpn], 'power')
                 for lpn in laser_power_names}

            excitation_power_dict['power'] = np.sum([self.metadata[lpn] for lpn in laser_power_names])

        for key in excitation_power_dict.keys():
            if excitation_power_dict[key].data is None or excitation_power_dict[key].data.m == 0:
                excitation_power_dict[key] = to_qty_force_units(np.nan, 'power')

        return excitation_power_dict

    def _applied_calibration(self) -> Qty:
        """ Apply the calibration data to the pixel list in order to obtain calibrated values
        (in wavelength, frequency or energy). """

        pixels = np.asarray(self.metadata['pixel'].data) + self._metadata['pixel_offset'].data

        calib_data = self._metadata['calibration_data'].data
        calib_data = Qty(list(calib_data.m.flatten()[0]), calib_data.u)  # unravels set from numpy array...
        result = Qty(np.zeros(pixels.shape), calib_data.u)
        for i, cal_coefficient in enumerate(calib_data):
            result += cal_coefficient * pixels ** i

        return result + self._metadata['wfe_offset'].data

    def _find_metadata_coord_names(self, xdata_array) -> tuple[str, ...]:
        data_shape = np.shape(xdata_array)
        file_no = len(self.data['file_index'])
        pixel_no = len(self.data['pixel'])

        if len(data_shape) == 0:
            coord_names = ()
        elif len(data_shape) == 1 and data_shape[0] == file_no:
            coord_names = ('file_index',)
        elif len(data_shape) == 1 and data_shape[0] == pixel_no:
            coord_names = ('pixel',)
        elif len(data_shape) == 2:
            coord_names = ('file_index', 'pixel')
        else:
            raise ValueError('...')  # TODO: Change error message

        return coord_names
