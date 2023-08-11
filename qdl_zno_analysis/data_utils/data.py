"""
This module includes the base classes for reading, storing and processing data and metadata from different files and
experiments.
Metadata includes metadata saved in the files, as well as extra information passed on the filenames.
"""
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pint_xarray  # needed for unit management in xarrays, even if IDE says it's not used.
import xarray as xr

from qdl_zno_analysis import Qty
from qdl_zno_analysis._extra_dependencies import plt, sl, xmltodict, read_sif
from qdl_zno_analysis.errors import assert_options, ValueOutOfOptionsError, IsNullError, ArrayShapeError
from qdl_zno_analysis.filename_utils.filename_info import FilenameInfo
from qdl_zno_analysis.filename_utils.filename_manager import FilenameManager
from qdl_zno_analysis.physics import WFE
from qdl_zno_analysis.typevars import AnyString, EnhancedNumeric, PLTArtist, MultiAnyString
from qdl_zno_analysis.utils import to_qty_force_units, normalize_dict, varname_to_title_string, \
    integrate_xarray, get_normalized_xarray, quick_plot_xarray, to_qty, indexify_xarray_coord, Dataclass, find_outliers


class Data:
    """
    A base class to read, store and process data and metadata from a file. Meant to be subclassed.
    """

    _ALLOWED_FILE_EXTENSIONS: list[str] = ['qdlf']
    """ A list of the expected file extensions. """
    _QDLF_DATATYPE: str = None
    """ A predetermined string for distinguishing different types of data for in-lab file type QDLF. """

    _DATA_DIM_NAMES: list[str] = ['file_index']
    """ The names of the dataset dimensions used to initialize
    the xarray dataset holding all the file and processed data. """
    _DATA_COORDS: dict[str, tuple[str, ...]] = None
    """ The names of the dataset coordinates and the dimension they 
    are associated with, in the form of a tuple that are used to initialize 
    the xarray dataset holding all the file and processed data. """
    _DATA_VARIABLES: dict[str, tuple[str, ...]] = None
    """ The names of the dataset variables and the dimension they are 
    associated with, in the form of a tuple that are used to initialize 
    the xarray dataset holding all the file and processed data. """

    _METADATA_DIM_NAMES: list[str] = ['file_index']
    """ The names of the dataset dimensions used to initialize
    the xarray dataset holding all the user, file and processed metadata. """
    _METADATA_COORDS: dict[str, tuple[str, ...]] = None
    """ The names of the dataset coordinates and the dimension they 
    are associated with, in the form a tuple that are used to initialize 
    the xarray dataset holding all the user, file and processed metadata. """
    _METADATA_VARIABLES: dict[str, tuple[str, ...]] = None
    """ The names of the dataset variables and the dimension they are 
    associated with, in the form of a tuple that are used to initialize 
    the xarray dataset holding all the user, file and processed metadata. """

    def __init__(self, filename_input: AnyString | MultiAnyString | FilenameManager):
        """
        Initialize a Data object with a filename(s) or a `FilenameManager` object.

        Parameters
        ----------
        filename_input: str | Path | List[str] | List[Path] | FilenameManager
            The name(s) of the file(s) to read, or the filename manager containing the filename(s) of interest.
            This class will use the `FilenameManager` instance internally, regardless of input.
        """
        #
        if isinstance(filename_input, FilenameManager):
            self.filename_manager = filename_input
        else:
            FilenameManager(filename_input)

        self._check_filename_manager_input()

        # --- initialize data and metadata ---

        # Get file_index array.
        file_index = np.array(range(len(self.filename_manager.valid_paths)))
        # Make empty xarray Dataset with predetermined dimensions.
        self._data: xr.Dataset = xr.Dataset(coords={dim: [] for dim in self._DATA_DIM_NAMES})
        # Set file index coord_array. Will be dropped in post-initialization if there is only one file.
        self._data['file_index'] = file_index

        # Here, the metadata are initialized only from user-provided information.
        # We will add file-dependent information to the metadata object in the _read_file(s) methods,
        # and then update and run all necessary metadata-related routines in the __post_init__() method.
        self._metadata: xr.Dataset = xr.Dataset(coords={dim: [] for dim in self._METADATA_DIM_NAMES})
        self._metadata['file_index'] = file_index

        self._insert_filename_info_to_datasets()

        self._read_files()
        self.__post_init__()

    def __post_init__(self):
        """ Perform post-initialization tasks for metadata and data. """
        self._metadata_post_init()
        self._data_post_init()

    def _check_filename_manager_input(self):
        """ Make sure that the `filename_manager` contains valid paths,
        and that the contained filetypes are allowed. """

        # check paths
        if not self.filename_manager.valid_paths:
            raise IsNullError(
                self.filename_manager.valid_paths,
                'filename_manager.valid_paths'
            )

        # check filetypes
        assert_options(
            self.filename_manager.available_filetypes,
            self._ALLOWED_FILE_EXTENSIONS,
            'filename_manager.available_filetypes',
            ValueOutOfOptionsError,
        )

    def _insert_filename_info_to_datasets(self):
        """ Take all filename information, find which ones change between files and which don't,
         and add the non-changing ones in the metadata, and the changing ones in both data and metadata. """
        for key, value in self.filename_manager.non_changing_filename_info_dict.items():
            if np.ndim(value) == 0:
                self._metadata[key] = (), value
            elif isinstance(value, Qty):
                self._metadata[key] = (), Qty(set(value.m), value.u)
            else:
                self._metadata[key] = (), np.array(set(value))

        for key, value in self.filename_manager.changing_filename_info_dict.items():
            if isinstance(value[0], Qty):
                value = Qty.from_list(value)

            if np.ndim(value) == 1:
                self._metadata = self._metadata.assign_coords({key: (('file_index',), value)})
                self._data = self._data.assign_coords({key: (('file_index',), value)})
                # self._data = indexify_xarray_coord(self.data, key, add_only_if_necessary=True, verbose=False)
            else:
                self._metadata[key] = ('file_index',), set(value)  # TODO: fix, will not work.
                self._data[key] = ('file_index',), set(value)  # TODO: fix, will not work.

    def _read_files(self):
        """ A method to read data from multiple files. calls on `_read_file` multiple times and appends data to a parent
        `xrarray.Dataset`. """
        for file_index in range(len(self.filename_manager.valid_paths)):
            self._read_file(file_index)

    def _read_file(self, file_index: int):
        """
        A method to read data from the file. Must be overloaded in subclass.

        First initialize the data dimensions, and then the variables that depend on them.
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

    def _assert_data_variable_validity(self, string):
        """ A method to check if a data variable, coordinate or dimension exists. """
        assert_options(
            string,
            set(self.data.variables.keys()),
            'coordinate',
            ValueOutOfOptionsError,
        )

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

        excitation_power_dict = {}
        if 'lasers.name' in self.metadata.variables.keys():
            excitation_power_dict = {'power': to_qty_force_units(self.metadata['lasers.power'], 'power')}
        elif len(laser_power_names) > 0:
            excitation_power_dict = \
                {lpn.lstrip('lasers.').replace('.', '_'): to_qty_force_units(self.metadata[lpn], 'power')
                 for lpn in laser_power_names}

            excitation_power_dict['power'] = \
                to_qty_force_units(np.sum([self.metadata[lpn] for lpn in laser_power_names]), 'power')

        # replacing all 0 and None values with np.nan
        for key in excitation_power_dict.keys():
            if excitation_power_dict[key] is None:
                excitation_power_dict[key] = to_qty_force_units(np.nan, 'power')
                continue
            exc_power: Qty = excitation_power_dict[key].data
            if np.ndim(exc_power) > 0:
                exc_power[np.where(exc_power == 0)] = np.nan
                excitation_power_dict[key].data = exc_power
            elif exc_power.m == 0:
                excitation_power_dict[key] = to_qty_force_units(np.nan, 'power')

        return excitation_power_dict

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
            mode: str = 'nearest',
    ) -> xr.Dataset:
        """
        Integrates the area under the curve of the data array/set within a given range.

        Parameters
        ----------
        start: int | float | pint.Quantity | None, optional
            The start of the integration range. Defaults to the first element of the coordinate array.
            Must be within the range of the coordinate.
        end: int | float | pint.Quantity | None, optional
            The end of the integration range. Defaults to the last element of the coordinate array.
            Must be within the range of the range of the coordinate.
        coord: str | None, optional
            The data coordinate that will be used as the x-axis of the integration.
            Defaults to the last (deepest level) dimension of the dataset.
        var: str | None, optional
            The data variable that will be used as the y-axis of the integration.
            Defaults to the entire dataset.
        mode: str {'nearest', 'interp'}, optional
            - If 'nearest', uses the edge points nearest to the existing dimension-points
            - If 'interp', and if the start or end points are not values in the coordinate data array (x-axis),
            this method uses cubic interpolation to find the y-axis values at the given start or end points.
            Defaults to 'nearest.'

        Returns
        -------
        xr.Dataset | xr.DataArray
            The reduced data array/set of the integrated array values.
        """

        return integrate_xarray(self.data, start, end, coord, var, mode)

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
        norm_axis_val: int | float | Qty | xr.DataArray | np.ndarray | None, optional
            The value used for normalization along the `norm_axis`.
            If None, the maximum value of the normalization axis is used.
        norm_axis: str | None, optional
            The axis used for normalization. If None, the last axis (the deepest level) is used.
        norm_var: str | None, optional
            The variable used for normalization. If None, the first variable in the dataset is used.
        mode: {'nearest', 'linear', 'quadratic', 'cubic'}, optional
            The interpolation data_aggregation_method for finding norm_axis values.
            'nearest' - Find the nearest `norm_axis` to the `norm_axis_val` .
            'linear' - Perform linear interpolation between adjacent `norm_axis` values.
            'quadratic' - Perform quadratic interpolation between adjacent `norm_axis` values.
            'cubic' - Perform cubic interpolation between adjacent `norm_axis` values.
            Default is 'nearest.'
        subtract_min: bool, optional
            If True, subtract the minimum values from the data before normalization.

        Returns
        -------
        xarray.Dataset
            The normalized data.
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
        var: str | None, optional
            Name of the data variable to use as the y-axis for 2D data and z-axis for 3D data.
            If None, the first variable in the dataset is used.
        coord1: str | None, optional
            Name of the coordinate to use as the x-axis.
        coord2: str | None, optional
            Name of the coordinate to use as the y-axis (only for 3D data).
        var_units: str | None, optional
            The units to change the plotted variable to.
        coord1_units: str | None, optional
            The units to change the x-axis to.
        coord2_units: str | None, optional
            The units to change the y-axis to (only for 3D data).
        plot_method: str | None, optional
            The specific plotting method to use. See `xarray.DataArray.plot` for options.
            If None, the default method for the input variable will be used.
        **plot_kwargs
            Additional keyword arguments to pass to the matplotlib.pyplot.plot function.

        Returns
        -------
        plt.Artist
            The object returned by the `xarray.DataArray.plot` method.

        Notes
        -----
        The legend label for the line will be set to the filename or file number if it is not provided by the user.
        """

        result = quick_plot_xarray(self.data, var, coord1, coord2, var_units,
                                   coord1_units, coord2_units, plot_method, **plot_kwargs)

        # if not isinstance(result, Sequence):
        #     result_parse = [result]
        # else:
        #     result_parse = result
        #
        # for i, res in enumerate(result_parse):
        #     if res.get_label().startswith('_child'):
        #         fno = self.filename_manager.filename_info_dicts[i].get('file_number', None)
        #         label = fno if fno is not None else str(self.filename_manager.valid_paths[i])
        #         res.set_label(label)

        return result


class DataSpectrum(Data):
    """
    A class for handling spectral data files, such as sif and spe.
    """
    _ALLOWED_FILE_EXTENSIONS: list[str] = Data._ALLOWED_FILE_EXTENSIONS + ['sif', 'spe']
    """ List of allowed file extensions for spectral data. """

    _QDLF_DATATYPE: str = "spectrum"
    """ QDLF datatype for spectral data. """

    _DATA_DIM_NAMES: list[str] = Data._DATA_DIM_NAMES + ['frame', 'pixel']

    _DATA_COORDS: dict[str, tuple[str, ...]] = {
        'time': ('file_index', 'frame'),
        'wavelength_air': ('pixel',),
        'wavelength_vacuum': ('pixel',),
        'frequency': ('pixel',),
        'energy': ('pixel',),
    }

    _DATA_VARIABLES: dict[str, tuple[str, ...]] = {
        'counts': ('file_index', 'frame', 'pixel'),
        'counts_per_cycle': ('file_index', 'frame', 'pixel'),
        'counts_per_time': ('file_index', 'frame', 'pixel'),
        'counts_per_time_per_power': ('file_index', 'frame', 'pixel'),
        'nobg_counts': ('file_index', 'frame', 'pixel'),
        'nobg_counts_per_cycle': ('file_index', 'frame', 'pixel'),
        'nobg_counts_per_time': ('file_index', 'frame', 'pixel'),
        'nobg_counts_per_time_per_power': ('file_index', 'frame', 'pixel'),
    }

    _METADATA_DIM_NAMES: list[str] = Data._METADATA_DIM_NAMES + ['pixel']
    _METADATA_COORDS: dict[str, tuple[str, ...]] = {}
    _METADATA_VARIABLES: dict[str, tuple[str, ...]] = {
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
    class CosmicRayFilter(Dataclass):
        rolling_window_size: int = 5
        """ Size of the rolling window for calculating statistics, by default 5 points 
        (must be at least 2 * maximum_outlier_cluster_size + 1). """
        flagging_significance: float = 3.
        """ Significance factor (in units of rolling standard deviation) for flagging outliers, by default 5.0.
        The smaller the value the stricter the filtering (more outliers will be found). """
        maximum_outlier_cluster_size: int = 2
        """ Maximum size of outlier clusters to consider, by default 1. """
        hard_min_limit: EnhancedNumeric | np.ndarray = np.nan
        """ Either a value or an array of values with dimension size up to the same as `y_data`,
        this attribute is a minimum cut-off for the `y_data` that does not depend on the rolling statistics. """
        hard_max_limit: EnhancedNumeric | np.ndarray = np.nan
        """ Either a value or an array of values with dimension size up to the same as `y_data`,
        this attribute is a maximum cut-off for the `y_data` that does not depend on the rolling statistics. """
        rolling_data_method: str = 'median'
        """ Method for calculating rolling data statistics. Options are 'mean' and 'median', by default 'median'. """
        approximate_local_stdev_method: str = 'mean'
        """ Method for calculating the expected data jitter (mean/median local stdev of data). Defaults to 'mean'.
        If there is a small dataset with huge outliers, it would be better to use median. For larger
        datasets with smaller outliers, 'mean' may be better. """
        repeat: int = -1
        """ The amount of times this process will be repeated. Repetition will help when there are
        different levels of outliers in the data (some very strong, some softer, but still stronger than expected).
        The special value "-1" (default) will repeat until the average of the rolling stdev is stabilized. """

    @dataclass
    class __UserInfo(Dataclass):
        """
        This is an internal class that helps define subclass specific user inputs.
        """
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
        heal_cosmic_rays: bool = False
        """ If True, runs a `utils.find_outliers` routine to find outliers and replace their values with 
        a new estimated value. Fine-tune the cosmic ray filter by defining the `cosmic_ray_filter` parameter.
        Defaults to False. """
        cosmic_ray_filter: "DataSpectrum.CosmicRayFilter" = None
        """ Specific settings to be passed to the `utils.find_outliers` method, via the DataSpectrum.CosmicRayFilter
        dataclass."""

    def __init__(
            self,
            filename_input: AnyString | FilenameManager,
            wfe_offset: EnhancedNumeric = Qty(0., 'nm'),
            pixel_offset: float = 0.,
            background_per_cycle: EnhancedNumeric | np.ndarray | list = np.nan,
            diffraction_order: int = 1,
            heal_cosmic_rays: bool = False,
            cosmic_ray_filter: CosmicRayFilter = CosmicRayFilter(),
    ):
        """
        Initializes a DataSpectrum object.

        Parameters
        ----------
        filename_input: str | Path | List[str] | List[Path] | FilenameManager
            The name(s) of the file(s) to read, or the filename manager containing the filename(s) of interest.
            This class will use the `FilenameManager` instance internally, regardless of input.
        wfe_offset: int | float | Qty
            The wavelength/frequency/energy (WFE) offset of the spectrum. Default is Qty(0, 'nm').
        pixel_offset: int
            The pixel offset of the spectrum. Default is 0.
        background_per_cycle: int | float | Qty | np.ndarray | list
            The background counts per cycle. Default is Qty(0, 'counts'). The dimensions of the background can be up to
            (file_index, pixel). The deepest axis will be assumed to be pixels, and the other axis will be file_index.
            Frames can not have different backgrounds. Defaults to 300 counts for sif files and 0 counts for spe files.
        diffraction_order: int
            The expected diffraction order of the observed spectrum. This will not change the refractive index that is
            used to calculate the calibrated WFE values. Default is 1.
        heal_cosmic_rays: bool, optional
            If True, runs a `utils.find_outliers` routine to find outliers and replace their values with
            a new estimated value. Fine-tune the cosmic ray filter by defining the `cosmic_ray_filter` parameter.
            This filter is not good at removing cosmic rays very close to the edge of the spectrum.
            Defaults to False.
        cosmic_ray_filter: DataSpectrum.CosmicRayFilter, optional
            Specific settings to be passed to the `utils.find_outliers` method, via the DataSpectrum.CosmicRayFilter
            dataclass.
        """
        self._user_info = self.__UserInfo(
            to_qty(wfe_offset, 'length'),
            pixel_offset,
            to_qty_force_units(background_per_cycle, 'counts'),
            diffraction_order,
            heal_cosmic_rays,
            cosmic_ray_filter,
        )

        super().__init__(filename_input)

    def _read_file(self, file_index: int):
        """ Reads the spectral data file. """

        file_path = self.filename_manager.valid_paths[file_index]
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
            self._data['counts'] = self.data['counts'].pint.quantify('count')  # add units to empty array

            # set pixel coordinate in metadata
            self._metadata['pixel'] = self.data['pixel'].data

            # get calibration data -> We assume they are the same over all files!!
            self._metadata.attrs['calibration_data'] = Qty(acquisition_info['Calibration_data'], 'nm')

            # initialize empty exposure time and cycle metadata variables
            self._metadata['exposure_time'] = xr.DataArray(
                dims=['file_index'], coords={'file_index': self.metadata['file_index']})
            self._metadata['exposure_time'] = self.metadata['exposure_time'].pint.quantify('s')

            self._metadata['cycles'] = xr.DataArray(
                dims=['file_index'], coords={'file_index': self.metadata['file_index']})
        else:
            if len(self._data['frame'].data) < frame_no:
                # TODO: Add warning that size is changing
                self._data = self.data.reindex(y=np.array(range(frame_no)), fill_value=np.nan)

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
        with open(str(file_path)) as f:
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
            self.data['counts'] = self.data['counts'].pint.quantify('count')  # add units to empty array

            # set pixel coordinate in metadata
            self._metadata['pixel'] = self.data['pixel'].data

            # get calibration values -> We assume they are the same over all files!!
            self._metadata['calibrated_values'] = ('pixel',), Qty(spe_info.wavelength, 'nm')

            # initialize empty exposure time and cycle metadata variables
            self._metadata['exposure_time'] = xr.DataArray(
                dims=['file_index'], coords={'file_index': self.metadata['file_index']})
            self._metadata['exposure_time'] = self.metadata['exposure_time'].pint.quantify('s')

            self._metadata['cycles'] = xr.DataArray(
                dims=['file_index'], coords={'file_index': self.metadata['file_index']})
        else:
            if len(self._data['frame'].data) < frame_no:
                # TODO: Add warning that size is changing
                self._data = self.data.reindex(y=np.array(range(frame_no)), fill_value=np.nan)

        # get primary data
        self._data['counts'][file_index] = Qty(counts_info.reshape((frame_no, pixel_no)), 'count')

        # get exposure time. This can vary between different files.
        exposure_text = 'exposuretime._text'
        value = float(norm_footer_dict[find_partial_key(exposure_text)[0]])
        self._metadata['exposure_time'][file_index] = Qty(value, 'ms').to('s')

        # get cycle-counts. This can vary between different files.
        cycles_text = 'cyclecount._text'
        value = int(norm_footer_dict[find_partial_key(cycles_text)[0]])
        self.metadata['cycles'][file_index] = value

    def _set_all_metadata_coords(self):
        """ Sets all the coordinate-related metadata for the spectral data. """
        pass

    def _set_all_metadata_variables(self):
        """ Sets all the metadata variables (dependent on dims) for the spectral data. """

        if np.all(self.metadata['exposure_time'] == self.metadata['exposure_time'][0]):
            self._metadata['exposure_time'] = (), self.metadata['exposure_time'].data[0]

        if np.all(self.metadata['cycles'] == self.metadata['cycles'][0]):
            self._metadata['cycles'] = (), self.metadata['cycles'].data[0]

        self._metadata['wfe_offset'] = self._user_info.wfe_offset
        self._metadata['pixel_offset'] = self._user_info.pixel_offset
        self._metadata['diffraction_order'] = self._user_info.diffraction_order

        if 'calibrated_values' not in self.metadata.variables \
                and 'calibration_data' in self.metadata.attrs:
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

            coord_names = self._find_corresponding_dataset_dims(bgpc)
            self._metadata['background_per_cycle'] = coord_names, bgpc

        self._metadata['background'] = \
            self._metadata['background_per_cycle'] * self.metadata['cycles']
        self._metadata['background_per_time'] = \
            self._metadata['background_per_cycle'] / self.metadata['exposure_time']

        excitation_power_dict = self._get_excitation_power_dict()
        for laser_power_name, laser_power in excitation_power_dict.items():
            data_key = f"background_per_time_per_{laser_power_name}"
            self._metadata[data_key] = self.metadata['background_per_time'] / laser_power

    def _data_post_init(self):
        # Remove cosmic rays if applicable before we create all associated variables
        if self._user_info.heal_cosmic_rays:
            for file_index in self._data['file_index']:
                for frame in self._data['frame']:
                    _, _, self.data['counts'].loc[{'file_index': file_index, 'frame': frame}] = find_outliers(
                        self.data['pixel'],
                        self.data['counts'].sel(file_index=file_index, frame=frame),
                        self._user_info.cosmic_ray_filter.rolling_window_size,
                        self._user_info.cosmic_ray_filter.flagging_significance,
                        self._user_info.cosmic_ray_filter.maximum_outlier_cluster_size,
                        self._user_info.cosmic_ray_filter.hard_min_limit,
                        self._user_info.cosmic_ray_filter.hard_max_limit,
                        self._user_info.cosmic_ray_filter.rolling_data_method,
                        self._user_info.cosmic_ray_filter.approximate_local_stdev_method,
                        True,
                        self._user_info.cosmic_ray_filter.repeat,
                        True
                    )

        super()._data_post_init()

        if len(self._data['frame']) == 1:  # If only one frame was taken, remove frame-related coordinates
            self._data: xr.Dataset = self._data.isel(frame=0, drop=True)

    def _set_all_data_coords(self):
        """ Sets all the coordinate-related data for the spectral data. """

        self._data = self._data.assign_coords({
            'time': self._data['frame'] * self.metadata['exposure_time']
        })

        # pixel-related coordinates
        # get calibrated WFE
        calibrated_values = self._metadata['calibrated_values'].data.copy()
        diff_ord = self._metadata['diffraction_order'].data
        wfe = WFE(calibrated_values, diffraction_order=diff_ord)  # assumes air as medium

        self._data = self._data.assign_coords({
            'energy': (('pixel',), wfe.energy),
            'frequency': (('pixel',), wfe.frequency),
            'wavelength_vacuum': (('pixel',), wfe.wavelength_vacuum),
            'wavelength_air': (('pixel',), wfe.wavelength_medium),
        })

    def _set_all_data_variables(self):
        """ Sets all the data variables (dependent on dims) for the spectral data. """
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

    def _applied_calibration(self) -> Qty:
        """ Apply the calibration data to the pixel list in order to obtain calibrated values
        (in wavelength, frequency or energy). """

        pixels = np.asarray(self.metadata['pixel'].data) + self._metadata['pixel_offset'].data

        calib_data = self._metadata.attrs['calibration_data']

        result = Qty(np.zeros(pixels.shape), calib_data.u)
        for i, cal_coefficient in enumerate(calib_data):
            result += cal_coefficient * pixels ** i

        return result + self._metadata['wfe_offset'].data

    def _find_corresponding_dataset_dims(self, data: EnhancedNumeric | list | np.ndarray) -> tuple[str, ...]:
        """ Returns the corresponding dataset dimensions of an array with unmarked dimensions. """
        data_shape = np.shape(data)
        file_no = len(self.data['file_index'])
        pixel_no = len(self.data['pixel'])

        expected_shapes = {(), (file_no,), (pixel_no,), (file_no, pixel_no), (pixel_no, file_no)}
        assert_options(
            data_shape,
            expected_shapes,
            'array with unmarked dimensions',
            ArrayShapeError,
            data)

        if len(data_shape) == 0:
            return ()
        elif len(data_shape) == 1 and data_shape[0] == file_no:
            return ('file_index',)
        elif len(data_shape) == 1 and data_shape[0] == pixel_no:
            return ('pixel',)
        elif len(data_shape) == 2:
            return 'file_index', 'pixel'
