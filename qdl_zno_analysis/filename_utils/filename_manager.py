"""
The filename_manager module provides utilities for managing filenames and their associated metadata.
It provides methods to easily find the filenames of a given type by using the associated file number.
"""

from typing import Iterable, Iterator

import numpy as np
from pathlib import Path

from qdl_zno_analysis.errors import InvalidFileNumbersError
from qdl_zno_analysis.filename_utils.filename_info import FilenameInfo
from qdl_zno_analysis.typevars import AnyString, MultiAnyString
from qdl_zno_analysis.utils import find_changing_values_in_list_of_dict


def get_filename_info(filename: AnyString) -> FilenameInfo:
    """
    Get `FilenameInfo` object from filename string.

    Parameters
    ----------
    filename : string | Path
        Filename string.

    Returns
    -------
    FilenameInfo
        FilenameInfo object with parsed information from filename string.

    Examples
    --------
    >>> get_filename_info('001_Lsr~Matisse-737p8n-10n-2_Tmp~6p1K_MgF~5p1_Col~PnH~40u_Spt~n2p1u-4p4u.csv')
    FilenameInfo(file_number=1, lasers=SourceInfo(name='Matisse', wfe=<Quantity(737.8, 'nanometer')>, wavelength_vacuum=<Quantity(369.001625, 'nanometer')>, wavelength_air=<Quantity(368.9, 'nanometer')>, frequency=<Quantity(812442.109, 'gigahertz')>, energy=<Quantity(3.35999058, 'electron_volt')>, power=<Quantity(10.0, 'nanowatt')>, order=2, medium='Air'), magnetic_field=<Quantity(5.1, 'tesla')>, temperature=<Quantity(6.1, 'kelvin')>, collection_path_optics=OpticsInfo(pinhole=<Quantity(40.0, 'micrometer')>), spot=<Quantity([-2.1  4.4], 'micrometer')>)
    """
    return FilenameInfo.from_filename(filename)


def get_filename_info_norm_dict(filename: AnyString) -> dict:
    """
    Get normalized dictionary from FilenameInfo object parsed from filename string.

    Parameters
    ----------
    filename : string | Path
        Filename string.

    Returns
    -------
    dict
        Normalized dictionary with parsed information from filename string.

    Examples
    --------
    >>> get_filename_info_norm_dict('001_Lsr~Matisse-737p8n-10n-2_Tmp~6p1K_MgF~5p1_Col~PnH~40u_Spt~n2p1u-4p4u.csv')
     {'file_number': 1, 'sample_name': None, 'lasers.name': 'Matisse', 'lasers.wfe': <Quantity(737.8, 'nanometer')>, 'lasers.wavelength_vacuum': <Quantity(369.001625, 'nanometer')>, 'lasers.wavelength_air': <Quantity(368.9, 'nanometer')>, 'lasers.frequency': <Quantity(812442.109, 'gigahertz')>, 'lasers.energy': <Quantity(3.35999058, 'electron_volt')>, 'lasers.power': <Quantity(10.0, 'nanowatt')>, 'lasers.order': 2, 'lasers.medium': 'Air', 'lasers.miscellaneous': None, 'rf_sources': None, 'magnetic_field': <Quantity(5.1, 'tesla')>, 'temperature': <Quantity(6.1, 'kelvin')>, 'excitation_path_optics': None, 'collection_path_optics.half_waveplate_angle': None, 'collection_path_optics.quarter_waveplate_angle': None, 'collection_path_optics.polarizer': None, 'collection_path_optics.pinhole': <Quantity(40.0, 'micrometer')>, 'collection_path_optics.filters': None, 'collection_path_optics.miscellaneous': None, 'exc_and_col_path_optics': None, 'rf_lines': None, 'measurement_type': None, 'miscellaneous': None, 'spot': <Quantity([-2.1  4.4], 'micrometer')>, 'other': None}
    """
    return FilenameInfo.from_filename(filename).to_normalized_dict()


def get_changing_filename_values_dict(filenames: list[AnyString], reverse_result=False) -> dict:
    """
    Get dictionary with changing values in file names.

    Parameters
    ----------
    filenames : list[string | Path]
        List of filename strings.
    reverse_result : bool, optional
        Returns unchanged values if True. Defaults to false.

    Returns
    -------
    dict
        Dictionary with (non)-repeating filename values.

    Examples
    --------
    >>> fnm_prototype = r'{fno}_Lsr~Matisse-737p8n-{power}n-2_Tmp~6p1K_MgF~5p1_Col~PnH~40u_Spt~n2p1u-4p4u.csv'
    >>> fnms = [fnm_prototype.format(fno=str(ii).zfill(3), power=ii*100) for ii in range(1, 4)]
    >>> get_changing_filename_values_dict(fnms)
    {'file_number': [1, 2, 3], 'lasers.power': <Quantity([100. 200. 300.], 'nanowatt')>}

    >>> fnm_prototype = r'{fno}_Lsr~Matisse-737p8n-10n-2_Tmp~6p1K_MgF~5p1_Col~PnH~40u_Spt~n2p1u-{spt_y}u.csv'
    >>> fnms = [fnm_prototype.format(fno=str(ii).zfill(3), spt_y=ii) for ii in range(1, 4)]
    >>> get_changing_filename_values_dict(fnms)
    {'file_number': [1, 2, 3], 'spot': [<Quantity([-2.1  1. ], 'micrometer')>, <Quantity([-2.1  2. ], 'micrometer')>, <Quantity([-2.1  3. ], 'micrometer')>]}

    """
    info_list = [get_filename_info_norm_dict(filename) for filename in filenames]
    return find_changing_values_in_list_of_dict(info_list, reverse_result)


def get_path_list(string: MultiAnyString) -> list[Path]:
    """
    Get list of Path objects from list or string of file paths.

    Parameters
    ----------
    string : MultiAnyString
        List or string of file paths.

    Returns
    -------
    list[Path]
        List of Path objects parsed from the given string.

    Examples
    --------
    >>> get_path_list('001_Lsr~Matisse-737p8n-10n-2_Tmp~6p1K_MgF~5p1_Col~PnH~40u_Spt~n2p1u-4p4u.csv')
    [WindowsPath('001_Lsr~Matisse-737p8n-10n-2_Tmp~6p1K_MgF~5p1_Col~PnH~40u_Spt~n2p1u-4p4u.csv')]
    """
    if isinstance(string, Iterable) and not isinstance(string, Path | str):
        return [Path(s) for s in string]
    else:
        return [Path(string)]


class FilenameManager:
    """
    Class for managing a list of filenames.

    Examples
    --------
    >>> fnm = FilenameManager([f'{str(i).zfill(3)}_Lsr~Matisse-737p8n-{i*100}n-2_Tmp~6p1K_MgF~5p1_Col~PnH~40u_Spt~n2p1u-4p4u.csv' for i in range(1,3)])

    Using this class, you can access all the filenames provided.

    >>> fnm.filenames
    [WindowsPath('001_Lsr~Matisse-737p8n-100n-2_Tmp~6p1K_MgF~5p1_Col~PnH~40u_Spt~n2p1u-4p4u.csv'), WindowsPath('002_Lsr~Matisse-737p8n-200n-2_Tmp~6p1K_MgF~5p1_Col~PnH~40u_Spt~n2p1u-4p4u.csv')]

    You can also find which ones are existing paths (check_validity -> True). This will change all the class property values. For the example's sake, we do not check the validity, so the valid path is populated with all filenames.

    >>> fnm.valid_paths
    [WindowsPath('001_Lsr~Matisse-737p8n-100n-2_Tmp~6p1K_MgF~5p1_Col~PnH~40u_Spt~n2p1u-4p4u.csv'), WindowsPath('002_Lsr~Matisse-737p8n-200n-2_Tmp~6p1K_MgF~5p1_Col~PnH~40u_Spt~n2p1u-4p4u.csv')]

    You can acces the `FilenameInfo` objects for each filename.

    >>> fnm.filename_info_list
    [FilenameInfo(file_number=1, lasers=SourceInfo(name='Matisse', wfe=<Quantity(737.8, 'nanometer')>, wavelength_vacuum=<Quantity(369.001625, 'nanometer')>, wavelength_air=<Quantity(368.9, 'nanometer')>, frequency=<Quantity(812442.109, 'gigahertz')>, energy=<Quantity(3.35999058, 'electron_volt')>, power=<Quantity(100.0, 'nanowatt')>, order=2, medium='Air'), magnetic_field=<Quantity(5.1, 'tesla')>, temperature=<Quantity(6.1, 'kelvin')>, collection_path_optics=OpticsInfo(pinhole=<Quantity(40.0, 'micrometer')>), spot=<Quantity([-2.1  4.4], 'micrometer')>), FilenameInfo(file_number=2, lasers=SourceInfo(name='Matisse', wfe=<Quantity(737.8, 'nanometer')>, wavelength_vacuum=<Quantity(369.001625, 'nanometer')>, wavelength_air=<Quantity(368.9, 'nanometer')>, frequency=<Quantity(812442.109, 'gigahertz')>, energy=<Quantity(3.35999058, 'electron_volt')>, power=<Quantity(200.0, 'nanowatt')>, order=2, medium='Air'), magnetic_field=<Quantity(5.1, 'tesla')>, temperature=<Quantity(6.1, 'kelvin')>, collection_path_optics=OpticsInfo(pinhole=<Quantity(40.0, 'micrometer')>), spot=<Quantity([-2.1  4.4], 'micrometer')>)]

    And print them out as dictionaries.

    >>> fnm.filename_info_dicts
    [{'file_number': 1, 'sample_name': None, 'lasers': {'name': 'Matisse', 'wfe': <Quantity(737.8, 'nanometer')>, 'wavelength_vacuum': <Quantity(369.001625, 'nanometer')>, 'wavelength_air': <Quantity(368.9, 'nanometer')>, 'frequency': <Quantity(812442.109, 'gigahertz')>, 'energy': <Quantity(3.35999058, 'electron_volt')>, 'power': <Quantity(100.0, 'nanowatt')>, 'order': 2, 'medium': 'Air', 'miscellaneous': None}, 'rf_sources': None, 'magnetic_field': <Quantity(5.1, 'tesla')>, 'temperature': <Quantity(6.1, 'kelvin')>, 'excitation_path_optics': None, 'collection_path_optics': {'half_waveplate_angle': None, 'quarter_waveplate_angle': None, 'polarizer': None, 'pinhole': <Quantity(40.0, 'micrometer')>, 'filters': None, 'miscellaneous': None}, 'exc_and_col_path_optics': None, 'rf_lines': None, 'measurement_type': None, 'miscellaneous': None, 'spot': <Quantity([-2.1  4.4], 'micrometer')>, 'other': None}, {'file_number': 2, 'sample_name': None, 'lasers': {'name': 'Matisse', 'wfe': <Quantity(737.8, 'nanometer')>, 'wavelength_vacuum': <Quantity(369.001625, 'nanometer')>, 'wavelength_air': <Quantity(368.9, 'nanometer')>, 'frequency': <Quantity(812442.109, 'gigahertz')>, 'energy': <Quantity(3.35999058, 'electron_volt')>, 'power': <Quantity(200.0, 'nanowatt')>, 'order': 2, 'medium': 'Air', 'miscellaneous': None}, 'rf_sources': None, 'magnetic_field': <Quantity(5.1, 'tesla')>, 'temperature': <Quantity(6.1, 'kelvin')>, 'excitation_path_optics': None, 'collection_path_optics': {'half_waveplate_angle': None, 'quarter_waveplate_angle': None, 'polarizer': None, 'pinhole': <Quantity(40.0, 'micrometer')>, 'filters': None, 'miscellaneous': None}, 'exc_and_col_path_optics': None, 'rf_lines': None, 'measurement_type': None, 'miscellaneous': None, 'spot': <Quantity([-2.1  4.4], 'micrometer')>, 'other': None}]

    Or as normalized dictionaries.

    >>> fnm.filename_info_norm_dicts
    [{'file_number': 1, 'sample_name': None, 'lasers.name': 'Matisse', 'lasers.wfe': <Quantity(737.8, 'nanometer')>, 'lasers.wavelength_vacuum': <Quantity(369.001625, 'nanometer')>, 'lasers.wavelength_air': <Quantity(368.9, 'nanometer')>, 'lasers.frequency': <Quantity(812442.109, 'gigahertz')>, 'lasers.energy': <Quantity(3.35999058, 'electron_volt')>, 'lasers.power': <Quantity(100.0, 'nanowatt')>, 'lasers.order': 2, 'lasers.medium': 'Air', 'lasers.miscellaneous': None, 'rf_sources': None, 'magnetic_field': <Quantity(5.1, 'tesla')>, 'temperature': <Quantity(6.1, 'kelvin')>, 'excitation_path_optics': None, 'collection_path_optics.half_waveplate_angle': None, 'collection_path_optics.quarter_waveplate_angle': None, 'collection_path_optics.polarizer': None, 'collection_path_optics.pinhole': <Quantity(40.0, 'micrometer')>, 'collection_path_optics.filters': None, 'collection_path_optics.miscellaneous': None, 'exc_and_col_path_optics': None, 'rf_lines': None, 'measurement_type': None, 'miscellaneous': None, 'spot': <Quantity([-2.1  4.4], 'micrometer')>, 'other': None}, {'file_number': 2, 'sample_name': None, 'lasers.name': 'Matisse', 'lasers.wfe': <Quantity(737.8, 'nanometer')>, 'lasers.wavelength_vacuum': <Quantity(369.001625, 'nanometer')>, 'lasers.wavelength_air': <Quantity(368.9, 'nanometer')>, 'lasers.frequency': <Quantity(812442.109, 'gigahertz')>, 'lasers.energy': <Quantity(3.35999058, 'electron_volt')>, 'lasers.power': <Quantity(200.0, 'nanowatt')>, 'lasers.order': 2, 'lasers.medium': 'Air', 'lasers.miscellaneous': None, 'rf_sources': None, 'magnetic_field': <Quantity(5.1, 'tesla')>, 'temperature': <Quantity(6.1, 'kelvin')>, 'excitation_path_optics': None, 'collection_path_optics.half_waveplate_angle': None, 'collection_path_optics.quarter_waveplate_angle': None, 'collection_path_optics.polarizer': None, 'collection_path_optics.pinhole': <Quantity(40.0, 'micrometer')>, 'collection_path_optics.filters': None, 'collection_path_optics.miscellaneous': None, 'exc_and_col_path_optics': None, 'rf_lines': None, 'measurement_type': None, 'miscellaneous': None, 'spot': <Quantity([-2.1  4.4], 'micrometer')>, 'other': None}]

    You can find the values that chage between FilenameInfo objects and access them as a dictionary, of lists.

    >>> fnm.changing_filename_info_dict
    {'file_number': [1, 2], 'lasers.power': <Quantity([100. 200.], 'nanowatt')>}

    You can choose to not omit the non-repeating values in the dictionary. These appear a single value.

    >>> fnm.non_changing_filename_info_dict
    {'sample_name': None, 'lasers.name': 'Matisse', 'lasers.wfe': <Quantity(737.8, 'nanometer')>, 'lasers.wavelength_vacuum': <Quantity(369.001625, 'nanometer')>, 'lasers.wavelength_air': <Quantity(368.9, 'nanometer')>, 'lasers.frequency': <Quantity(812442.109, 'gigahertz')>, 'lasers.energy': <Quantity(3.35999058, 'electron_volt')>, 'lasers.order': 2, 'lasers.medium': 'Air', 'lasers.miscellaneous': None, 'rf_sources': None, 'magnetic_field': <Quantity(5.1, 'tesla')>, 'temperature': <Quantity(6.1, 'kelvin')>, 'excitation_path_optics': None, 'collection_path_optics.half_waveplate_angle': None, 'collection_path_optics.quarter_waveplate_angle': None, 'collection_path_optics.polarizer': None, 'collection_path_optics.pinhole': <Quantity(40.0, 'micrometer')>, 'collection_path_optics.filters': None, 'collection_path_optics.miscellaneous': None, 'exc_and_col_path_optics': None, 'rf_lines': None, 'measurement_type': None, 'miscellaneous': None, 'spot': <Quantity([-2.1  4.4], 'micrometer')>, 'other': None}

    Additionally, you can retrieve all available filetypes.

    >>> fnm.available_filetypes
    ['csv']

    And print a dictionary of filetypes and filenames.

    >>> fnm.filenames_by_filetype
    {'csv': [WindowsPath('001_Lsr~Matisse-737p8n-100n-2_Tmp~6p1K_MgF~5p1_Col~PnH~40u_Spt~n2p1u-4p4u.csv'), WindowsPath('002_Lsr~Matisse-737p8n-200n-2_Tmp~6p1K_MgF~5p1_Col~PnH~40u_Spt~n2p1u-4p4u.csv')]}

    And finally, you can find all the available file numbers.

    >>> fnm.available_file_numbers
    [1, 2]
    """

    def __init__(self, filenames: MultiAnyString, folder: AnyString = Path('.'), check_validity=False):
        """
        Parameters
        ----------
        filenames : str | Path | list[str | Path]
            List or string of file names.
        folder : str | Path, optional
            Folder path, by default Path('.').
        check_validity : bool, optional
            Check the validity of the folder and file names, by default False.
        """
        self.filenames: list[Path] = get_path_list(filenames)
        self.folder: Path = Path(folder)
        self.check_validity = check_validity

        if self.check_validity:
            self._check_folder_exists()

        self.__post_init__()

    def _check_folder_exists(self):
        """
        Check if the folder exists.

        Raises
        ------
        FileExistsError
            If the folder does not exist.
        """
        if not self.folder.is_dir():
            raise FileExistsError(f"The folder `{self.folder}` does not exist.")

    def __post_init__(self):
        self.valid_paths = self._get_valid_paths()
        self.filename_info_list = self._get_filename_info_list()

    def _get_valid_paths(self) -> list[Path]:
        """ Get list of valid file paths. """

        full_filenames = [self.folder.joinpath(filename) for filename in self.filenames]

        return full_filenames if not self.check_validity else [full_filename for full_filename in full_filenames
                                                               if full_filename.exists()]

    def _get_filename_info_list(self) -> list[FilenameInfo]:
        """ Get list of `FilenameInfo` objects. """
        return [get_filename_info(valid_path) for valid_path in self.valid_paths]

    @property
    def filename_info_dicts(self) -> list[dict]:
        """ Get list of dictionaries with parsed information from filenames. """
        return [filename_info.to_dict() for filename_info in self.filename_info_list]

    @property
    def filename_info_norm_dicts(self) -> list[dict]:
        """ Get list of dictionaries with normalized parsed information from filenames. """
        return [filename_info.to_normalized_dict() for filename_info in self.filename_info_list]

    @property
    def changing_filename_info_dict(self) -> dict:
        """ Get dictionary with changing values in file names. """
        return find_changing_values_in_list_of_dict(self.filename_info_norm_dicts)

    @property
    def non_changing_filename_info_dict(self) -> dict:
        """ Get dictionary with non-changing values in file names. """
        return find_changing_values_in_list_of_dict(self.filename_info_norm_dicts, True)

    @property
    def available_filetypes(self) -> list[str]:
        """ Get list of available file types. """
        return list(np.unique([filename.suffix.replace('.', '') for filename in self.filenames]))

    @property
    def filenames_by_filetype(self) -> dict[str, list[Path]]:
        """ Get dictionary with filenames grouped by file type. """
        return {filetype: [filename for filename in self.filenames if filename.suffix.replace('.', '') == filetype]
                for filetype in self.available_filetypes}

    @property
    def available_file_numbers(self) -> list[int]:
        """ Get list of available file numbers. """
        return_list = list(np.unique([filename_info.fno for filename_info in self.filename_info_list]))
        return_list.sort()
        return return_list

    @classmethod
    def from_matching_string(cls, matching_string: str, folder=Path('.')):
        """
        Generates the list of filenames based on the matching string that exists in the given folder.
        Uses `Path.glob()`

        Parameters
        ----------
        matching_string : str
            A string that will be used to match the filenames.
        folder : str | Path
            The folder where the files are stored. Defaults to Path('.')

        Examples
        --------
        >>> # holds all valid paths that have the word Lifetime in them and end in .txt
        >>> fnm = FilenameManager.from_matching_string('Lifetime*.txt')
        """

        folder = Path(folder)
        filenames = list(folder.glob(matching_string))

        return cls(filenames, folder, True)

    @classmethod
    def from_file_numbers(cls, file_numbers: range | tuple | Iterable | int | float,
                          filetypes: str | list = None, folder=Path('.')):
        """
        Generates the list of filenames based on the file numbers and file types that exist in the given folder.

        Parameters
        ----------
        file_numbers : range | tuple | Iterable | int | float
            A sequence of file numbers to load.
            It can be specified as a range, iterable, tuple or single integer/float value. If a tuple is provided,
            it will be used to generate a range.
            Remember that range starts from 0 and does not include the last digit.
        filetypes : str | list[str], optional
            The file types to load. It can be specified as a single string or a list of strings. Defaults to None.
        folder : str | Path
            The folder where the files are stored. Defaults to Path('.')

        Examples
        --------
        >>> # holds all valid paths with file number 1 and filetype csv
        >>> fnm = FilenameManager.from_file_numbers(1, 'csv')
        >>> # holds all valid paths with file numer in range(1,3)==[1, 2] and filetype txt
        >>> fnm = FilenameManager.from_file_numbers((1,3), 'txt')
        >>> # holds all valid paths with file numer in range(1,7,2)==[1, 3, 5] and filetypes txt or csv
        >>> fnm = FilenameManager.from_file_numbers(range(1,3), ['txt', 'csv'])
        >>> # holds all valid paths with file numer in [11, 14, 25] and filetypes txt, csv or a folder
        >>> fnm = FilenameManager.from_file_numbers([11, 14, 25], ['txt', 'csv', ''])
        """
        if isinstance(file_numbers, range):
            file_numbers = list(file_numbers)
        elif isinstance(file_numbers, tuple):
            file_numbers = list(range(*file_numbers))
        elif isinstance(file_numbers, Iterable):
            file_numbers = [int(number) for number in file_numbers]
        elif isinstance(file_numbers, int | float):
            file_numbers = [int(file_numbers)]
        else:
            raise InvalidFileNumbersError(file_numbers)

        if isinstance(filetypes, str):
            filetypes = [filetypes]

        folder = Path(folder)
        dir_filenames = list(folder.iterdir())
        dir_file_numbers = [get_filename_info(filename).fno for filename in dir_filenames]
        filenames = []

        for i, dir_fno in enumerate(dir_file_numbers):
            if dir_fno in file_numbers:
                if filetypes is not None:
                    if len(filetypes):
                        if dir_filenames[i].suffix.replace('.', '') in filetypes:
                            filenames.append(dir_filenames[i].name)
                else:
                    filenames.append(dir_filenames[i].name)

        return cls(filenames, folder, True)

    def __getitem__(self, item):
        """
        Get an item from the list of file names.

        Parameters
        ----------
        item : int
            Index of item to get.

        Returns
        -------
        Any
            The item at the specified index.
        """
        return self.filenames.__getitem__(item)

    def __iter__(self) -> Iterator:
        """
        Get an iterator over the list of file names.

        Returns
        -------
        Iterator
            Iterator over the list of file names.
        """
        return self.filenames.__iter__()
