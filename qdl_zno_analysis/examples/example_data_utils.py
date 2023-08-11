import warnings
from dataclasses import dataclass
from typing import ClassVar

from pathlib import Path

from qdl_zno_analysis._extra_dependencies import HAS_EXAMPLE_DATA_DEP, requests
from qdl_zno_analysis.errors import NotFoundError

if HAS_EXAMPLE_DATA_DEP:
    from gdown.download import _get_session, download
    from gdown.download_folder import _parse_google_drive_file, download_folder
else:
    pass

GOOGLE_DRIVE_URL = "https://drive.google.com/drive/folders/1yPZ5BKT5_7zENuy6-cKib_W9ze5gPXNA?usp=share_link"
""" The URL of the Google Drive folder with example data. """


@dataclass
class GoogleDriveContent:
    """
    Represents content hosted on Google Drive. This class is heavily inspired by
    the `gdown.download_folder._GoogleDriveFile` class and other methods from `gdown` package.

    Parameters

    Attributes
    ----------
    gd_id : str
        The id of the Google Drive file or folder.
    name : str
        The name of the Google Drive file or folder.
    gd_type : str
        The type of the Google Drive file or folder.
    children : list["GoogleDriveContent"] | None
        The list of children if it's a folder; otherwise, None.
    """
    gd_id: str
    name: str
    gd_type: str
    children: list["GoogleDriveContent"] | None

    TYPE_GD_AUDIO: ClassVar[str] = "application/vnd.google-apps.audio"
    """ MIME type for Google Drive audio file. """
    TYPE_GD_DOCUMENT: ClassVar[str] = "application/vnd.google-apps.document"
    """ MIME type for Google Drive document file. """
    TYPE_GD_DRIVE_SDK: ClassVar[str] = "application/vnd.google-apps.drive-sdk"
    """ MIME type for Google Drive drive-sdk file. """
    TYPE_GD_DRAWING: ClassVar[str] = "application/vnd.google-apps.drawing"
    """ MIME type for Google Drive drawing file. """
    TYPE_GD_FILE: ClassVar[str] = "application/vnd.google-apps.file"
    """ MIME type for Google Drive file. """
    TYPE_GD_FOLDER: ClassVar[str] = "application/vnd.google-apps.folder"
    """ MIME type for Google Drive folder. """
    TYPE_GD_FORM: ClassVar[str] = "application/vnd.google-apps.form"
    """ MIME type for Google Drive form file. """
    TYPE_GD_FUSIONTABLE: ClassVar[str] = "application/vnd.google-apps.fusiontable"
    """ MIME type for Google Drive fusiontable file. """
    TYPE_GD_JAM: ClassVar[str] = "application/vnd.google-apps.jam"
    """ MIME type for Google Drive jam file. """
    TYPE_GD_MAP: ClassVar[str] = "application/vnd.google-apps.map"
    """ MIME type for Google Drive map file. """
    TYPE_GD_PHOTO: ClassVar[str] = "application/vnd.google-apps.photo"
    """ MIME type for Google Drive photo file. """
    TYPE_GD_PRESENTATION: ClassVar[str] = "application/vnd.google-apps.presentation"
    """ MIME type for Google Drive presentation file. """
    TYPE_GD_SCRIPT: ClassVar[str] = "application/vnd.google-apps.script"
    """ MIME type for Google Drive script file. """
    TYPE_GD_SHORTCUT: ClassVar[str] = "application/vnd.google-apps.shortcut"
    """ MIME type for Google Drive shortcut file. """
    TYPE_GD_SITE: ClassVar[str] = "application/vnd.google-apps.site"
    """ MIME type for Google Drive site file. """
    TYPE_GD_SPREADSHEET: ClassVar[str] = "application/vnd.google-apps.spreadsheet"
    """ MIME type for Google Drive spreadsheet file. """
    TYPE_GD_UNKNOWN: ClassVar[str] = "application/vnd.google-apps.unknown"
    """ MIME type for Google Drive unknown file. """
    TYPE_GD_VIDEO: ClassVar[str] = "application/vnd.google-apps.video"
    """ MIME type for Google Drive video file. """

    def is_folder(self):
        """ Returns True if this Google Drive content is a folder. False otherwise. """
        return self.gd_type == GoogleDriveContent.TYPE_GD_FOLDER

    @classmethod
    def from_url(cls, url, proxy=None, use_cookies=True, verify=True):
        """
        Creates a GoogleDriveContent object from the given URL.

        Parameters
        ----------
        url : str
            The URL of the Google Drive file or folder.
        proxy : str | None
            The proxy to use for the request.
        use_cookies : bool
            Whether to use cookies.
        verify : bool
            Whether to verify the SSL certificate.

        Returns
        -------
        GoogleDriveContent | None
            The GoogleDriveContent object if the URL is valid; otherwise, None.
        """
        # canonicalize the language into English
        url += "&hl=en" if "?" in url else "?hl=en"

        sess = _get_session(proxy=proxy, use_cookies=use_cookies)
        try:
            result = sess.get(url, verify=verify)
        except requests.exceptions.ProxyError as e:
            warnings.warn('Some message')  # TODO: change message
            return None
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f'Received session error: {e}.\n'
                          'Google Drive may have blocked the folder due to intense activity. '
                          f'Check out the link directly: {GOOGLE_DRIVE_URL}')
            return None

        if result.status_code != 200:
            return None

        gdown_gdrive_file, id_name_type_iter = _parse_google_drive_file(url, result.text)

        children = []
        for child_id, child_name, child_type in id_name_type_iter:
            if child_type == cls.TYPE_GD_FOLDER:
                child = cls.from_id(child_id, proxy, use_cookies, verify)
            else:
                child = cls(child_id, child_name, child_type, None)
            children.append(child)

        obj = cls(gdown_gdrive_file.id, gdown_gdrive_file.name, gdown_gdrive_file.type, children)

        return obj

    @classmethod
    def from_id(cls, gd_id: str, proxy=None, use_cookies=True, verify=True):
        """
        Creates a GoogleDriveContent object from the given Google Drive ID.

        Parameters
        ----------
        gd_id : str
            The Google Drive ID.
        proxy : str | None
            The proxy to use for the request.
        use_cookies : bool
            Whether to use cookies.
        verify : bool
            Whether to verify the SSL certificate.

        Returns
        -------
        GoogleDriveContent | None
            The GoogleDriveContent object if the ID is valid; otherwise, None.
        """
        url = f"https://drive.google.com/drive/folders/{gd_id}"
        return cls.from_url(url, proxy, use_cookies, verify)

    @property
    def children_name_dict(self):
        """ Returns a dictionary of children by their name. """
        return {child.name: child for child in self.children} if self.children is not None else None

    @property
    def children_gd_id_dict(self):
        """ Returns a dictionary of children by their Google Drive ID. """
        return {child.gd_id: child for child in self.children} if self.children is not None else None

    @property
    def family_tree_path_dict(self) -> dict[Path, "GoogleDriveContent"] | None:
        """
        Returns a dictionary of children by their family tree path.
        The family tree path is the path from the root of the tree to the child.

        Returns
        -------
        dict[Path, "GoogleDriveContent"] | None
            The dictionary of children by their family tree path.
        """

        if self.children is None:
            return None

        path_dict = {}
        for child in self.children:
            child_path = Path(child.name)
            path_dict[child_path] = child
            if child.family_tree_path_dict is not None:
                for grandchild_path, grandchild in child.family_tree_path_dict.items():
                    path_dict[child_path.joinpath(grandchild_path)] = grandchild

        return path_dict

    def get_child(self, child_name):
        """ Returns the child with the given name. """
        if self.children is not None:
            return self.children_name_dict.get(child_name, None)
        else:
            return None

    def get_content_from_path(self, path: str):
        """ Returns the content at the given Google Drive path. """
        path = Path(path)
        return self.family_tree_path_dict.get(path, None)

    def download(
            self, content_path=None, output=None, quiet=False, proxy=None, speed=None, use_cookies=True, verify=True,
            fuzzy=False, resume=False, format=None, remaining_ok=False) -> list[str] | str:
        """
        Downloads the content at the given Google Drive path.

        Parameters
        ----------
        content_path : str | Path | None
            The relative path within the Google Drive content. If None, the content is downloaded from the root of
            the Google Drive content.
        output : str | Path | None
            Output filename. Default is basename of URL.
        quiet : bool
            Whether to print the progress.
        proxy : str | None
            The proxy to use for the request.
        speed : int | None
            TDownload byte size per second (e.g., 256KB/s = 256 * 1024).
        use_cookies : bool
            Whether to use cookies.
        verify : bool
             Either a bool, in which case it controls whether the server's TLS certificate is verified,
              or a string, in which case it must be a path to a CA bundle to use. Default is True.
        fuzzy : bool
            Fuzzy extraction of Google Drive's file Id. Default is False.
        resume : bool
            Resume the download from existing tmp file if possible. Defaults to False.
        format : str | None
            Format of Google Docs, Spreadsheets and Slides. Default is:
            - Google Docs: 'docx'
            - Google Spreadsheet: 'xlsx'
            - Google Slides: 'pptx'
        remaining_ok : bool
            Whether to continue the download even if the download is not finished.

        Returns
        -------
        list of str
            List of files downloaded, or None if failed.
        """

        if content_path is None:
            content = self
        else:
            content = self.get_content_from_path(content_path)

        if output is not None:
            output = str(output)

        if content.is_folder():
            return download_folder(id=content.gd_id, output=output, quiet=quiet, proxy=proxy, speed=speed,
                                   use_cookies=use_cookies, remaining_ok=remaining_ok, verify=verify)
        else:
            return download(id=content.gd_id, output=output, quiet=quiet, proxy=proxy, speed=speed,
                            use_cookies=use_cookies, verify=verify, fuzzy=fuzzy, resume=resume, format=format)


def get_example_local_filepath(local_filename):
    """
    Returns the filepath of a data file within the `examples` module.

    Parameters
    ----------
    local_filename : str
        The local filename of the file.

    Returns
    -------
    Path
        The filepath of a data file.

    Raises
    ------
    NotFoundError
        If the data file is not found.

    Examples
    --------
    >>> local_example_folder = get_example_local_filepath('spectra')
    >>> local_example_file = get_example_local_filepath('spectra/002_Smp~QD1_Lsr~Pwr~5u_Tmp~7_Msc~30sec.spe')
    """
    file_path = Path(__file__).parent.joinpath(local_filename)

    if not file_path.exists():
        raise NotFoundError(file_path)

    return file_path


def get_example_data_content(relative_path):
    """
    Returns the `GoogleDriveContent` of a folder (`relative_path`) within the Google Drive example data folder.

    Parameters
    ----------
    relative_path : str
        The relative path of the file/folder.

    Returns
    -------
    GoogleDriveContent
        The content object.

    Examples
    --------
    >>> get_example_data_content('spectra')
    GoogleDriveContent(gd_id='1hum4wxtLe7aYgrXkJgiyQHV575ZaazP-?hl=en', name='spectra', gd_type='application/vnd.google-apps.folder', children=[GoogleDriveContent(gd_id='1zF-FBRpgE_iOCQsirdmu_ybHJjflZDNx', name='001_Smp~FibbedTDEpi_Lsr~CNI-360n-760n_Tmp~5p3_MgF~0_Col~Flt~PLx1-PnH~In_Spt~n0p91u-n2p02u.sif', gd_type='application/octet-stream', children=None), GoogleDriveContent(gd_id='1qbcXIhb3Ht6XQRqYph9TsEmWvtyc2Tpt', name='002_Smp~QD1_Lsr~Pwr~5u_Tmp~7_Msc~30sec.spe', gd_type='application/octet-stream', children=None)])
    """
    url = GOOGLE_DRIVE_URL
    gd_content = GoogleDriveContent.from_url(url)
    return gd_content.get_content_from_path(relative_path)


def download_example_data(path_in: str, path_out=None):
    """
    Downloads the example data from the Google Drive example data folder.

    Parameters
    ----------
    path_in : str
        The relative path of the file/folder.
    path_out : str, optional
        The local path to save the data. Defaults to local package folder.

    Returns
    -------
    list[str] | str
        The path(s) to the downloaded data.
    """
    if path_out is None:
        path_out = Path(__file__).parent.joinpath(path_in)

    content = GoogleDriveContent.from_url(GOOGLE_DRIVE_URL)
    return content.download(path_in, path_out, quiet=False)


def assure_example_data_exist(example_path: str):
    """
    Ensures that the example data exist.

    Parameters
    ----------
    example_path : str
        The relative path of the file/folder within the Google Drive example data folder.

    Returns
    -------
    None | list[str] | str
        None if the example data exist. If the example data do not exist, returns a list of strings or a string
         of the downloaded data paths.
    """
    example_path_content = get_example_data_content(example_path)
    local_example_folder_path = Path(__file__).parent.joinpath(example_path)
    expected_filename_paths = [local_example_folder_path.joinpath(key)
                               for key in example_path_content.family_tree_path_dict]

    if not all([expected_fp.exists() for expected_fp in expected_filename_paths]):
        proceed = input(f"The example data path '{example_path}' does not exist. "
                        f"Would you like to download it? (y/n): ")
        if proceed.lower() in ['y', 'yes']:
            return example_path_content.download(output=local_example_folder_path)
        else:
            warnings.warn("The example data were not downloaded.")

    return None
