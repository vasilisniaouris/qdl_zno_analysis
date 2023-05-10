from pathlib import Path

from qdl_zno_analysis.errors import NotFoundError


def get_example_filepath(local_filename):
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
    >>> fp = get_example_filepath('spectra/data/002_Smp~QD1_Lsr~Pwr~5u_Tmp~7_Msc~30sec.spe')
    """
    file_path = Path(__file__).parent.joinpath(local_filename)

    if not file_path.exists():
        raise NotFoundError(file_path)

    return file_path
