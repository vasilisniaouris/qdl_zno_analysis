from pathlib import Path

import pandas as pd

from ..errors import NotFoundError


def df_from_ext_data(local_filename):
    """
    Returns a pandas dataframe from a data file within the `extrernal_data` module.

    Parameters
    ----------
    local_filename : str
        The local filename of the external data file.

    Returns
    -------
    pandas.DataFrame
        The pandas dataframe from the data file.

    Raises
    ------
    NotFoundError
        If the data file is not found.

    Examples
    --------
    >>> df_from_ext_data('refractive_index_air.csv')
    """
    file_path = Path(__file__).parent.joinpath(local_filename)

    if not file_path.exists():
        raise NotFoundError(file_path)

    df = pd.read_csv(file_path)
    return df
