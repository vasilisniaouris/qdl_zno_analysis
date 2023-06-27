"""
Type aliases for different base-types, e.g. string, pathlib.Path, list of strings/pathlib.Path.
"""

from pathlib import Path
from typing import TypeVar

import xarray as xr

from qdl_zno_analysis import Qty
from ._extra_dependencies import plt, HAS_VISUALIZATION_DEP

AnyString = TypeVar("AnyString", str, Path)
""" Type for string/pathlib.Path. """

MultiAnyString = TypeVar("MultiAnyString", str, Path, list[str], list[Path])
""" Type for string/pathlib.Path or list of strings/pathlib.Path. """

Numeric = TypeVar("Numeric", int, float)
""" Type for int and float. """

EnhancedNumeric = TypeVar("EnhancedNumeric", int, float, Qty)
""" Type for int, float, and pint.Quantity. """


XRObject = TypeVar("XRObject", xr.Dataset, xr.DataArray)


if HAS_VISUALIZATION_DEP:
    PLTArtist = TypeVar("PLTArtist", bound=plt.Artist)
else:
    PLTArtist = None
