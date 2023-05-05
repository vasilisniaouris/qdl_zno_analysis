"""
Type aliases for different base-types, e.g. string, pathlib.Path, list of strings/pathlib.Path.
"""

from pathlib import Path
from typing import TypeVar, List

AnyString = TypeVar("AnyString", str, Path)
""" Type for string/pathlib.Path. """

MultiAnyString = TypeVar("MultiAnyString", str, Path, List[str], List[Path])
""" Type for string/pathlib.Path or list of strings/pathlib.Path. """
