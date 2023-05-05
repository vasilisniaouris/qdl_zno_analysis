"""
This module contains the base exception classes for the package.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Dict


@dataclass(frozen=False)
class QZError(Exception):
    """ Base exception for all errors. """

    def __reduce__(self):
        return self.__class__, tuple(getattr(self, f.name) for f in fields(self))


@dataclass(frozen=False)
class MethodInputError(ValueError, QZError):
    """ Raised when input argument in method is not in the allowed list of values. """

    arg: str
    value: any
    allow_list: list
    mthd: str

    def __str__(self):
        msg = f"The value  {self.value} for the input argument {self.arg} in {self.mthd} method is invalid. " \
              f"Choose any of the following instead: {self.allow_list}"
        return msg


@dataclass(frozen=False)
class NotFoundError(FileNotFoundError, QZError):
    """ Raised when file is not found. """

    file_path: Path

    def __str__(self):
        return f"File '{self.file_path.name}' (local path: {self.file_path}) does not exist."


@dataclass(frozen=False)
class InvalidFileError(ValueError, QZError):
    """ Raised when file is not in the expected format. """

    file_path: Path

    def __str__(self):
        return f"File '{self.file_path.name}' (local path: {self.file_path}) is not in the expected format."


@dataclass(frozen=False)
class InvalidUnitStringError(ValueError, QZError):
    """ Raised when the provided unit string is invalid or cannot be converted to a pint.Unit object. """

    unit_str: str

    def __str__(self):
        return f"Invalid unit string '{self.unit_str}', not found in pint.UnitRegistry."


@dataclass(frozen=False)
class InvalidDerivedUnitStringError(ValueError, QZError):
    """
    Raised when the derived unit string obtained by adding the default core unit for the
    primary physical type to the prefix is invalid.
    """

    unit_str: str
    primary_physical_type: str
    default_units: Dict[str, Dict[str, str]]

    def __str__(self):
        core_unit = self.default_units[self.primary_physical_type]['core']
        combined_unit_str = self.unit_str + core_unit
        return f"Invalid derived unit string '{combined_unit_str}' derived by the primary physical type " \
               f"'{self.primary_physical_type}' with core unit '{core_unit}', and user-defined prefix " \
               f"{self.unit_str}, not found in the pint.UnitRegistry."


@dataclass(frozen=False)
class UnsupportedUnitError(ValueError, QZError):
    """ Raised when the provided unit string is not supported for the given primary physical type and context. """

    unit_str: str
    primary_physical_type: str
    context: str = None

    def __str__(self):
        message = f"Unsupported unit string: {self.unit_str} for primary physical type {self.primary_physical_type}"
        if self.context:
            message += f" in context {self.context}"
        return message + '.'


@dataclass(frozen=False)
class IncompatibleUnitWarning(UserWarning):
    """ Raised when the input unit string is not compatible with the main unit of the primary physical type. """

    unit_str: str
    primary_physical_type: str
    context: str
    default_units: Dict[str, Dict[str, str]]

    def __str__(self):
        context_message = ''
        if self.context is not None:
            context_message = f" In context '{self.context}',"
        message = f"Incompatible unit string: {context_message} unit '{self.unit_str}' " \
                  f"is not compatible with primary physical type " \
                  f"{self.primary_physical_type} that has a core unit of " \
                  f"'{self.default_units[self.primary_physical_type]['core']}'. " \
                  f"Consider using a unit that is compatible with the primary physical type."

        return message


@dataclass(frozen=False)
class InvalidFileNumbersError(TypeError, QZError):
    """ Raised when the file numbers argument is not in the expected format. """

    file_numbers: any

    def __str__(self):
        return f"The file_numbers argument must be a range, tuple, iterable, or a single integer/float value. " \
               f"Got {type(self.file_numbers)} instead."


@dataclass(frozen=False)
class InfoSubclassArgumentNumberError(ValueError, QZError):
    """ Raised when the subclass argument number is larger than the Info subclass available attributes. """

    parsed_object: List
    attrs: List[str]

    def __str__(self):
        return f"More arguments in parsed object: {self.parsed_object} than attributes in Info subclass {self.attrs}."
