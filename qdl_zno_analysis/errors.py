"""
This module contains the base exception classes for the package.
"""

from dataclasses import dataclass, fields, field
from pathlib import Path

from typing import TypeVar, Type, Sequence, Iterable

import numpy as np

from qdl_zno_analysis import ureg, Qty

T_ = TypeVar('T_')


@dataclass(frozen=False)
class QDLAnalysisError(Exception):
    """ Base exception for all package-related errors. """

    message: str = field(init=False)

    def __reduce__(self):
        return self.__class__, tuple(getattr(self, f.name) for f in fields(self))

    def __str__(self):
        return self.message

    def __post_init__(self):
        self.message = 'Base exception, unidentified error.'


@dataclass(frozen=False)
class ValueOutOfBoundsError(ValueError, QDLAnalysisError):
    """ Raised when the test value(s) is outside given bounds. """
    value: Sequence[T_] | Iterable[T_] | T_
    limits: (T_, T_)
    value_name: str

    def __post_init__(self):
        self.message = f"Invalid value {self.value_name} = {self.value}.\n" \
                       f"Must be between {self.limits[0]} and {self.limits[1]}."


@dataclass(frozen=False)
class OutOfOptionsError(QDLAnalysisError):
    """ Raised when the test value is not in a given set of options. """
    value: Sequence[T_] | Iterable[T_] | T_
    options: set[T_] | list[T_] | tuple[T_]
    value_name: str

    def __post_init__(self):
        self.options = set(self.options)
        self.message = f"Invalid value {self.value_name} = {self.value}.\n" \
                       f"Must choose between the following options: {self.options}."


@dataclass(frozen=False)
class ValueOutOfOptionsError(ValueError, OutOfOptionsError):
    """ Raised when the test value is not in a given set of options. """


@dataclass(frozen=False)
class ArrayShapeError(ValueError, OutOfOptionsError):
    """ Raised when the array of interest has an unrecognized shape. """
    array: T_

    def __post_init__(self):
        self.message = f"Invalid array shape {self.value} for {self.value_name} = {self.array}.\n" \
                       f"Array shape may only be any of the following: {self.options}."


@dataclass(frozen=False)
class ArrayDimensionNumberError(ValueError, OutOfOptionsError):
    array: T_

    def __post_init__(self):
        self.message = f"Invalid number of array dimensions {self.value} for {self.value_name} = {self.array}.\n" \
                       f"Array dimensions may only be any of the following {self.options}."


@dataclass(frozen=False)
class IsNullError(QDLAnalysisError):
    """ Raised when the test value is None or empty, depending on the context. """
    value: T_
    value_name: str

    def __post_init__(self):
        self.message = f"Invalid value {self.value_name} = {self.value}.\n" \
                       f"Value can not be null/empty."

# @dataclass(frozen=False)
# class InvalidFileError(ValueError, QZError):
#     """ Raised when file is not in the expected format. """
#
#     file_path: Path
#
#     def __str__(self):
#         return f"File '{self.file_path.name}' (local path: {self.file_path}) is not in the expected format."


@dataclass(frozen=False)
class UnitError(QDLAnalysisError):
    """ Raised when there is a unit-related error. """

    unit_str: str

    def __post_init__(self):
        self.message = f"Base pint.Unit exception, unidentified error."


@dataclass(frozen=False)
class InvalidUnitStringError(ValueError, UnitError):
    """ Raised when the provided unit string is invalid or cannot be converted to a pint.Unit object. """

    def __post_init__(self):
        self.message = f"Invalid unit string '{self.unit_str}', not found in pint.UnitRegistry."


@dataclass(frozen=False)
class InvalidDerivedUnitStringError(ValueError, UnitError):
    """
    Raised when the derived unit string obtained by adding the default core unit for the
    primary physical type to the prefix is invalid.
    """

    primary_physical_type: str
    default_units: dict[str, dict[str, str]]

    def __post_init__(self):
        core_unit = self.default_units[self.primary_physical_type]['core']
        combined_unit_str = self.unit_str + core_unit
        self.message = f"Invalid derived unit string '{combined_unit_str}' derived by the primary physical type " \
                       f"'{self.primary_physical_type}' with core unit '{core_unit}', and user-defined prefix " \
                       f"{self.unit_str}, not found in the pint.UnitRegistry."


@dataclass(frozen=False)
class IncompatibleUnitError(ValueError, UnitError):
    """ Raised when the provided unit string is not supported for the given primary physical type and context. """

    primary_physical_type: str = None
    context: str = None

    def __post_init__(self):
        message = f"Unsupported unit string: {self.unit_str}"
        if self.primary_physical_type:
            message = f" for primary physical type {self.primary_physical_type}"
        if self.context:
            message += f" in context {self.context}"
        self.message = message + '.'


@dataclass(frozen=False)
class IncompatibleUnitWarning(UserWarning):
    """ Raised when the input unit string is not compatible with the main unit of the primary physical type. """

    unit_str: str
    primary_physical_type: str
    context: str
    default_units: dict[str, dict[str, str]]

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
class NotFoundError(FileNotFoundError, QDLAnalysisError):
    """ Raised when file is not found. """

    file_path: Path

    def __post_init__(self):
        self.message = f"File '{self.file_path.name}' (local path: {self.file_path}) does not exist."


@dataclass(frozen=False)
class InvalidFileNumbersError(TypeError, QDLAnalysisError):
    """ Raised when the file numbers argument in `FilenameManager.from_file_numbers` is not in the expected format. """

    file_numbers: any

    def __post_init__(self):
        self.message = f"The file_numbers argument must be a range, tuple, iterable, " \
                       f"or a single integer/float value. Got {type(self.file_numbers)} instead."


@dataclass(frozen=False)
class InfoSubclassArgumentNumberError(ValueError, QDLAnalysisError):
    """ Raised when the subclass argument number is larger than the Info subclass available attributes. """

    parsed_object: list
    attrs: list[str]

    def __post_init__(self):
        self.message = f"More arguments in parsed object: {self.parsed_object} " \
                       f"than attributes in Info subclass {self.attrs}."


def assert_bounds(
        value: Sequence[T_] | Iterable[T_] | T_,
        limits: (T_, T_),
        value_name: str,
):
    condition = np.all(limits[0] <= np.array(value)) and np.all(np.array(value) <= limits[1])
    if not condition:
        raise ValueOutOfBoundsError(value, limits, value_name)


def assert_options(
        value: Sequence[T_] | Iterable[T_] | T_,
        options: set[T_] | list[T_] | tuple[T_],
        value_name: str,
        error_class: Type[OutOfOptionsError] = OutOfOptionsError,
        *error_class_args,
        **error_class_kwargs,
):
    if len(np.shape(value)) > 0:
        condition = np.all([v in options for v in value])
    else:
        condition = value in options

    if not condition:
        raise error_class(value, options, value_name, *error_class_args, **error_class_kwargs)


def assert_unit_on_value(
        value: Qty,
        unit: str | ureg.Unit,
        context: str | None = None,
):
    if context is not None:
        with ureg.context(context):
            if not value.is_compatible_with(unit):
                raise IncompatibleUnitError(unit, context=context)
    else:
        if not value.is_compatible_with(unit):
            raise IncompatibleUnitError(unit)


# @dataclass(frozen=False)
# class FilePathNotFoundError(FileNotFoundError, OutOfOptionsError):
#     """ Raised when the file path is not found in a given directory. """
#
#
#     def __str__(self):
#         msg = f"The value  {self.value} for the input argument {self.arg} in {self.mthd} method is invalid. " \
#               f"Choose any of the following instead: {self.allow_list}"
#         return msg
#
#
# @dataclass(frozen=False)
# class ArrayError(QDLAnalysisError):
#     """ Raised when there is an error related to a multi-dimensional object. """
#     array: T_
#     array_name: T_
#
#     def __post_init__(self):
#         self.message = f'Base array exception, unidentified error regarding array {self.array_name} = {self.array}'
#
#
# @dataclass(frozen=False)
# class ArrayShapeError(ArrayError, QDLAnalysisError):
#     """ Raised when the array of interest has an unrecognized shape. """
#     expected_shapes: set[tuple[int, ...]] | list[tuple[int, ...]] | tuple[tuple[int, ...]]
#
#     shape: tuple[int, ...] = field(init=False)
#
#     def __post_init__(self):
#         self.shape = np.shape(self.array)
#         self.message = f"Invalid array shape {self.shape} for {self.array_name} = {self.array}.\n" \
#                        f"Array shape may only be any of the following: {self.expected_shapes}."
