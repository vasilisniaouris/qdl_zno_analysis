"""
This package serves as a data analysis toolkit for research performed on ZnO at the Quantum Defect Laboratory of the
University of Washington under the supervision of Prof. Kai-Mei C. Fu
"""

from pint import Quantity, _DEFAULT_REGISTRY

ureg = _DEFAULT_REGISTRY
""" `pint.UnitRegistry` instance to be used throughout the package. Do not create a new instance of this class. """
Qty = Quantity
""" `pint.Quantity` class to be used throughout the package. Do not create a new instance of this class. """

