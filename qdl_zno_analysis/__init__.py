"""
This package serves as a data analysis toolkit for research performed on ZnO at the Quantum Defect Laboratory of the
University of Washington under the supervision of Prof. Kai-Mei C. Fu.
"""

from qdl_zno_analysis._pint import _ureg

ureg = _ureg
""" `pint.UnitRegistry` instance to be used throughout the package. Do not create a new instance of this class. """

Qty = ureg.Quantity
""" `pint.Quantity` class to be used throughout the package. Do not create a new instance of this class. """

