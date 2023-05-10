
import numpy as np
import pint_pandas
import pandas as pd

from qdl_zno_analysis import Qty


class QuantityDataFrame(pd.DataFrame):
    """
    A subclass of `pandas.DataFrame` that supports `Qty` objects with units via the `pint` library.
    """

    def __getitem__(self, item):
        """
        Returns the value of the underlying data as a pint.Quantity if the underlying data are of `pint[unit]` `dtype`.
        Otherwise, it returns the data as a NumPy array.
        """
        obj = super().__getitem__(item)

        if hasattr(obj, 'pint'):
            if hasattr(obj.pint, 'quantity'):
                return obj.pint.quantity

        return np.array(obj)

    def __setitem__(self, key, value):
        """
        Allows setting of `Qty` objects to the data frame by converting them to a `pd.Series` with
        a `dtype` corresponding to the unit of the `Qty` object.
        """
        if isinstance(value, Qty):
            value = pd.Series(value.m,  dtype=f"pint[{value.u}]")

        super().__setitem__(key, value)

