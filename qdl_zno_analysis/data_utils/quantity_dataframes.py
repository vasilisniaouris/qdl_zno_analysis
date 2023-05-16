from io import StringIO

import numpy as np
import pint
import pint_pandas
import pandas as pd
from pandas.io.formats import format as fmt

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

    def __repr__(self) -> str:
        """
        Return a string representation for a particular DataFrame. Modified cause PintArray can not properly display
        2D arrays.
        """

        def formatting_function(quantity):
            return "{:{float_format}}".format(
                getattr(quantity, 'magnitude', quantity), float_format=getattr(quantity, 'default_format', '')
            )

        if self._info_repr():
            buf = StringIO()
            self.info(buf=buf)
            return buf.getvalue()

        repr_params = fmt.get_dataframe_repr_params()
        repr_params['formatters'] = [formatting_function] * len(self.columns)
        return self.to_string(**repr_params)
