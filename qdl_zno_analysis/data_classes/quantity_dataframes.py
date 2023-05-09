
import numpy as np
import pint_pandas
import pandas as pd

from qdl_zno_analysis import Qty


class QuantityDataFrame(pd.DataFrame):

    def __getitem__(self, item):
        obj = super().__getitem__(item)

        if hasattr(obj, 'pint'):
            if hasattr(obj.pint, 'quantity'):
                return obj.pint.quantity

        return np.asarray(obj)

    def __setitem__(self, key, value):
        if isinstance(value, Qty):
            value = pd.Series(value.m,  dtype=f"pint[{value.u}]")

        super().__setitem__(key, value)

