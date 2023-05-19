
from pint_xarray import unit_registry as _ureg


class _Q(_ureg.Quantity):
    """
    A `pint.Quantity` subclass based on the xarray UnitRegistry. The goal of this class is to allow for
     operations and comparisons between Quantity objects and other objects, such as `int`, `float`, or `np.ndarray`,
     assuming that the latter has the same units as the Quantity.

     This class is designed to be used as a mixin class, and should not be instantiated.
     """

    def __add__(self, other):
        if isinstance(other, _ureg.Quantity):
            return super().__add__(other)
        else:
            return _Q(self.m + other, self.u)

    def __sub__(self, other):
        if isinstance(other, _ureg.Quantity):
            return super().__sub__(other)
        else:
            return _Q(self.m - other, self.u)

    def __le__(self, other):
        if isinstance(other, _ureg.Quantity):
            return super().__le__(other)
        else:
            return self.m <= other

    def __lt__(self, other):
        if isinstance(other, _ureg.Quantity):
            return super().__lt__(other)
        else:
            return self.m < other

    def __eq__(self, other):
        if isinstance(other, _ureg.Quantity):
            return super().__eq__(other)
        else:
            return self.m == other

    def __ne__(self, other):
        if isinstance(other, _ureg.Quantity):
            return super().__ne__(other)
        else:
            return self.m != other

    def __ge__(self, other):
        if isinstance(other, _ureg.Quantity):
            return super().__ge__(other)
        else:
            return self.m >= other

    def __gt__(self, other):
        if isinstance(other, _ureg.Quantity):
            return super().__gt__(other)
        else:
            return self.m > other


_ureg.Quantity = _Q
