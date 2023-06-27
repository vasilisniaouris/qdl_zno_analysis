"""
This module is intended to be used as an internal helper module for importing dependencies
that are optional in the base installation.
"""

# Visualization dependencies
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    HAS_VISUALIZATION_DEP = True
except ModuleNotFoundError:
    mpl = None
    plt = None

    HAS_VISUALIZATION_DEP = False


# Spectroscopy dependencies
try:
    import spe_loader as sl
    import xmltodict
    from sif_parser import np_open as read_sif

    HAS_SPECTROSCOPY_DEP = True
except ModuleNotFoundError:
    sl = None
    xmltodict = None
    read_sif = None

    HAS_SPECTROSCOPY_DEP = False

# Example data dependencies
try:
    import requests
    import gdown

    HAS_EXAMPLE_DATA_DEP = True
except ModuleNotFoundError:
    requests = None
    gdown = None

    HAS_EXAMPLE_DATA_DEP = False


HAS_ALL_DEP = all([
    HAS_VISUALIZATION_DEP,
    HAS_EXAMPLE_DATA_DEP,
    HAS_SPECTROSCOPY_DEP
])
