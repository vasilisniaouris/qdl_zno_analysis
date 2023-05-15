import matplotlib.pyplot as plt
import pandas as pd

from qdl_zno_analysis.data_utils.data import DataSpectrum
from qdl_zno_analysis.examples.example_data_utils import assure_example_data_exist
from qdl_zno_analysis.filename_utils.filename_manager import FilenameManager

pd.set_option('display.max_rows', 20)  # to show 20 rows of the dataframe
pd.set_option('display.max_columns', None)  # to show all columns of the dataframe

assure_example_data_exist('spectra')

# parse file
fnm = FilenameManager.from_file_numbers(2, 'spe', '../_data/spectra')  # get filename easily
file = DataSpectrum(fnm.filenames[0])  # get all the data, metadata and processed data.

# access data and file and filename metadata directly. Everything is unit aware!
file_data = file.data
file_metadata = file.metadata
file_filename_info = file.filename_info
file_filename_info_as_dict = file.filename_info_dict
file_filename_info_as_normalized_dict = file.filename_info_norm_dict

# find area under the curve that is unit aware!
file_area_under_curve = file.integrate_in_region(file.data['wavelength_air'][0], file.data['wavelength_air'][-1],
                                                 x_axis='wavelength_air', y_axis='nobg_counts')
default_total_area_under_curve = file.integrate_all()


# get a copy of the data, renormalized e.g. to location of the maximum of the 'counts' column.
data_norm = file.get_normalized_data(y_axis='counts')


# quick plot spectrum, choose axes from from file.data.column. Can choose non-default units.
file.quick_plot('wavelength_air', 'nobg_counts_per_time_per_power', 'um', 'counts/second/nW')

plt.show()
