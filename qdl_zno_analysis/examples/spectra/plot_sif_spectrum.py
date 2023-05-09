import pandas as pd

from qdl_zno_analysis.data_classes.data_classes import DataSIF
from qdl_zno_analysis.filename_utils.filename_manager import FilenameManager

pd.set_option('display.max_rows', 20)  # to show 20 rows of the dataframe
pd.set_option('display.max_columns', None)  # to show all columns of the dataframe


fnm = FilenameManager.from_file_numbers(1, 'sif')
file = DataSIF(fnm.filenames[0])

print(file.data)
print(file.metadata)

print(file.integrate_in_region(file.data['wavelength_air'][0], file.data['wavelength_air'][-1],
                               x_axis='wavelength_air', y_axis='nobg_counts'))
print(file.integrate_all())

print(file.quick_plot_labels)

