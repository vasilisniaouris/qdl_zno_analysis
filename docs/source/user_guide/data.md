# Data and Metadata

The `Data` class is used to store, manage and process data from various filetypes and different experiments.
It is designed so that all data are processed under the same framework, thus, making data processing faster 
and easier.

The `Metadata` class serves a similar purpose, but for all the information about the experiment in question. 

## Quick and Dirty

To open a datafile, e.g. a spectrum in either `SIF` or `SPE` formats, get the filename from the `FilenameManager`, 
and then initialize the spectrum.

```pycon
>>> from qdl_zno_analysis import Qty, ureg
>>> from qdl_zno_analysis.data_utils.data import DataSpectrum
>>> from qdl_zno_analysis.filename_utils.filename_manager import FilenameManager
>>> fnm = FilenameManager("001_Smp~FibbedTDEpi_Lsr~CNI-360n-760n_Tmp~5p3_MgF~0_Col~Flt~PLx1-PnH~In_Spt~n0p91u-n2p02u.sif")
>>> file = DataSpectrum(fnm.valid_paths[0], wavelength_offset=0.0 * ureg.nm, pixel_offset=0.0, second_order=False)
```

We can easily **access the data** and many processed data-related colums:

```pycon
>>> file.data.columns
Index(['pixel', 'wavelength_air', 'wavelength_vacuum', 'frequency', 'energy',
       'counts', 'counts_per_cycle', 'counts_per_time',
       'counts_per_time_per_power', 'nobg_counts', 'nobg_counts_per_cycle',
       'nobg_counts_per_time', 'nobg_counts_per_time_per_power',
       'counts_per_time_per_cni_power', 'nobg_counts_per_time_per_cni_power'],
      dtype='object')
```

We can also access data-file and filename **metadata**:

```pycon
>>> file.metadata
MetadataSpectrum(pixel_offset=0, background_per_cycle=<Quantity(300, 'count')>, second_order=False, exposure_time=<Quantity(1.0, 'second')>, cycles=10, calibration_data=<Quantity([ 3.60809931e+02  2.66502734e-02 -2.24827635e-05  1.24084332e-08], 'nanometer')>, input_medium='air', pixel_no=1024, pixels=array([   1,    2,    3, ..., 1022, 1023, 1024]), calibrated_values=<Quantity([360.83655833 360.86314124 360.88967934 ... 377.80916207 377.82875438
 377.84837788], 'nanometer')>, background=<Quantity(3000, 'count')>, background_per_time=<Quantity(300.0, 'count / second')>, background_per_time_per_power={'cni': <Quantity(394.736842, 'count / microwatt / second')>, '': <Quantity(394.736842, 'count / microwatt / second')>})
>>> file.filename_info
FilenameInfo(file_number=1, sample_name='FibbedTDEpi', lasers=SourceInfo(name='CNI', wfe=<Quantity(360.0, 'nanometer')>, wavelength_vacuum=<Quantity(360.102716, 'nanometer')>, wavelength_air=<Quantity(360.0, 'nanometer')>, frequency=<Quantity(832519.292, 'gigahertz')>, energy=<Quantity(3.44302314, 'electron_volt')>, power=<Quantity(760.0, 'nanowatt')>, order=1, medium='Air'), magnetic_field=<Quantity(0.0, 'tesla')>, temperature=<Quantity(5.3, 'kelvin')>, collection_path_optics=OpticsInfo(pinhole='In', filters='PLx1'), spot=<Quantity([-0.91 -2.02], 'micrometer')>)
```

Additionally, we can **quickly plot** the data with unit-aware title-styled labels:


```pycon
>>> file.quick_plot_labels
{'pixel': 'Pixel', 'wavelength_air': 'Wavelength Air (nm)', 'wavelength_vacuum': 'Wavelength Vacuum (nm)', 'frequency': 'Frequency (GHz)', 'energy': 'Energy (eV)', 'counts': 'Counts (count)', 'counts_per_cycle': 'Counts/Cycle (count)', 'counts_per_time': 'Counts/Time (count/s)', 'counts_per_time_per_power': 'Counts/Time/Power (count/s/µW)', 'nobg_counts': 'Nobg Counts (count)', 'nobg_counts_per_cycle': 'Nobg Counts/Cycle (count)', 'nobg_counts_per_time': 'Nobg Counts/Time (count/s)', 'nobg_counts_per_time_per_power': 'Nobg Counts/Time/Power (count/s/µW)', 'counts_per_time_per_cni_power': 'Counts/Time/Cni Power (count/s/µW)', 'nobg_counts_per_time_per_cni_power': 'Nobg Counts/Time/Cni Power (count/s/µW)'}
>>> line = file.quick_plot(x_axis='energy', y_axis='nobg_counts_per_time', x_units='meV', y_units='Hz', color='k', marker='o')
```

The `Data` class provides us with two useful methods. 
- The **integration** method, for find the area under the curve:
  ```pycon
  >>> file.integrate_in_region(369 * ureg.nm, 370 * ureg.nm, x_axis='wavelength_air', y_axis='nobg_counts_per_time')
  <Quantity(227.345742, 'count * nanometer / second')>
  >>> file.integrate_all(x_axis='wavelength_air', y_axis='nobg_counts_per_time')
  <Quantity(3885.87431, 'count * nanometer / second')>
  ```
- The **normalization** method, for normalizing the entire `Data.data` object to the same f(x) value:
  ```pycon
  >>> normalized_dat = file.get_normalized_data(x_value=Qty(369.5, 'nm'), x_axis='wavelength_air', mode='nearest', subtract_min=False)
  ```
Finally, you can input metadata-related parameters that are not of a single value, but arrays. For example, we can
use a pixel-by-pixel background correction:

```pycon
>>> data_filename = FilenameManager.from_file_numbers(1, 'sif').filenames[0]
>>> bg_filename = FilenameManager.from_file_numbers(2, 'sif').filenames[0]
>>> file_bg = DataSpectrum(bg_filename, wavelength_offset = 0.0 * ureg.nm)
>>> file = DataSpectrum(bg_filename, wavelength_offset = 0.0 * ureg.nm, background_per_cycle=file_bg.data['counts_per_cycle'])
```

## The data superclass

The `Data` class defines a few properties that can be used throughout all `Data` subclasses. 

### Data - quantity dataframes

The most important, of course, is the property `Data.data`, where all the ... data are stored in the form of a 
`QuantityDataFrame`. A `QuantityDataFrame` differs from a regular `pandas.Dataframe` in two regards. 

- It is unit aware, meaning that its objects are aware of any `pint.Quantity` objects they are initialized with.
- Instead of returning `pandas.Series` upon a call, it returns either a `pint.Quantity` or a `numpy.ndarray` object.
  This is done with the idea that we are not usually interested in the `pandas.Series` objects themselves, and most 
  often than not, we convert them to the aforementioned objects anyway.  

### File metadata

The second most important is `Data.metadata`, a `Metadata`-type object (a dataclass with redefined str, repr and 
to-dictionary methods), that holds all the information carried over by the original file and then-some. An example
would be the exposure time, cycle count, number of pixels, the background counts, etc. of a spectrum.

### Filename metadata

Other noteworthy properties are
- the `Data.filename_info` that store information taken from the filename of the data file (See more
  [here](filenames.md#filenameinfo)),
- other properties directly derived from the `FilenameManager` (See more [here](filenames.md#filenamemanager)), such as 
  `Data.filename`, `Data.filename_info_dict`, and `Data.filename_info_norm_dict`.
- `Data.quick_plot_labels`, a dictionary of title-styled strings that are aware of the default `Data.data` column units
  (e.g. "Counts/Second/Power (count/s/μW)").

### Useful methods
The `Data` class defines three noteworthy functions that could be used in different types of experimenta datasets.

- `Data.integrate_in_region` allows for easy integration under any data curve, either with exact x-axis values (via 
  linear interpolation), or approximate ones (by finding the nearest x-axis value from the requested one).
- `Data.get_normalized_data` may help users normalize data faster.
- `Data.quick_plot` plots any two data-columns with the corresponding unit aware, title-styled labels. 

## Using predefined Data subclasses 

Pending...

## Defining your own Data and Metadata subclasses

Pending...