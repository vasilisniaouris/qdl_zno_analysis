# Data

The `Data` class is used to store, manage and process data from various filetypes and different experiments.
It is designed so that all data are processed under the same framework, thus, making data processing faster 
and easier.

This part of the package (which is the core of the package, really), utilizes xarrays and Pint units. 
The XArray package allows us to easily work with named multidimensional datasets, and is an 
extension of pandas dataframes and numpy arrays. 
Using Pint, we can make our datasets unit-aware, never again having to worry about units (even when you integrate!).

The cool thing about `Data` is that it can handle multidimensionality that stems from reading multiple files and not the
data directly.
This means that if you take, for example, multiple spectra with different experimental conditions (e.g., varying power), 
the `DataSpectrum` class can recognize these different conditions (if the filename conventions are followed).

## Available data types:

These are all the available data types this package can help you analyze as of now:

1. Spectra: `DataSpectrum`

Learn how to define you own `Data` subclass in [Defining your own Data subclasses](#defining-your-own-data-subclasses) 
section.

## Examples

I wrote five different examples explaining the main functionality of the `Data` class by using the `DataSpectrum` class 
as an example. 

In this [google drive link](https://drive.google.com/drive/folders/1yPZ5BKT5_7zENuy6-cKib_W9ze5gPXNA?usp=share_link) 
you can find these examples on how to use the data classes as jupyter notebook files alongside the relevant data.
More specifically:

1. How to read a single `SIF` spectrum. [google drive link](https://drive.google.com/file/d/1pqfbNp2CAD-ORvGqwlih8gJerV4Lyhhf/view?usp=drive_link)
2. How to read a single `SPE` spectrum. [google drive link](https://drive.google.com/file/d/1AtRqesX-aK6NcTAgZX31jybvWMDKMPbS/view?usp=drive_link)
3. How to read multi-frame `SPE` spectra (single file). [google drive link](https://drive.google.com/file/d/1pN8nSk3ZP5f1WMO8z4DnAftaIrXWIWwy/view?usp=drive_link)
4. How to read many files (power-series) of multi-frame `SPE` spectra. [google drive link](https://drive.google.com/file/d/1uimDFpWWbAwNcnQ5kGwGMue7fuLS5Dym/view?usp=drive_link)
5. How to read many files (PLE) of single-frame `SIF` spectra. [google drive link](https://drive.google.com/file/d/1UXXQ9vTh0AZdDdk1hp21lLWR0HRGLjEt/view?usp=drive_link)
6. Pending: How to read many files with two file-related dimensions (e.g. excitation power and excitation wavelength).

These files and related data can be easily downloaded them on your local package installation path using the 
`examples.example_data_utils.get_example_local_filepath` method.


## Defining your own Data subclasses

Pending...