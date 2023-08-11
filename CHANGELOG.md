# Change Log

- 08/10/2023; v0.3.1:
  - Code:
    - Updated the way package-related errors are defined.
    - Added utility methods to more easily detect and "heal" data outliers with a simple iterative
      rolling statistics method used in `utils.find_outliers`.
    - Moved excitation power dictionary detection from `DataSpectrum` to `Data`.
    - Added cosmic ray removal option in `DataSpectrum` based on outlier finder code.
  - Documentation:
    - Updated data handling section.

- 06/27/2023; v0.3.0:
  State of package:
  - Code:
    - Created the `Data` superclass which will be used to define and manipulate data files.
      `Data` utilizes multidimensional xarrays. By default, you can read groups of files collectively with this class.
      This class contains two main data-holding attributes, `data` and `metadata`. We make sure to include the 
      filename information to the `data` and `metadata` attributes. 
      This class has three important methods, `integrate`, `get_normalized_data`, and `quick_plot`.
      - As of now, we have only defined one such subclass, `DataSpectrum` (which is one of the most complex to define).
        This subclass can manage spectral data contained in SIF or SPE files.
    - Wrote up examples that are contained in the `example` sub-package.
      The example data-files can be automatically downloaded from Google Drive.
    - Added a plethora of utility classes and functions to assist in the creation of the above.
  - Documentation:
    - Populated the documentation on filename handling and data handling.

... Forsaken changes

- 04/20/2023; v0.1.0: Initial commit.
  - Code:
    - Created `error` module to handle customized errors throughout the package.
    - Created `utils` module to implement miscellaneous methods throughout the package.
    - Created `constants` module that contains all the primary invariable physical values used throughout the package.
    - Created `typevars` module to hold all the complex typevars used throughout the package.
    - Created `filename_parsing` module to parse filenames.
    - Created `filename_info` module with `Info` dataclasses to hold filename metadata. Filename conventions were 
      determined. 
    - Created `FilenameManager` that utilizes `filename_parsing` methods and `Info` classes to help the user manage 
      metadata stored in a filename. 
  - Documentation:
    - Wrote introduction
    - Started on User guide. Populated with information on filename-related functionalities.
