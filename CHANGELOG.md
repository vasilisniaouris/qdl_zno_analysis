# Change Log

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
