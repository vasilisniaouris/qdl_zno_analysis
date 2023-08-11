# Filenames

The acquisition of large amounts of data often results in a large number of data files that can be difficult to 
navigate. To simplify this process, it is helpful to use strict but versatile file-naming conventions. 
In this guide, we will present the file-naming conventions used in our package and provide examples of valid filenames. 
We will also showcase some common mistakes to avoid.

Our file-naming system was developed to store experimental metadata concisely in a given data file. 
This system has evolved over the years and has recently been updated for increased functionality. 
We use the `FilenameManager` class to easily access the metadata from one or more files simultaneously. 
The `FilenameManager` provides three methods to parse through multiple filenames, which we will demonstrate in this
guide.

To use this guide, it is helpful to define some terms. 
"Experimental metadata" refers to the information stored in a data file that describes the experiment's conditions, 
equipment, and procedures. "FilenameManager" is a class that provides easy access to filename metadata.

In this guide, we will first present the rules that dictate our filename conventions, which are designed to 
simplify data file navigation.  We will then provide examples of valid filenames and showcase some of the most common 
mistakes to avoid. Additionally, we will show you how to use the `FilenameManager` to easily access the metadata from 
one or more files simultaneously.  For more information on how to use the `FilenameManager` and the `FilenameInfo` 
classes, go [here](#accessing-filename-metadata-with-filenameinfo-and-filenamemanager). 
For a thorough guide on how to write a valid filename, go [here](#filename-conventions). 
For a comprehensive list of all the predetermined filename headers, go 
[here](#comprehensive-reference-guide-to-filename-conventions).

## Quick and dirty

To easily access the filename metadata from a file that follows the 
[filename conventions](#comprehensive-reference-guide-to-filename-conventions), use the `FilenameManager` class. 
Three methods are provided to parse through multiple filenames.

```pycon
>>> from qdl_zno_analysis.filename_utils.filename_manager import FilenameManager
>>> filenames = ['001_Msc~Example1a_Tmp~1p2.csv', '001_Msc~Example1b_Tmp~1p2.txt', '002_Msc~Example2_Tmp~1p2.csv']
>>> folder = 'some_folder'
>>> # Method 1: Directly provide a list of filenames and the folder path. 
>>> # This is useful if you already have a list of filenames that you want to analyze.
>>> fnm = FilenameManager(filenames, folder)
>>> # Method 2: Provide a range of file numbers and a list of file types. 
>>> # This is useful if you want to analyze a range of files with specific file types, 
>>> # such as all CSV files from file 1 to file 2. 
>>> fnm = FilenameManager.from_file_numbers(range(1, 3), None, folder)
>>> # Method 3:  Find all files in a folder that match a specific string in their filename. 
>>> # This is useful if you want to analyze all files in a folder that have a certain pattern in their name, 
>>> # such as all files with "Msc~Example" in the filename. 
>>> fnm = FilenameManager.from_matching_string('*Msc~Example*', folder)
```

All methods will result in the same object. The class that stores all parsed filename information is called 
`FilenameInfo`. Through this class you can access all the information you are interested in three forms: 

- a python dataclass,
- a dictionary with embedded sub-dictionaries, or
- a normalized/flattened dictionary.

See more on `FilenameInfo` [here](#filenameinfo). You can easily access all three from the `fnm` object we just created:

```pycon
    >>> fnm.filenames
    >>> fnm.filename_info_list  # list of FilenameInfo objects for each filename
    >>> fnm.filename_info_dicts  # dictionary of embedded sub-dictionaries
    >>> fnm.changing_filename_info_dict  # normalized/flattened dictionary
```

For more information on `FilenameManager`, go [here](#filenamemanager).

## Comprehensive reference guide to filename conventions

This section is here to provide an experienced user with a memory refresher of the filename conventions.
If you are a new user, you may wat to skip forward to the next section.

### Special symbols

To separate and properly assign information:
- Equality/Assignment: `~`
- Primary header separator: `_`
- Secondary separators in order: `-`, `;`, `,`.

### Input types

| Input Type | Format                                                                                              | Special Symbols                                                                                                                                        | Example                                              |
|------------|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| String     | Letters, Numbers, no special symbols                                                                | N/A                                                                                                                                                    | "ExampleName1"                                       |
| Numeric    | - Sign (optional)<br>- Integer part (mandatory) <br>- Floating part (optional)<br>- Unit (optional) | - "p" for positive (optional) or "n" for negative<br>- "p" for "." followed by at least one numeric<br>- any character valid as a unit prefix or unit. | "n0p13n"                                             |
| List Entry | Strings or numerics in a serial format, separated by secondary separators.                          | Secondary separators                                                                                                                                   | "Msc~Potato-Tomato"<br>"Col~PhH~40u-Flt~LP380;BP370" |

### Headers

Here 
- \[I\] stands for input,
- \[H\] stands for header name, 
- \[S(N)\] stands for subheader, with N (optional) being the order in which subheader comes in the list of subheaders within a header,
- \[SS(N)|] stands for subsubheader.
- etc.

The following use all special symbols except the primary separator.

| Format Type | Format                                               | Examples                                                                                                                                                  |
|-------------|------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dictionary  | \[H\]~\[S\]~\[I\]-\[S\]~\[I\]...                     | "Lsr~Wvl~737p8n_Pwr~100n"                                                                                                                                 |
| List        | \[H\]~\[I for S1\]-\[I for S2\]...                   | "Lsr~Matisse-737p8-100n-2-Air"                                                                                                                            |
| Nested      | \[H\]~\[S\]~\[SS\]~\[I\];\[SS\]~\[I\]-\[S\]~\[I\]... | "Lsr~Matisse~Wvl~737p8;Pwr~10n;Ord~2-Toptica~Wvl~368p1;Pwr~20n"<br>"Lsr~Matisse~Wvl~From~737p7,To~737p9,Step~50p;Pwr~10n;Ord~2-Toptica~Wvl~368p1;Pwr~20n" |


#### Filename headers

Example: 

```text
001_Smp~ZnO1_Lsr~Matisse-737p8-10n-2_Col~HWP~45deg-PnH~40u-Flt~LP380;BP370_MgF~5_Tmp~120m_Spt~From~n2u;3u-To~4u;8u-StepNo~61;26_MsT~ConfocalScanPL.csv
```

| Header name | Full name                              | `Info`-class property/attribute name       | `Info` subclass  | Input types                                          | Examples                       |
|-------------|----------------------------------------|--------------------------------------------|------------------|------------------------------------------------------|--------------------------------|
| "FNo"       | file number                            | fno, file_number                           | None             | Integer                                              | "001"                          |
| "Smp"       | sample name                            | smp, sample                                | None             | String (rec), Numeric, List entry                    | "Smp~Zn1"                      |
| "MgF"       | magnet field                           | mgf, magnet_field                          | None, `ScanInfo` | String, Numeric (rec), List entry, Dictionary format | "MgF~5"                        |
| "Tmp"       | temperature                            | tmp, temperature                           | None, `ScanInfo` | String, Numeric (rec), List entry, Dictionary format | "Tmp~120m"                     |
| "MsT"       | measurement type                       | mst, measurement_type                      | None             | String (rec), Numeric, List entry                    | "MsT~Lifetime"                 |
| "Msc"       | miscellaneous                          | msc, miscellaneous                         | None             | String, Numeric, List entry                          | "Msc~Any-Information;you,want" |
| "Spt"       | spot                                   | spt, spot                                  | None, `ScanInfo` | List entry, Dictionary format                        | "Spt~n1p05u-2p45u"             |
| "Lsr"       | lasers                                 | lsr, lasers                                | `SourceInfo`     | Dictionary format, List format                       | "Lsr~Matisse-737p8n-10n-2"     |
| "Exc"       | excitation path optics                 | exc, excitation_path_optics                | `OpticsInfo`     | Dictionary format, List format                       | "Exc~Plr~V-HWP~45deg"          |
| "EnC"       | excitation and collection path optics  | enc, excitation_and_collection_path_optics | `OpticsInfo`     | Dictionary format, List format                       | "EnC~QWP~90deg"                |
| "Col"       | collection path optics                 | exc, collection_path_optics                | `OpticsInfo`     | Dictionary format, List format                       | "Col~PnH~40u-Flt~LP380;BP370"  |
| "RFS"       | RF sources                             | rfs, rf_sources                            | `SourceInfo`     | Dictionary format, List format                       | "RFS~Name~Synth-Frq~8p8G"      |
| "RFL"       | RF lines                               | rfl, rf_lines                              | `RFLinesInfo`    | Not Implemented                                      | Not Implemented                |
| "Misused"   | Misused, other (For internal use only) | other                                      | None             | Any elements not matching the conversions            | Do not use! Use "Msc" instead! |


#### Source headers
For lasers and RF sources. Example: 

```text
Lsr~Matisse-737p8-10n-2
```

| Header name | Full name            | `Info`-class property/attribute name                     | `Info` subclass  | Input types                                                                         | Examples                    |
|-------------|----------------------|----------------------------------------------------------|------------------|-------------------------------------------------------------------------------------|-----------------------------|
| "Name"      | name                 | name                                                     | None             | String (rec), Numeric, List entry                                                   | "Matisse"                   |
| "Wvl"       | wavelength           | wfe, wvl_air, wvl_vac, wavelength_air, wavelength_vacuum | None, `ScanInfo` | String, Numeric (rec), List entry, Dictionary format                                | "Wvl~737p8n"                |
| "Frq"       | frequency            | wfe, frq, frequency                                      | None, `ScanInfo` | String, Numeric (rec), List entry, Dictionary format                                | "Frq~8p8G"                  |
| "Eng"       | energy               | wfe, eng, energy                                         | None, `ScanInfo` | String, Numeric (rec), List entry, Dictionary format                                | "Eng~334p5m"                |
| "Pwr"       | power                | pwr, power                                               | None, `ScanInfo` | String, Numeric (rec), List entry, Dictionary format                                | "Pwr~100n"                  |
| "Ord"       | Nth order generation | ord, order                                               | None             | Integer                                                                             | "Ord~2"                     |
| "Mdm"       | Input medium         | mdm, medium                                              | None             | String: "Air" (default) or "Vacuum".<br>Only affects wfe if wavelength is provided. | "Mdm~Air"                   |
| "Msc"       | miscellaneous        | msc, miscellaneous                                       | None             | String, Numeric, List entry                                                         | "Msc~Any;Extra,Information" |


#### Optical path headers
For excitation, collection, and excitation/collection path optics. Example:

```text
HWP~n21p2deg-QWP~0p33rad-Plr~V-PnH~40u-Flt~LP380;BP370
```

| Header name  | Full name               | `Info`-class property/attribute name | `Info` subclass  | Input types                                          | Examples                    |
|--------------|-------------------------|--------------------------------------|------------------|------------------------------------------------------|-----------------------------|
| "HWP", "WP2" | half waveplate angle    | hwp, wp2, half_waveplate_angle       | None, `ScanInfo` | String, Numeric (rec), List entry, Dictionary format | "HWP~n21p2deg"              |
| "QWP", "WP4" | quarter waveplate angle | qwp, wp4, quarter_waveplate_angle    | None, `ScanInfo` | String, Numeric (rec), List entry, Dictionary format | "QWP~0p33rad"               |
| "Plr"        | polarizer               | plr, polarizer                       | None, `ScanInfo` | String, Numeric (rec), List entry, Dictionary format | "Plr~V"                     |
| "PnH"        | pinhole                 | pnh, pinhole                         | None             | String, Numeric (rec), List entry                    | "PnH~40u"                   |
| "Flt"        | filter                  | flt, filter                          | None             | String, Numeric (rec), List entry                    | "Flt~LP380;BP370"           |
| "Msc"        | miscellaneous           | msc, miscellaneous                   | None             | String, Numeric, List entry                          | "Msc~Any;Extra,Information" |


#### RF-line headers
Not implemented, yet.

#### Scan Headers

Example:

```text
Spt~From~n2u;3u-To~4u;8u-StepNo~61;26
```

| Header name                        | Full name               | `Info`-class property/attribute name | `Info` subclass | Input types                            | Examples           |
|------------------------------------|-------------------------|--------------------------------------|-----------------|----------------------------------------|--------------------|
| "Start", "From", "Init", "Initial" | start value             | start, initial                       | None            | String, Numeric (rec), List entry      | "From~n2u;3u"      |
| "Stop", "To", "Final"              | final value             | stop, final                          | None            | String, Numeric (rec), List entry      | "To~4u;8u"         |
| "Step", "Res", "Resolution"        | step size               | step                                 | None            | String, Numeric (rec), List entry      | "Step~100n;50n"    |
| "StepΝο", "Res", "Resolution"      | step number             | step_no                              | None            | String, Numeric (rec), List entry      | "StepNo~61;26"     |
| "StepΝο", "Res", "Resolution"      | step number             | step_no                              | None            | String, Numeric (rec), List entry      | "StepNo~61;26"     |
| "Rate"                             | scanning rate           | rate                                 | None            | String, Numeric (rec), List entry      | "Rate~0p005;0p003" |
| "Dur", "Duration"                  | total accumulation time | duration                             | None            | String, Numeric (rec), List entry      | "Dur~100s"         |
| "Mode"                             | scan mode               | mode                                 | None            | String (e.g. "Continuous", "Discreet") | "Mode~Continuous"  |
| "Msc"                              | miscellaneous           | msc, miscellaneous                   | None            | String, Numeric, List entry            | "Msc~Anything"     |


## Filename conventions

As per our lab's tradition, each filename starts with a n-digit numeral, e.g. `001` to indicate that this is the 1st
file in the folder. This convention helps us identify the files in our notes.

### Headers

The file number is the only metadatum that is stored without any prefix or _header_.
A _header_ is usually a 3-letter sequence of letters that corresponds to a specific metadatum family.
For example, the backend file number header is `FNo`. The header is followed by `~` to indicate an assignment or
equality. In our file number example, it would be `FNo~001`, however, as we said, this header can be omitted.
As we progress through this section, we will slowly add more metadata to our dummy filename. As of now our filename is: 

```text
001.csv 
```

### String values
Since we are performing different types of measurements, and the more often than not, the file extension is not enough 
to determine the type of measurement stored in a file, we define the `MsT` header. e.g. `MsT~PL`. The primary 
symbol that is used to differentiate between headers is `_`. This is a special symbol and can not be used for anything
else. Hence, our filename becomes:

```text
001_MsT~PL.csv
```

As a material lab, we work with different samples. Hence, another simple header we define is `Smp` for sample, 
e.g. `Smp~ZnO1`. Our filename is:

```text
001_Smp~ZnO1_MsT~PL.csv
```

### Numerical values and units

The environment in which an experiment is performed at is also important.
We therefore define two more headers, `MgF` and `Tmp` for magnetic field and temperature, respectively. 
Both of these usually take numeric values, which follow specific conventions:

- We start with an optional sign `n` used for negative and `p` for positive (can be omitted).
- Then comes the integer part. If its is `0`, then it is mandatory to state so.
- The optional floating part should always start with `p` and be followed by at lease one numeral.
- The optional unit part. Units are handles with `pint.Quantity` objects, and you can find more information about them
  here. There are three formats that units can be defined in.
  - No units. If the header you are populated has default units (e.g. `K` for temperature). If the unit is
    omitted, then the program will assume it is in the default units.
  - Only unit prefix. You can use any valid unit prefix defined in `pint.UnitRegistry` (e.g. `m` for milli). 
    The program will assume that the full unit is the prefix + default core unit (e.g. `mK`). 
    **Note**: Beware that some units are also identified as prefixes. In that case the program will assume
    that you are using the prefix, not the unit. For example setting `MgF~5T` would result in a teratesla unit! Hence, 
    omit the unit all together when possible. 
  - The full unit. You can use any valid combination of prefixes and base units (e.g. `Tmp~120mK`)

With these in mind, we can now update our filename:

```text
001_Smp~ZnO1_MgF~5_Tmp~120m_MsT~PL.csv
```

### Subheaders

Since our lab works primarily with optical setups, properly storing input laser information is paramount. Hence,
we define a header `Lsr` that can store multiple information. This header, alongside some others we will discuss later,
can take subheaders. For laser information, these are: 

- `Name`
- `Wvl`, `Frq`, or `Eng` for wavelength, frequency or energy respectively. Only define one.
- `Pwr` for power, 
- `Ord` for frequency conversion order and 
- `Mdm` for medium (only important if you provide wavelength, to distinguish between `air` (default) and `vacuum`). 

We can define such a header in two ways, a _dictionary_ or a _list format_. In both cases
secondary separators are used. The defined secondary separators in order of increasing depth are `-`, `;`, `,`.

---

#### Subheader dictionary format.

A dictionary format is defined by a series of subheader-value items separated by the appropriate secondary 
separator (in this case: `-`), e.g `Lsr~Wvl~737p8n-Pwr~10n`. Subheaders, same as headers do not need to be provided in 
a specific order, hence the following is also valid: `Lsr~Pwr~100n-Wvl~737p8n`. This is best used when you do not need 
to define all the subheadings. 

#### Subheader list format.

A list format is defined by a series of values separated by the appropriate separator (in this case, again, `-`), e.g.
`Lsr~Matisse-737p8-10n-2-Air`. This definition assumes that you provide the values is a specific predefined order. This 
string tells us that the Matisse laser was used, at 737.8 nm measured in air, then was converted via SHB to 368.9 nm
and the reference power was 10 nW.

---

Going back to the filename we are building, we can see its starting to flesh out:

```text
001_Smp~ZnO1_Lsr~Matisse-737p8-10n-2_MgF~5_Tmp~120m_MsT~PL.csv
```

### List entry

In an optics lab, we often use various optical elements in the beam path. Such elements can be listed in three
headers:

- `Exc` for excitation path
- `Col` for collection path
- `EnC` for the common path between excitation and collection

Each of these headers can take various subheaders, such as: 

- `HWP` or `WP2` for half waveplate angle
- `QWP` or `WP4` for quarter waveplate angle
- `Plr` for polarizer
- `PnH` for pinhole
- `Flt` for filters

An example of this subheader would look like: `Col~HWP~45deg-PnH~40u-Flt~LP380;BP370`. You probably noticed the use of
the second secondary separator, `;`. Here we are using it so that we can assign a list of filters to the `Flt` 
subheader. With this, our filename looks like:

```text
001_Smp~ZnO1_Lsr~Matisse-737p8-10n-2_Col~HWP~45deg-PnH~40u-Flt~LP380;BP370_MgF~5_Tmp~120m_MsT~PL.csv
```

### Scan Entry

The final piece of the puzzle, scans! Often times you spend a lot of time performing measurements manually, and that 
can certainly be tedious. Then, sometimes, you are lucky (or hardworking) and you find (or compile) a program that 
automates this process. In order to be able to tell the scan parameters from a plethora of scan files, I created a 
special entry that can be implemented on almost all headers. 

Let's say that we want to perform a confocal scan, which is scanning e.g. a mirror on a piezo-stage. A scan can take
multiple subheaders:

- `Start`, `From`, `Initial` or `Init` for the starting value of the scan.
- `Stop`, `To` or `Final` for the final value of the scan.
- `Step`, `Resolution` or `Res` for the step size (calculated by the start, stop and step number values, if left empty)
- `StepNo` for the step number (calculated by the start, stop and step size values, if left empty)
- `Rate` for the scanning rate.
- `Duration` or `Dur` for the total accumulation time.
- `Mode` for the scan mode (e.g. continuous vs discrete)

Let's use the header `Spt` that stands for spot for this example. A simple spot example can be `Spt~1` (meaning that you
are on your first identified spot), or `Spt~1p1u;n2p67u` meaning you are at the coordinates (x = 1.1 μm, y = -2.67 μm). 
A scan Spot will look like: `Spt~From~n2u;3u-To~4u;8u-StepNo~61;26` which means that we scan from (x = -2 μm, y = 3 μm) 
to (x = 4 μm, y = 8 μm) with a total step number of (xsn = 61, ysn = 26). This would automatically calculate a step size
of (xs = 0.1 μm, ys = 0.2 μm). This will get us a quite hefty but informative filename:

```text
001_Smp~ZnO1_Lsr~Matisse-737p8-10n-2_Col~HWP~45deg-PnH~40u-Flt~LP380;BP370_MgF~5_Tmp~120m_Spt~From~n2u;3u-To~4u;8u-StepNo~61;26_MsT~ConfocalScanPL.csv
```

**Note**: a scan can not be set with the list format, but only the dictionary format!!

### Nested sources

Only for sources, we are able to input more than one element via the dict format. A simple example with two laser 
sources would be:
```text
Lsr~Matisse~Wvl~737p8;Pwr~10n;Ord~2-Toptica~Wvl~368p1;Pwr~20n
```
Where the name of each source is used as the dictionary header (notice the `~` after their name)
We can even have a scanning source while the other source is stationary.
```text
Lsr~Matisse~Wvl~From~737p7,To~737p9,Step~50p;Pwr~10n;Ord~2-Toptica~Wvl~368p1;Pwr~20n
```

### Miscellaneous information

Of course, we allow the user to input more information for both headers and subheaders. The header miscellaneous is 
called `Msc`. Only specific headers can hold more information in the `Msc` subheader, and these are the Sources, 
the Optical paths, RF-lines and the Scans. 

### Wrongful usage of conventions

If a filename includes wrongful notation, performance is not guaranteed. 
For the most common mistake, using `_` between any value, we make sure to store all that misused information as a list.


## Accessing filename metadata with FilenameInfo and FilenameManager
Now that we know how to name our files, we can use the `FilenameManager` for sort to easily access all of this 
stored information. For each file, the FilenameManager creates a `FilenameInfo` dataclass. Each filename header will be
converted to a class attribute. The user will not need to interface with the `FilenameInfo` object, unless they need 
some additional functionality. 

As we said earlier, unit-based values will be stored as `pint.Quantity` objects. If that fails, the values fall back
to strings. Lists are stored as lists, and subheaders within embedded `Info` objects. 

### FilenameInfo
Nevertheless, lets see how our example filename would like as a `FilenameInfo` object. The following snippet shows how
to initialize a `FilenameInfo` object from a filename.

```pycon
>>> from qdl_zno_analysis.filename_utils.filename_info import FilenameInfo
>>> filename = '001_Smp~ZnO1_Lsr~Matisse-737p8-10n-2_Col~HWP~45deg-PnH~40u-Flt~LP380;BP370_MgF~5_Tmp~120m_Spt~From~n2u;3u-To~4u;8u-StepNo~61;26_MsT~ConfocalScanPL.csv'
>>> fni = FilenameInfo.from_filename(filename)
>>> fni
FilenameInfo(file_number=1, sample_name='ZnO1', lasers=SourceInfo(name='Matisse', wfe=<Quantity(737.8, 'nanometer')>, wavelength_vacuum=<Quantity(369.001625, 'nanometer')>, wavelength_air=<Quantity(368.9, 'nanometer')>, frequency=<Quantity(812442.109, 'gigahertz')>, energy=<Quantity(3.35999058, 'electron_volt')>, power=<Quantity(10.0, 'nanowatt')>, order=2, medium='Air'), magnetic_field=<Quantity(5.0, 'tesla')>, temperature=<Quantity(120.0, 'millikelvin')>, collection_path_optics=OpticsInfo(half_waveplate_angle=<Quantity(45.0, 'degree')>, pinhole=<Quantity(40.0, 'micrometer')>, filters=['LP380', 'BP370']), measurement_type='ConfocalScanPL', spot=ScanInfo(start=<Quantity([-2.  3.], 'micrometer')>, stop=<Quantity([4. 8.], 'micrometer')>, step=<Quantity([0.1 0.2], 'micrometer')>, step_no=array([61, 26]), direction=array([1, 1])))
```

You see that in the printed statement, only the fields that have been initialized are printed out. You can access each 
element you by calling the corresponding dataclass attribute:
```pycon
>>> fni.filenumber
1
>>> fni.lasers
SourceInfo(name='Matisse', wfe=<Quantity(737.8, 'nanometer')>, wavelength_vacuum=<Quantity(369.001625, 'nanometer')>, wavelength_air=<Quantity(368.9, 'nanometer')>, frequency=<Quantity(812442.109, 'gigahertz')>, energy=<Quantity(3.35999058, 'electron_volt')>, power=<Quantity(10.0, 'nanowatt')>, order=2, medium='Air')
```

To access `fni` as a dictionary:

```pycon
>>> fni.to_dict()
{'file_number': 1, 'sample_name': 'ZnO1', 'lasers': {'name': 'Matisse', 'wfe': <Quantity(737.8, 'nanometer')>, 'wavelength_vacuum': <Quantity(369.001625, 'nanometer')>, 'wavelength_air': <Quantity(368.9, 'nanometer')>, 'frequency': <Quantity(812442.109, 'gigahertz')>, 'energy': <Quantity(3.35999058, 'electron_volt')>, 'power': <Quantity(10.0, 'nanowatt')>, 'order': 2, 'medium': 'Air', 'miscellaneous': None}, 'rf_sources': None, 'magnetic_field': <Quantity(5.0, 'tesla')>, 'temperature': <Quantity(120.0, 'millikelvin')>, 'excitation_path_optics': None, 'collection_path_optics': {'half_waveplate_angle': <Quantity(45.0, 'degree')>, 'quarter_waveplate_angle': None, 'polarizer': None, 'pinhole': <Quantity(40.0, 'micrometer')>, 'filters': ['LP380', 'BP370'], 'miscellaneous': None}, 'exc_and_col_path_optics': None, 'rf_lines': None, 'measurement_type': 'ConfocalScanPL', 'miscellaneous': None, 'spot': {'start': <Quantity([-2.  3.], 'micrometer')>, 'stop': <Quantity([4. 8.], 'micrometer')>, 'step': <Quantity([0.1 0.2], 'micrometer')>, 'step_no': array([61, 26]), 'direction': array([1, 1]), 'rate': None, 'duration': None, 'data_aggregation_method': None, 'miscellaneous': None}, 'other': None}
```
The dictionary includes all the attributes, initialized or not. To access the `fni` as a normalized dictionary (
no embedded dictionaries):

```pycon
>>> fni.to_normalized_dict()
{'file_number': 1, 'sample_name': 'ZnO1', 'lasers.name': 'Matisse', 'lasers.wfe': <Quantity(737.8, 'nanometer')>, 'lasers.wavelength_vacuum': <Quantity(369.001625, 'nanometer')>, 'lasers.wavelength_air': <Quantity(368.9, 'nanometer')>, 'lasers.frequency': <Quantity(812442.109, 'gigahertz')>, 'lasers.energy': <Quantity(3.35999058, 'electron_volt')>, 'lasers.power': <Quantity(10.0, 'nanowatt')>, 'lasers.order': 2, 'lasers.medium': 'Air', 'lasers.miscellaneous': None, 'rf_sources': None, 'magnetic_field': <Quantity(5.0, 'tesla')>, 'temperature': <Quantity(120.0, 'millikelvin')>, 'excitation_path_optics': None, 'collection_path_optics.half_waveplate_angle': <Quantity(45.0, 'degree')>, 'collection_path_optics.quarter_waveplate_angle': None, 'collection_path_optics.polarizer': None, 'collection_path_optics.pinhole': <Quantity(40.0, 'micrometer')>, 'collection_path_optics.filters': ['LP380', 'BP370'], 'collection_path_optics.miscellaneous': None, 'exc_and_col_path_optics': None, 'rf_lines': None, 'measurement_type': 'ConfocalScanPL', 'miscellaneous': None, 'spot.start': <Quantity([-2.  3.], 'micrometer')>, 'spot.stop': <Quantity([4. 8.], 'micrometer')>, 'spot.step': <Quantity([0.1 0.2], 'micrometer')>, 'spot.step_no': array([61, 26]), 'spot.direction': array([1, 1]), 'spot.rate': None, 'spot.duration': None, 'spot.data_aggregation_method': None, 'spot.miscellaneous': None, 'other': None}
```
In the normalized dictionary the attributes of sub-dataclasses are separated by `.`.


### FilenameManager

The `FilenameManager` can provide easy access to all these variables, with the added benefit of managing multiple 
filenames at the same time. You can initialize the manager in three ways:

- Directly, with a filename or a list of filenames:
  ```pycon
  >>> from qdl_zno_analysis.filename_utils.filename_manager import FilenameManager
  >>> filenames = ['001_Msc~Example1a_Tmp~1p2.csv', '001_Msc~Example1b_Tmp~1p2.txt', '002_Msc~Example2_Tmp~1p2.csv']
  >>> folder = 'some_folder'
  >>> fnm = FilenameManager(filenames, folder)
  ```
- By providing file numbers and file extensions:
  ```pycon
  >>> from qdl_zno_analysis.filename_utils.filename_manager import FilenameManager
  >>> folder = 'some_folder'
  >>> fnm = FilenameManager.from_file_numbers([1,2], ['csv'. 'txt'], folder)
  ```
  This method searches the provided folder (defaults to '.'). 
  It will find all files with file numbers of 1 or 2, even multiple identical files, as long as they have any 
  of the file extensions in the filetype list. If the filetype list is empty, all files with the corresponding file
  number will be returned.
- By providing a string to be matched with `Path.glob()`:
  ```pycon
  >>> from qdl_zno_analysis.filename_utils.filename_manager import FilenameManager
  >>> folder = 'some_folder'
  >>> fnm = FilenameManager.from_matching_string('*Msc~Example*', folder)
  ```
  This method searches the provided folder (defaults to '.').
  It will find all files containing the string `'Msc~Example'` in the folder.

Of course, you can access all the filenames or just the valid filenames (once found in path - defaults to filenames if 
validation is not requested at initialization):
```pycon
>>> fnm.filenames
[WindowsPath('001_Msc~Example1a_Tmp~1p2.csv'), WindowsPath('001_Msc~Example1b_Tmp~1p2.txt'), WindowsPath('002_Msc~Example2_Tmp~1p2.csv')]
>>> fnm.valid_paths
[WindowsPath('001_Msc~Example1a_Tmp~1p2.csv'), WindowsPath('001_Msc~Example1b_Tmp~1p2.txt'), WindowsPath('002_Msc~Example2_Tmp~1p2.csv')]
```

From here, you can access the `FilenameInfo` objects directly if needed:
```pycon
>>> fnm.filename_info_list
[FilenameInfo(file_number=1, temperature=<Quantity(1.2, 'kelvin')>, miscellaneous='Example1a'), FilenameInfo(file_number=1, temperature=<Quantity(1.2, 'kelvin')>, miscellaneous='Example1b'), FilenameInfo(file_number=2, temperature=<Quantity(1.2, 'kelvin')>, miscellaneous='Example2')]
```

Or can access the aforementioned dictionaries:
```pycon
>>> fnm.filename_info_dicts
[{'file_number': 1, 'sample_name': None, 'lasers': None, 'rf_sources': None, 'magnetic_field': None, 'temperature': <Quantity(1.2, 'kelvin')>, 'excitation_path_optics': None, 'collection_path_optics': None, 'exc_and_col_path_optics': None, 'rf_lines': None, 'measurement_type': None, 'miscellaneous': 'Example1a', 'spot': None, 'other': None}, {'file_number': 1, 'sample_name': None, 'lasers': None, 'rf_sources': None, 'magnetic_field': None, 'temperature': <Quantity(1.2, 'kelvin')>, 'excitation_path_optics': None, 'collection_path_optics': None, 'exc_and_col_path_optics': None, 'rf_lines': None, 'measurement_type': None, 'miscellaneous': 'Example1b', 'spot': None, 'other': None}, {'file_number': 2, 'sample_name': None, 'lasers': None, 'rf_sources': None, 'magnetic_field': None, 'temperature': <Quantity(1.2, 'kelvin')>, 'excitation_path_optics': None, 'collection_path_optics': None, 'exc_and_col_path_optics': None, 'rf_lines': None, 'measurement_type': None, 'miscellaneous': 'Example2', 'spot': None, 'other': None}]
>>> fnm.filename_info_norm_dicts  # in this case they are the same, since there are no embedded `Info` objects
[{'file_number': 1, 'sample_name': None, 'lasers': None, 'rf_sources': None, 'magnetic_field': None, 'temperature': <Quantity(1.2, 'kelvin')>, 'excitation_path_optics': None, 'collection_path_optics': None, 'exc_and_col_path_optics': None, 'rf_lines': None, 'measurement_type': None, 'miscellaneous': 'Example1a', 'spot': None, 'other': None}, {'file_number': 1, 'sample_name': None, 'lasers': None, 'rf_sources': None, 'magnetic_field': None, 'temperature': <Quantity(1.2, 'kelvin')>, 'excitation_path_optics': None, 'collection_path_optics': None, 'exc_and_col_path_optics': None, 'rf_lines': None, 'measurement_type': None, 'miscellaneous': 'Example1b', 'spot': None, 'other': None}, {'file_number': 2, 'sample_name': None, 'lasers': None, 'rf_sources': None, 'magnetic_field': None, 'temperature': <Quantity(1.2, 'kelvin')>, 'excitation_path_optics': None, 'collection_path_optics': None, 'exc_and_col_path_optics': None, 'rf_lines': None, 'measurement_type': None, 'miscellaneous': 'Example2', 'spot': None, 'other': None}]
```

You can also find all the (non)-repeating values between filenames:
```pycon
>>> fnm.changing_filename_info_dict
{'file_number': [1, 1, 2], 'miscellaneous': ['Example1a', 'Example1b', 'Example2']}
>>> fnm.non_changing_filename_info_dict
{'sample_name': None, 'lasers': None, 'rf_sources': None, 'magnetic_field': None, 'temperature': <Quantity(1.2, 'kelvin')>, 'excitation_path_optics': None, 'collection_path_optics': None, 'exc_and_col_path_optics': None, 'rf_lines': None, 'measurement_type': None, 'spot': None, 'other': None}
```

Finally, you can find available filetypes, filenames by filetypes and available file numbers:
```pycon
>>> fnm.available_filetypes
['csv', 'txt']
>>> fnm.filenames_by_filetype
{'csv': [WindowsPath('001_Msc~Example1a_Tmp~1p2.csv'), WindowsPath('002_Msc~Example2_Tmp~1p2.csv')], 'txt': [WindowsPath('001_Msc~Example1b_Tmp~1p2.txt')]}
>>> fnm.available_file_numbers
[1, 2]
```

With all of the above, I hope that handling filename metadata will become simple and easy!
The `FilenameManager` class will be thoroughly utilized internally to ease the initialization of data-holding classes
in the rest of this package.

