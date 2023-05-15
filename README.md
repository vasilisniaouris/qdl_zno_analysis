# Description 
The quantum defect laboratory (QDL) ZnO Data Analysis Toolkit is a python package created to aid and streamline 
the analysis spectral and temporal data collected on ZnO substrates (and other materials we research).

**Links**:
- [Documentation (GitHub Pages)](https://vasilisniaouris.github.io/qdl_zno_analysis/)
- [GitHub Repository](https://github.com/vasilisniaouris/qdl_zno_analysis)

# Installation
There's two ways to install the core package through GitHub.

1. To directly install it from GitHub, you can use the following command:
   ~~~shell
   pip install git+https://github.com/vasilisniaouris/qdl_zno_analysis.git
   ~~~
   which is equivalent to: 
   ~~~shell
   pip install "qdl_zno_analysis @ git+https://github.com/vasilisniaouris/qdl_zno_analysis.git"
   ~~~
2. Or you can clone the repository and then use pip to install the package.
   ~~~shell
   git clone https://github.com/vasilisniaouris/qdl_zno_analysis.git
   pip install qdl_zno_analysis
   ~~~

## Installation extras

To install all extra dependencies, append [all] to the end of the `pip install qdl_zno_analysis[all]` command.
For specific dependencies, check out the different headers in the [Dependencies](#dependencies) section.
~~~shell
pip install "qdl_zno_analysis[header1,header2,...] @ git+https://github.com/vasilisniaouris/qdl_zno_analysis.git"
~~~

If you want to install the developer's branch (dev) that has the newest features but may break more easily, 
append `@dev` at the end of the GitHub link:
~~~shell
pip install git+https://github.com/vasilisniaouris/qdl_zno_analysis.git@dev
~~~

If you want to install a specific release, you can append the version of your choice, e.g. `@v0.1.0`, 
to the GitHub link:
~~~shell
pip install git+https://github.com/vasilisniaouris/qdl_zno_analysis.git@v0.1.0
~~~

# Examples
Pending...

# Dependencies
The QDL ZnO Data Analysis Toolkit has the following core dependencies:

~~~text
"numpy>=1.24.2"
"pandas"
"pint~0.20.1"
"pint-pandas"
"scipy"
~~~

For extra functionality, you may want to install additional dependencies, with the command  
`pip install ...[header]`

| Header          | Dependencies                          |
|-----------------|---------------------------------------|
| `all`           | everything listed below               |
| `visualization` | `"matplotlib>=3.7.1"`                 |
| `spectroscopy`  | `"sif_parser", "spe2py", "xmltodict"` |
| `example_data`  | `"requests", "gdown"`                 |

To install all optional dependencies, use the header `all`: `pip install ...[all]`.


# License
The QDL ZnO Data Analysis Toolkit is released under the GNU GPL v3 license. See LICENSE for more information.
Find a copy of the GNU General Public License [here](https://www.gnu.org/licenses/gpl-3.0.html).

# Copyright
Copyright (C) 2023, Vasilis Niaouris

# Change Log
[Found here](./CHANGELOG.md).

# Credits
The QDL ZnO Data Analysis Toolkit is created by Vasilis Niaouris under the Quantum Defect Laboratory at the 
University of Washington. Eventually, it will be heavily influenced by the functions26 package. 
Once it reaches a similar level of functionality, it will be based on code written primarily by Vasilis Niaouris and 
Chris Zimmermann.  

# Contact
If you have any questions or comments, please feel free to contact me at vasilisniaouris*gmail.com.
