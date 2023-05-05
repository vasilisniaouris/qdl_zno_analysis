# Description 
The quantum defect laboratory (QDL) ZnO Data Analysis Toolkit is a python package created to aid and streamline 
the analysis spectral and temporal data collected on ZnO substrates (and other materials we research).

**Links**:
- [Documentation (GitHub Pages)](https://vasilisniaouris.github.io/qdl_zno_analysis/)
- [GitHub Repository](https://github.com/vasilisniaouris/qdl_zno_analysis)

# Installation
To install the package you can either
1.  directly install it from GitHub you can use the following command:
    ~~~shell
    pip install git+https://github.com/vasilisniaouris/qdl_zno_analysis.git
    ~~~
    This will install the latest version of the package from the master branch. 

    If you want to install a specific release, you can use the tag name instead of master, like this:
    ~~~shell
    pip install git+https://github.com/vasilisniaouris/qdl_zno_analysis.git@v0.1.0
    ~~~
    Replace v0.1.0 with the tag name of the release you want to install.

2. Or, Alternatively, you can clone the repository:

    ~~~shell
    git clone https://github.com/vasilisniaouris/qdl_zno_analysis.git
    ~~~
    
    Navigate to the cloned directory:
    ~~~shell
    cd qdl_zno_analysis
    ~~~
    
    And, finally, use pip to install the package:
    ~~~shell
    pip install .
    ~~~

And you are all done. The qdl_zno_analysis should be available to you as a python module.

# Examples
Pending...

# Dependencies
The QDL ZnO Data Analysis Toolkit requires the following dependencies:

~~~
"matplotlib>=3.7.1"
"numpy>=1.24.2"
"pandas"
"pint>=0.20.1"
"scipy"
~~~

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
