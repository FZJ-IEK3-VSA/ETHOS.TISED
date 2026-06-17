# ETHOS.TISED


TISED (TIme SEries Downscaler) is part of [ETHOS (Energy Transformation Pathway Optimization Suite)](https://go.fzj.de/ethos_suite) as a Python library for global downscaling of Global Horizontal Irradiance (GHI) from one-hour resolution to one-minute for energy system applications. The package utilizes the non-dimensionalization of solar irradiance and time with statistical parameters matching to increase the temporal resolution of GHI.

## Working Principle
* Collection and Preparation of Input Parameters
* Extraction of Defining Parameters from Low-Resolution Data
* Matching Algorithm
* Selection of High-Resolution Data
* Unpacking the 1 Minute Data

## Getting Started
The package is continually developed. However, for the use case, the ghi.ipynb file in the Example folder for several locations can be accessed. The complete database is uploaded on [zenodo](https://doi.org/10.5281/zenodo.15226264)


## Installation
It is recommended to install tised within its own environment. If you are not familiar with Python environments, please consider reading some [external documentation](https://realpython.com/python-virtual-environments-a-primer/). In the following, we assume you have a [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) or [conda](https://www.anaconda.com/) installation.  All conda and mamba commands are interchangeable.


ETHOS.TISED is installed from GitHub as code. For use, every time a new Köppen-Geiger climate class is called through data downscaling of any location, the package downloads the data for this climate class and caches it. Subsequent downscaling in the same climate class only uses the cached data.

To prevent any possible dependency conflicts, create a new environment using the following command:

	mamba create -n tised_env python pip

Activate an existing or the newly created environment afterward

	mamba activate tised_env

Alternatively, a virtual environment can be used

        python -m venv tised_env

Then activate (Command Prompt) - Windows

        tised_env\Scripts\activate

Activate (PowerShell) - Windows

        tised_env\Scripts\Activate.ps1
	
Then install via

        pip install -e .



### Python Example
Read the time series data set with numpy
```python
	hourly_data = np.genfromtxt("load.csv", delimiter=",")
```

Initialize the SolarModel from ethos.tised and define the latitude, longitude, date (year of the data), and the hourly_data, which has been read as a single column array. 

```python
	synthetic = SolarModel(Lat= 52.455778, 
        Lon=13.523917, 
        date=2018, 
        data=hourly_data
        )
```

The model assumes that the input hourly data-single column array-is complete without errors. However, for incomplete data, the user needs to use the sample of the 'hourly_data_missing' in the data folder before using the model. This way, the model performs KNN imputation methods for complete data imputation, then downscales.

Further examples can be found in the Examples folder.


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/FZJ-IEK3-VSA/ETHOS.TISED/blob/main/LICENSE) file for details.

Copyright (c) 2025 Olalekan Omoyele (FZJ ICE-2), Julian Belina (FZJ ICE-2), Maximilian Hoffmann (FZJ ICE-2), Jann Michael Weinand (FZJ ICE-2), Miguel Larrañeta (Universidad de Sevilla), Jochen Linßen (FZJ ICE-2), Detlef Stolten (FZJ ICE-2).

You should have received a copy of the MIT License along with this program.
If not, see https://opensource.org/licenses/MIT



## Citing and further reading

If you want to use ETHOS.TISED in a published work, **please kindly cite** our latest journal articles:
* Omoyele et al. (2026):\
[**ETHOS.TISED: A python package for temporal downscaling of solar irradiance**](https://doi.org/10.1016/j.softx.2026.102688)

* Omoyele et al. (2026):\
[**A High-Resolution Downscaling Approach for Solar Irradiance Using Statistical Parameter Matching**](https://doi.org/10.1016/j.renene.2025.124551) 


## About Us

We are the [Methodology laboratory](https://www.fz-juelich.de/en/ice/ice-2/research-1/integrated-scenarios/methodology-lab) department at the [Institute of Energy and Climate Research: Jülich Systems Analysis (ICE-2)](https://www.fz-juelich.de/en/ice/ice-2), belonging to the Forschungszentrum Jülich. Our interdisciplinary department's research focuses on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions, and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.

## Acknowledgements

This work is supported by the Helmholtz Association as part of the program “Energy System Design”.

<p float="left">
<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px"></a>
</p>
