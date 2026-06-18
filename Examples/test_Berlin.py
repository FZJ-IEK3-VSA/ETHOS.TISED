from pathlib import Path

import numpy as np

from ethos_tised import SolarModel

ROOT = Path(__file__).resolve().parents[1]

hourly_irrad_m = np.genfromtxt(
    ROOT / "Examples" / "hourly_data" / "Berlin.csv",
    delimiter=",",
)

synthetic = SolarModel(
    Lat=52.455778,
    Lon=13.523917,
    date=2018,
    data=hourly_irrad_m,
)

#output_path = ROOT / "Examples" / "Results" / "synthetic_data_Berlin.csv"
#synthetic.to_csv(output_path, index=False)
