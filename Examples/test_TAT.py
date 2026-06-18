import numpy as np

from ethos_tised import SolarModel

ROOT = Path(__file__).resolve().parents[1]

hourly_irrad_m = np.genfromtxt(
    ROOT / "Examples" / "hourly_data" / "TAT.csv",
    delimiter=",",
)

synthetic = SolarModel(
    Lat=36.0581, 
    Lon=140.1258, 
    date=2013, 
    data=hourly_irrad_m
)

#output_path = ROOT / "Examples" / "Results" / "synthetic_data_TAT.csv"
#synthetic.to_csv(output_path, index=False)