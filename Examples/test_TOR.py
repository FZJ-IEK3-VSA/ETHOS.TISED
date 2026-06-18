import numpy as np

from ethos_tised import SolarModel

ROOT = Path(__file__).resolve().parents[1]

hourly_irrad_m = np.genfromtxt(
    ROOT / "Examples" / "hourly_data" / "TOR.csv",
    delimiter=",",
)

synthetic = SolarModel(
    Lat=58.254, 
    Lon=26.462, 
    date=2008, 
    data=hourly_irrad_m
)

#output_path = ROOT / "Examples" / "Results" / "synthetic_data_TOR.csv"
#synthetic.to_csv(output_path, index=False)