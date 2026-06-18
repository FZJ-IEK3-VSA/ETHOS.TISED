import numpy as np

from ethos_tised import SolarModel

ROOT = Path(__file__).resolve().parents[1]

hourly_irrad_m = np.genfromtxt(
    ROOT / "Examples" / "hourly_data" / "Milan.csv",
    delimiter=",",
)

synthetic = SolarModel(
    Lat=45.5028249, 
    Lon=9.1561092, 
    date=2017, 
    data=hourly_irrad_m
)
#output_path = ROOT / "Examples" / "Results" / "synthetic_data_Milan.csv"
#synthetic.to_csv(output_path, index=False)