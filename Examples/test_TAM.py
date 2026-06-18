import numpy as np

from ethos_tised import SolarModel

ROOT = Path(__file__).resolve().parents[1]

hourly_irrad_m = np.genfromtxt(
    ROOT / "Examples" / "hourly_data" / "TAM.csv",
    delimiter=",",
)

synthetic = SolarModel(
    Lat=22.7903, 
    Lon=5.5292, 
    date=2009, 
    data=hourly_irrad_m
)

#output_path = ROOT / "Examples" / "Results" / "synthetic_data_TAM.csv"
#synthetic.to_csv(output_path, index=False)
