import numpy as np
from ethos_tised import SolarModel

hourly_irrad_m = np.genfromtxt(
    r"C:\Users\o.omoyele\Desktop\Australia\ND_Model_Mean_QC\Validation_new_new\Berlin\Berlin18\hourly_irrad_m_modified_2018.csv",
    delimiter=",",
)

synthetic = SolarModel(Lat=52.455778, Lon=13.523917, date=2018, data=hourly_irrad_m)
# synthetic.to_csv(r'C:\Users\o.omoyele\Desktop\Ola\Software\tised\ethos_tised\data\synthetic_data.csv')
synthetic.describe()
