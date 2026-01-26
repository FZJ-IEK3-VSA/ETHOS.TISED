import numpy as np

from ethos_tised import SolarModel

hourly_irrad_m = np.genfromtxt(
    r"C:\Users\o.omoyele\Desktop\Australia\ND_Model_Mean_QC\Validation_new_new\BSRN\TAM\hourly_irrad_m_modified_2019.csv",
    delimiter=",",
)

synthetic = SolarModel(Lat=22.7903, Lon=5.5292, date=2009, data=hourly_irrad_m)
# synthetic.to_csv(r'C:\Users\o.omoyele\Desktop\Ola\Software\tised\ethos_tised\data\synthetic_data.csv')
synthetic.describe()
