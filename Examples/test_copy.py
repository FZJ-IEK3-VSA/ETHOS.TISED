import pandas as pd
import numpy as np
from ethos_tised import SolarModel

hourly_irrad_m = np.genfromtxt(r'C:\Users\o.omoyele\Desktop\Australia\ND_Model_Mean_QC\Validation_Sensitivity\BSRN\TOR\hourly_irrad_m_modified_2020.csv',delimiter=',')

synthetic = SolarModel(Lat= 58.254, Lon=26.462, Altitude=70, date=2008, data=hourly_irrad_m)
synthetic.to_csv(r'C:\Users\o.omoyele\Desktop\Ola\Software\tised\ethos_tised\data\synthetic_data.csv')
synthetic.describe()
