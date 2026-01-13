import pandas as pd
import numpy as np
from ethos_tised import SolarModel

hourly_irrad_m = np.genfromtxt(r'C:\Users\o.omoyele\Desktop\Ola\Software\tised\ethos_tised\data\hourly_irrad_m_modified_new.csv',delimiter=',')

synthetic = SolarModel(Lat=45.5028249, Lon=9.1561092, date=2017, data=hourly_irrad_m)
#synthetic.to_csv(r'C:\Users\o.omoyele\Desktop\Ola\Software\tised\ethos_tised\data\synthetic_data.csv')
synthetic.describe()
