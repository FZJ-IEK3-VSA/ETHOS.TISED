import pandas as pd
import numpy as np
from ethos_tised import SolarModel

hourly_irrad_m = np.genfromtxt(r'C:\Users\o.omoyele\Desktop\Ola\ND_Model\Data\New folder\Model\hourly_irrad_m_modified_2018.csv',delimiter=',')

synthetic = SolarModel(Lat= 52.455778, Lon=13.523917, Altitude=34, date=2018, data=hourly_irrad_m)
#synthetic.to_csv(r'C:\Users\o.omoyele\Desktop\Ola\Software\tised\ethos_tised\data\synthetic_data.csv')
synthetic.describe()
