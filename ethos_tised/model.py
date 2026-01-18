import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pvlib
from pvlib.location import lookup_altitude
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import root_mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from scipy.optimize import minimize
from scipy.stats import ks_2samp
from scipy.interpolate import interp1d
from timezonefinder import TimezoneFinder
import importlib.resources as pkg_resources
from kgcpy import lookupCZ
from datetime import datetime
import pytz
import ethos_tised.data



# -----------------------------
# Input functions
# -----------------------------
'''
Look up the Köppen-Geiger climate zone for the given latitude and longitude,
then map it to the target climate zone categories used in this model.
'''
def map_climate_zone(lat, lon):
    '''Map Köppen-Geiger code to target climate zone category.'''
    kgc = lookupCZ(lat, lon)
    mapping = {
        'Af': 'Aw', 'Am': 'Aw', 'As': 'Aw', 'Aw': 'Aw',
        'BSh': 'BSh', 'BSk': 'BSk', 'BWh': 'BWh', 'BWk': 'BWh',
        'Cfa': 'Cfa', 'Cfb': 'Cfb', 'Cfc': 'Cfb',
        'Csa': 'Csa', 'Csb': 'Csb', 'Csc': 'Csb',
        'Cwa': 'Csb', 'Cwb': 'Csb', 'Cwc': 'Csb',
        'Dfa': 'Others', 'Dfb': 'Others', 'Dfc': 'Others', 'Dfd': 'Others',
        'Dsa': 'Others', 'Dsb': 'Others', 'Dsc': 'Others', 'Dsd': 'Others',
        'Dwa': 'Others', 'Dwb': 'Others', 'Dwc': 'Others', 'Dwd': 'Others',
        'EF': 'Others', 'ET': 'Others'
    }

    mapped_zone = mapping.get(kgc, 'Others')

    # Print both the detected and mapped zones
    print(f'Detected Köppen-Geiger Zone: {kgc}')
    print(f'Mapped Climate Zone: {mapped_zone}')

    return mapped_zone


'''
Input data () loading based on the mapped Köppen-Geiger climate zone.
The files are non-dimensional irradiance against time profiles as minutal_database_ghi - 
and the daily parameters
'''
def load_data_for_zone(mapped_zone):
    """Load irradiance data based on climate zone result."""
    with pkg_resources.path(ethos_tised.data, f"{mapped_zone}/input_knn.csv") as hourly_db_path:
        hourly_database_ghi = np.genfromtxt(hourly_db_path, delimiter=',')

    with pkg_resources.path(ethos_tised.data, f"{mapped_zone}/minutal_new.csv") as minutal_db_path:
        minutal_database_ghi = np.genfromtxt(minutal_db_path, delimiter=',')

    return hourly_database_ghi, minutal_database_ghi



# -----------------------------
# Main Model Class
# -----------------------------

class SolarModel:
    def __new__(cls, Lat, Lon, Altitude=None, date=None, data=None):
        '''Return synthetic irradiance DataFrame directly upon instantiation.'''
        self = super(SolarModel, cls).__new__(cls)
        self.__init__(Lat, Lon, Altitude, date, data)
        return self.synthetic

    def __init__(self, Lat, Lon, Altitude, date, data):
        '''Initialize the SolarModel and run the full pipeline automatically.'''

        # Basic inputs
        self.lat = Lat
        self.lon = Lon
        self.date = date
        self.Altitude = Altitude
        self.hourly_irrad_m = data

        #self.structure_hourly_input()
        self.handle_missing_hourly_data()

    '''
    def structure_hourly_input(self):
        """
        Convert 1D GHI array into structured 2D array:
        [day, hour, ghi]
        """

        ghi = np.asarray(self.hourly_irrad_m).flatten()

        self.days = int(len(ghi) / 24)

        day_col = np.repeat(np.arange(1, self.days + 1), 24)
        hour_col = np.tile(np.arange(0, 24), self.days)

        self.hourly_irrad_m = np.column_stack((day_col, hour_col, ghi))

        print(f"Structured hourly data shape: {self.hourly_irrad_m.shape}")
    '''



    def handle_missing_hourly_data(self):
        '''
        Check for missing values in the hourly measured irradiance data.
        If missing values exist, perform KNN imputation to fill them.
        '''
        if len(self.hourly_irrad_m) not in (8760, 8784):
            raise ValueError('Please restructure the data: hourly GHI length must be 8760 for ordinary year or 8784 for leap year.')

        if np.isnan(self.hourly_irrad_m).any():
            print('Missing values detected in hourly GHI data. Applying KNN imputation...')
            # Use KNNImputer from sklearn
            imputer = KNNImputer(n_neighbors=3)
            self.hourly_irrad_m = imputer.fit_transform(self.hourly_irrad_m)
            print('Missing values have been imputed.')
        else:
            ghi = np.asarray(self.hourly_irrad_m).flatten()
            self.days = int(len(ghi) / 24)
            day_col = np.repeat(np.arange(1, self.days + 1), 24)
            hour_col = np.tile(np.arange(0, 24), self.days)

            self.hourly_irrad_m = np.column_stack((day_col, hour_col, ghi))
            print(f'Structured hourly data shape: {self.hourly_irrad_m.shape}')
            print('The provided hourly GHI data has no missing values. Proceeding with original data.')


        # Resolve altitude for location if not provided
        if self.Altitude is None:
            try:
                self.altitude = lookup_altitude(self.lat, self.lon)
            except Exception:
                # Safe fallback if lookup fails
                self.altitude = 0.0
        else:
            self.altitude = self.Altitude
        #print(f'Using altitude: {self.altitude} meters')



        # time parameters constants
        '''
        Points per day = Number of hours per day = 24 (hourly data)
        Minutes per day = Number of minutes per day = 60*24 = 1440 (minutal data)
        Days = Total number of days in the input measured data (hourly_irrad_m)
                The number of days can be 365 or 366 depending on whether it's a leap year or not.
                The days is calculated based on the length of the input data divided by points per day (24).
        NoOfHrs = Total number of hours in the input measured data (hourly_irrad_m) = Days * Points per day
        NoOfMins = Total number of minutes in the input measured data (hourly_irrad_m) = Days * Minutes per day
        '''
        self.points_per_day = 24
        self.minutes_per_day = 1440
        self.days = int(len(self.hourly_irrad_m) / self.points_per_day)
        self.noOfHrs = self.days * self.points_per_day
        self.noOfMins = self.days * self.minutes_per_day

        # timezone input (must be created before pvlib calls)
        self.timezone_str, self.start, self.end = self._get_timezone_and_range()
        print(f'Timezone: {self.timezone_str}')
        print(f'Start: {self.start}')
        print(f'End: {self.end}')

        # climate zone detection and loading of reference databases
        self.zone = map_climate_zone(self.lat, self.lon)
        self.hourly_database_ghi, self.minutal_database_ghi = load_data_for_zone(self.zone)

        # run pipeline
        self._run_pipeline()



    # -------------------------
    # Obtaining timezone and time range
    # -------------------------
    def _get_timezone_and_range(self):
        '''Find timezone from lat/lon and create timezone-aware start/end timestamps.'''
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lat=self.lat, lng=self.lon)
        if timezone_str is None:
            timezone_str = 'UTC'

        # timezone-aware pandas timestamps
        start = pd.Timestamp(f'{self.date}-01-01 00:00:00', tz=timezone_str)
        end = pd.Timestamp(f'{self.date}-12-31 23:59:00', tz=timezone_str)
        return timezone_str, start, end



    # -------------------------
    # Running the full pipeline
    # -------------------------
    def _run_pipeline(self):
        # Clear-sky irradiance (hourly and minutal)
        self.get_clear_sky_irradiance_hr()
        self.get_clear_sky_irradiance_min()

        # angles and indicators
        self.get_solar_angles()
        self.compute_clear_sky_irradiance()
        self.compute_variability_indices()
        self.compute_morning_fraction()
        self.compute_normalized_ic_cdf()
        self.get_minute_resolution_solar_angles()
        self.create_positive_alpha_mask()

        # KNN, upscaling and final synthesis
        self.train_knn(self.hourly_database_ghi)
        similar_days_ghi = self.predict_knn(self.hourly_database_ghi)
        self.upscale_to_minute_resolution(similar_days_ghi)
        self.synthetic = self.generate_synthetic_data()




    # -------------------------
    # Methods
    # -------------------------
    '''
    Using the pvlib library to get the extraterrestrial irradiance data from the CAMS database for both hourly and minutal resolutions.
    Users must register for access to the CAMS database and provide the registered email as an input prompt.
    '''
    # getting hourly extratterrestrial irradiance from CAMS
    def get_clear_sky_irradiance_hr(self):
        self.hourly_cs, _ = pvlib.iotools.get_cams(
            latitude=self.lat,
            longitude=self.lon,
            start=self.start,
            end=self.end,
            email='jo.omoyele@gmail.com',
            identifier='mcclear',
            altitude=self.altitude,
            time_step="1h",
            time_ref='TST',
            verbose=False,
            integrated=False,
            map_variables=True,
            url='api.soda-solardata.com',
            timeout=600
        )
        # convert to 1D array of length noOfHrs
        self.hourly_irrad_cs_ghi = self.hourly_cs[['ghi_extra']].reset_index(drop=True).to_numpy().reshape(self.noOfHrs)


    # getting minute extraterrestrial irradiance from CAMS
    def get_clear_sky_irradiance_min(self):
        self.minutal_cs, _ = pvlib.iotools.get_cams(
            latitude=self.lat,
            longitude=self.lon,
            start=self.start,
            end=self.end,
            email='jo.omoyele@gmail.com',
            identifier='mcclear',
            altitude=self.altitude,
            time_step="1min",
            time_ref='TST',
            verbose=False,
            integrated=False,
            map_variables=True,
            url='api.soda-solardata.com',
            timeout=600
        )
        # convert to 1D array of length noOfMins
        self.min_irrad_cs_ghi = self.minutal_cs[['ghi_extra']].reset_index(drop=True).to_numpy().reshape(self.noOfMins)


    # Training KNN model
    def train_knn(self, hourly_database_ghi):
        self.neigh_ghi = KNeighborsClassifier(n_neighbors=1)
        self.neigh_ghi.fit(self.hourly_database_ghi[:, [1, 2, 3, 4, 5]], self.hourly_database_ghi[:, 0])

    # getting solar altitude angle (alpha) and hour angle (omega) in degrees
    # these angles are needed to create masks for daytime and before-noon calculations


    def get_solar_angles(self):
        date_range = pd.date_range(start=self.start, end=self.end, freq="1h")
        solar_position = pvlib.solarposition.get_solarposition(
            date_range, latitude=self.lat, longitude=self.lon, altitude=self.altitude, pressure=1013.25
        )
        self.alpha_deg = solar_position['elevation'].values
        self.EOT = solar_position['equation_of_time'].values
        self.omega_deg = pvlib.solarposition.hour_angle(date_range, longitude=self.lon, equation_of_time=self.EOT)


    def compute_clear_sky_irradiance(self):
        '''
        Calculation of the daily clearness index, Kt, 
        Kt is based on the ratio of the daily average measured global horizontal irradiance to the daily average clear-sky global horizontal irradiance.
        '''
        #creating mask for when the altitude angle is less than 0 (or is negative)
        neg_alpha_deg_mask = (self.alpha_deg <= 0) #boolean mask that keeps negative altitude angles (in degrees)

        #creating a 2d (daily) array of the mask
        neg_alpha_deg_mask_2d = np.copy(neg_alpha_deg_mask)
        neg_alpha_deg_mask_2d.shape = (self.days,self.points_per_day) #transformation of array to 2D from 1D

        #for before-noon mask for morning fraction
        hourly_irrad_cs_ghi_2d = np.copy(self.hourly_irrad_cs_ghi)
        hourly_irrad_cs_ghi_2d.shape = (self.days,self.points_per_day)
        hourly_irrad_cs_ghi_2d[neg_alpha_deg_mask_2d] = 0 #masked to 0 using negative altitude angles mask

        hourly_irrad_m_ghi_2d = np.copy(self.hourly_irrad_m[:,2])
        hourly_irrad_m_ghi_2d.shape = (self.days,self.points_per_day)
        hourly_irrad_m_ghi_2d[neg_alpha_deg_mask_2d] = 0 #masked to 0 using negative altitude angles mask

        Kt = hourly_irrad_m_ghi_2d.mean(1) / hourly_irrad_cs_ghi_2d.mean(1) #calculation for the daily direct fraction index Kt by summing along axis 1 (columns) and then dividing the sums element wise
        self.Kt = Kt
        self.hourly_irrad_m_ghi_2d = hourly_irrad_m_ghi_2d

        # New Kt calculation method
        data_A = self.hourly_cs['ghi_extra'].values.reshape((self.days, self.points_per_day))
        #data_A
        data_B = self.hourly_irrad_m[:,2].reshape((self.days, self.points_per_day))
        #data_B

        data_A1 = pd.DataFrame(data_A)
        data_B1 = pd.DataFrame(data_B)

        Kt_new = data_B1.mean(1) / data_A1.mean(1)
        self.Kt_new = Kt_new


    def compute_variability_indices(self):
        '''
        Calculation of the Variability Index 1 (VI1) and the Normalized Variability Index 1 (NVI1).
        VI1 is based on the ratio of the sum of the square root of the squared differences between consecutive hourly measured global horizontal irradiance values 
                to the same sum for the clear-sky global horizontal irradiance values.
        NVI1 is based on the ratio of the sum of the square root of the squared differences between consecutive hourly measured global horizontal irradiance values 
                to the same sum for the normalized (sorted descending) measured global horizontal irradiance values.
        '''

        data_A = self.hourly_cs['ghi_extra'].values.reshape((self.days, self.points_per_day))
        #data_A
        data_B = self.hourly_irrad_m[:,2].reshape((self.days, self.points_per_day))
        #data_B

        data_A1 = pd.DataFrame(data_A).transpose()
        data_B1 = pd.DataFrame(data_B).transpose()
        data_A1 = data_A1.values
        data_B1 = data_B1.values

        data_C1 = np.sort(data_B, axis=1)[:, ::-1]
        data_C1 = pd.DataFrame(data_C1)
        data_C1 = data_C1.transpose().values


        #data_A1
        a_values = np.zeros(data_A1.shape[1])
        # Loop over each column
        for i in range(data_A1.shape[1]):
            # Calculate daily differences for the current column
            daily_differences_a_values = np.diff(data_A1[:, i])
            
            # Compute the value of a for the current column
            a_values[i] = np.power(np.power(daily_differences_a_values, 2) + 1, 1/2).sum()
        irradiance_cs = pd.DataFrame({'days': a_values})
        #irradiance_cs

        b_values = np.zeros(data_B1.shape[1])
        # Loop over each column
        for i in range(data_B1.shape[1]):
            # Calculate daily differences for the current column
            daily_differences_b_values = np.diff(data_B1[:, i])
            
            # Compute the value of a for the current column
            b_values[i] = np.power(np.power(daily_differences_b_values, 2) + 1, 1/2).sum()

        irradiance = pd.DataFrame({'days': b_values})
        #irradiance

        c_values = np.zeros(data_C1.shape[1])
        # Loop over each column
        for i in range(data_C1.shape[1]):
            # Calculate daily differences for the current column
            daily_differences_c_values = np.diff(data_C1[:, i])
            
            # Compute the value of a for the current column
            c_values[i] = np.power(np.power(daily_differences_c_values, 2) + 1, 1/2).sum()

        irradiance_norm = pd.DataFrame({'days': c_values})
        #irradiance

        VI1 = irradiance['days'] / irradiance_cs['days']
        NVI1 = irradiance['days'] / irradiance_norm['days']
        self.VI1 = VI1
        self.NVI1 = NVI1


    def compute_morning_fraction(self):
        '''
        calculation of the morning fraction, Fm,
        Fm is based on the ratio of the sum of the hourly measured global horizontal irradiance values before noon 
                to the sum of the hourly measured global horizontal irradiance values for the whole day.
        '''
        before_noon_mask = (self.omega_deg<0) #boolean mask for before-noon solar hour angles in radians
        after_noon_mask = np.logical_not(before_noon_mask) #mask for afternoon - opposite of before noon mask
        after_noon_mask_2d = np.copy(after_noon_mask) #new copy for 2D transformation
        after_noon_mask_2d.shape = (self.days,self.points_per_day) #2D transformation


        hourly_irrad_m_after_noon_ghi_2d = np.copy(self.hourly_irrad_m_ghi_2d) #copy of measured irradiance array
        hourly_irrad_m_after_noon_ghi_2d[after_noon_mask_2d] = 0 #setting afternoon values to 0 - 

        Fm_ghi = hourly_irrad_m_after_noon_ghi_2d.sum(1) / self.hourly_irrad_m_ghi_2d.sum(1) #sum along the second axis and dividing element-wise for Fm
        Fm_ghi = np.nan_to_num(Fm_ghi) #if division leads to NaN, replace with 0
        self.Fm_ghi = Fm_ghi


    def compute_normalized_ic_cdf(self):
        '''
        calculation of the normalized integrated cumulative distribution function (ICCDF),
        ICCDF is based on the area under the complementary cumulative distribution function (CCDF).
        '''
        data = self.hourly_irrad_m[:, 2]
        normalized_ic_cdf_values = []
        for day in range(self.days):
            daily_data = data[day * self.points_per_day:(day + 1) * self.points_per_day]
            differenced_data = np.diff(daily_data)
            sorted_diff = np.sort(differenced_data)
            n = len(sorted_diff)
            cdf = np.arange(1, n + 1) / n
            ccdf = 1 - cdf
            ic_cdf = np.trapezoid(ccdf, sorted_diff)
            diff_range = sorted_diff.max() - sorted_diff.min()
            normalized_ic_cdf = ic_cdf / diff_range if diff_range != 0 else 0
            normalized_ic_cdf_values.append(normalized_ic_cdf)
        self.normalized_ic_cdf = normalized_ic_cdf_values


    def get_minute_resolution_solar_angles(self):
        '''
        getting solar altitude angle (alpha) in degrees at minutal resolution
        these angles are needed to create masks for daytime at minutal resolution
        '''
        date_range = pd.date_range(start=self.start, end=self.end, freq="1min")
        date_range_utc = date_range.tz_convert('UTC')
        solar_position = pvlib.solarposition.get_solarposition(
            date_range_utc, latitude=self.lat, longitude=self.lon, altitude=self.altitude, pressure=1013.25
        )
        self.alpha_deg_min = solar_position['elevation'].values


    def create_positive_alpha_mask(self):
        '''
        getting the needed mask - where the solar altitude angle (alpha) is positive at minutal resolution
        '''
        self.pos_alpha_deg_mask_min = self.alpha_deg_min > 0
        self.pos_alpha_deg_mask_min_2d = self.pos_alpha_deg_mask_min.reshape((self.days, self.minutes_per_day))


    def predict_knn(self, neigh_ghi):
        '''
        predicting similar days from the trained KNN model using the calculated daily indicators
        '''
        calculated_indicators_ghi = np.zeros((self.days,5))
        calculated_indicators_ghi[:,0] = self.Kt_new
        calculated_indicators_ghi[:,1] = self.VI1
        calculated_indicators_ghi[:,2] = self.Fm_ghi
        calculated_indicators_ghi[:,3] = self.NVI1
        calculated_indicators_ghi[:,4] = self.normalized_ic_cdf

        similar_days_ghi = self.neigh_ghi.predict(calculated_indicators_ghi) #predicting the class labels for similar days from calculated indices
        #print(similar_days_ghi.shape)

        return similar_days_ghi


    def upscale_to_minute_resolution(self, similar_days_ghi):
        '''
        Upscaling the predicted similar days from hourly to minutal resolution using the minutal database.
        The upscaling is done by sampling the minutal data for each similar day based on the time periods where the solar altitude angle is positive (daytime).
        The sampled minutal data is then used to create the synthetic minutal irradiance data for the entire year.
        1. For each day in the year, get the corresponding similar day from the predicted similar days.
        2. For each similar day, extract the minutal data from the minutal database.
        3. Create indices to sample the minutal data based on the time periods where the solar altitude angle is positive.
        4. Fill the synthetic day data with the sampled values from the minutal data for each similar day.
        5. Append the synthetic minutal data for each day to create the final synthetic minutal irradiance data for the entire year.
        6. Store the final synthetic minutal irradiance data.
        '''
        min_syn_kt = []
        syn_day = np.zeros((self.minutes_per_day,2)) #new array with daily data at minutal resolution represented by zeros

        for i in range(0, self.days): #iterating for the year across each day
            ups_day_ghi = self.minutal_database_ghi[np.where(self.minutal_database_ghi[:,0] == similar_days_ghi[i])] #gets the data from minutal dataset for each similar day -- in other words, get 1000 rows for each similar day
            indices = np.around(1000 * np.arange(1,np.sum(self.pos_alpha_deg_mask_min_2d[i])+1)/np.sum(self.pos_alpha_deg_mask_min_2d[i])) #here we generate indices to later sample the ups_day data -- the indices generated are generated as a fraction of the total time the sum is up and is therefore then multiplied by 1000 to scale it up
            syn_day[self.pos_alpha_deg_mask_min_2d[i]] = ups_day_ghi[indices.astype(int)-1][:,1:3] #fills the synthetic day data with the sampled values from the ups_day for each of the created indices (as integer) in each similar day -- so for each similar day, only those rows of the 1000 rows are picked for Kb and Kt which are indexed using indices
            min_syn_kt = np.append(min_syn_kt, syn_day[:,1]) #appends the synthetic solar irradiance data for the day to the min_syn_kt array

        self.min_syn_kt = min_syn_kt
        #print(ups_day_ghi.shape)


    def generate_synthetic_data(self):
        '''
        Optimizing the synthetic data by minimizing the percentage difference between the daily sums of the measured and synthetic data using a daily factor, k.
        '''

        data_B = self.hourly_irrad_m[:,2].reshape((self.days, self.points_per_day))
        hourly_irrad_m_new = pd.DataFrame(data_B).sum(1)
        syn_min_ghi_test = (self.min_syn_kt * self.min_irrad_cs_ghi ).reshape(self.days, self.minutes_per_day)
        #syn_min_ghi_test = pd.DataFrame(syn_min_ghi_test)
        syn_min_ghi_test = pd.DataFrame(syn_min_ghi_test).sum(1)/60


        # Function to optimize a daily factor, k, to minimize the difference in the daily sum between the measured and the synthetic data.
        def objective(k, i):
            percentage_change = (hourly_irrad_m_new[i] - (k * syn_min_ghi_test[i]))/hourly_irrad_m_new[i]
            return np.abs(percentage_change)

        # Array to store optimal k values
        optimal_k = np.zeros(self.days)

        # Optimize k for each day
        for i in range(self.days):
            result = minimize(objective, x0=0, args=(i,), bounds=[(0, 10)])
            optimal_k[i] = result.x[0]

        #print("Optimal k values for each day:", optimal_k)


        df_synthetic_non_optimized = (self.min_syn_kt * self.min_irrad_cs_ghi).reshape(self.days, self.minutes_per_day)
        #A = pd.DataFrame(A)
        df_synthetic_optimized = df_synthetic_non_optimized * optimal_k [:, np.newaxis]
        df_synthetic_optimized = pd.DataFrame(df_synthetic_optimized)

        df_synthetic = pd.melt(df_synthetic_optimized.transpose())
        df_synthetic = df_synthetic.rename(columns={'value': 'synthetic_ghi'})

        # Create 'day' column (1 to self.days, each repeated 1440 times)
        day_col = np.repeat(np.arange(1, self.days + 1), self.minutes_per_day)

        # Create 'date' column using minute-resolution timestamps
        date_range = pd.date_range(start=self.start, end=self.end, freq="1min")
        #date_range_utc = date_range.tz_convert('UTC')
        date_col = date_range[:len(df_synthetic)]  # ensure matching length

        # Combine everything into a final DataFrame
        df_syn = pd.DataFrame({
            'day': day_col,
            'date': date_col,
            'synthetic_ghi': df_synthetic['synthetic_ghi'].values
        })

        print(df_syn.describe())
        return df_syn
    
    # End of SolarModel class