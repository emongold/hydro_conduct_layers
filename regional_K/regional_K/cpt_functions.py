## cpt_functions.py contains the functions in regional_K to perform on CPT data
import pandas as pd
import geopandas as gpd
import os
import numpy as np
import utm
import datetime
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.warp import transform as warp_transform
import ruptures as rpt

def convert_raw(raw_folder, locations_file):
    ''' 
    Function convert_raw to take raw cpt data and convert it to a dictionary of dataframes
    Inputs:
    raw_folder: string, path to folder containing raw cpt data, in units of 
    locations_file: string, path to csv file containing location data for each cpt file
    Output:
    output_dict: dictionary of dataframes, each dataframe contains the processed cpt data for a single file
    '''
    output_dict = {}
    g = 9.81
    Pa = 0.101325  # MPa
    rho_w = 1  # Mg/m^3
    gamma_w = rho_w * g / 1000  # MPa
    for file_name in os.listdir(raw_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(raw_folder, file_name)      
            df_temp = pd.read_csv(file_path, delimiter=",", skiprows=2)
            df_temp = df_temp.dropna(axis='columns', how='all')
            df_temp.columns = ['Depth', 'qc', 'fs', 'u']

            # Filtering out invalid data
            df_temp = df_temp[-((df_temp['fs'] < 0) | (df_temp['qc'] < 0))]
            df_temp.reset_index(drop=True, inplace=True)

            # convert to units of MPa
            df_temp['fs'] = df_temp['fs'] / 10.44271713652
            df_temp['qc'] = df_temp['qc'] / 10.44271713652

            temp = pd.DataFrame(np.zeros(shape=(len(df_temp), 8)),
                                columns=['start', 'q_c', 'f_s', 'd', 'dz', 'u', 'gamma', 'R_f'])

            temp.loc[0,'q_c'] = df_temp.loc[0,'qc'] / 2
            temp.loc[0,'f_s'] = df_temp.loc[0,'fs'] / 2
            temp['u'] = df_temp['u']*0.00689476 ## convert psi to MPa
            temp.loc[0,'d'] = np.average([temp.loc[0,'start'], df_temp.loc[0,'Depth']])
            temp.loc[0,'dz'] = df_temp.loc[0,'Depth'] - temp.loc[0,'start']
            temp.loc[0,'R_f'] = 100 * temp.loc[0,'f_s'] / temp.loc[0,'q_c']

            for i in range(1, len(df_temp)):
                temp['start'][i] = df_temp['Depth'].iloc[i - 1]
                temp['f_s'][i] = np.average([df_temp['fs'].iloc[i], df_temp['fs'].iloc[i - 1]])
                temp['q_c'][i] = np.average([df_temp['qc'].iloc[i], df_temp['qc'].iloc[i - 1]])
                temp['d'][i] = np.average([temp['start'][i], df_temp['Depth'].iloc[i]])
                temp['dz'][i] = df_temp['Depth'].iloc[i] - temp['start'][i]
                temp['R_f'][i] = 100 * temp['f_s'][i] / temp['q_c'][i]

                if temp['R_f'][i] == 0:
                    if temp['q_c'][i] == 0:
                        temp['gamma'][i] = gamma_w * 1.236
                    else:
                        temp['gamma'][i] = gamma_w * (0.36 * np.log10(temp['q_c'][i] / Pa) + 1.236)
                elif temp['q_c'][i] == 0:
                    temp['gamma'][i] = gamma_w * (0.27 * (np.log10(temp['R_f'][i])) + 1.236)
                else:
                    temp['gamma'][i] = gamma_w * (0.27 * np.log10(temp['R_f'][i]) +
                                                  0.36 * np.log10(temp['q_c'][i] / Pa) + 1.236)  ## using Robertson and Cabal (2010), calculated in MPa
            temp['dsig_v'] = temp['dz'] * temp['gamma']

            dataset_key = file_name.split('.')[0]
            output_dict[dataset_key] = {
                'CPT_data': temp,
                'UTM-X': '',
                'UTM-Y': '',
                'Elev': np.nan,
                'Water depth': np.nan,
                'Lat': '',  
                'Lon': ''   
            }

    locations = pd.read_csv(locations_file, skiprows=1, header=None, names=['filename', 'Lat', 'Lon','wd']) #, dtype={'filename': str, 'Lat': float, 'Lon': float, 'wd': float})

    for dataset_key, cpt_data in output_dict.items():
        cpt_row = locations[locations['filename'] == dataset_key]
        if not cpt_row.empty:
            lat = cpt_row['Lat'].iloc[0]
            lon = cpt_row['Lon'].iloc[0]
            utm_x, utm_y, _, _ = utm.from_latlon(lat, lon)
            cpt_data['Lat'] = lat
            cpt_data['Lon'] = lon
            cpt_data['UTM-X'] = utm_x
            cpt_data['UTM-Y'] = utm_y
            cpt_data['Water depth'] = cpt_row['wd'].iloc[0]  # water depth in ft
    outputraw = output_dict
    return outputraw

def setup_cpt(datadir):
    ''' function setup_cpt to load USGS cpt data from folder and output a dictionary
    input: datadir is the directory path to the cpt data
    output: cpt is a dictionary with an np directory for each borehole set of cpt data
    '''
    d = {}
    names = []
    cpt = {}
    # Constants
    g = 9.81
    Pa = 0.101325  # MPa
    rho_w = 1  # Mg/m^3
    gamma_w = rho_w * g / 1000  # MPa

    for filename in filter(lambda x: x[-4:] == '.txt', os.listdir(datadir)):

        with open(os.path.join(datadir, filename)) as f:
            name = datadir + filename
            df_temp = pd.read_csv(name, delimiter="\s+", skiprows=17)
            df_temp = df_temp.dropna(axis='columns', how='all')
            df_temp.columns = ['Depth', 'Tip_Resistance', 'Sleeve_Friction', 'Inclination', 'Swave_travel_time']
            df_temp = df_temp[~((df_temp['Sleeve_Friction'] < 0) | (df_temp['Tip_Resistance'] < 0))]

            # df_temp = df_temp[df_temp['Depth'] <= 20]
            df_temp['Sleeve_Friction'] = df_temp['Sleeve_Friction'] / 1000  # convert to units of MPa

            # df_temp = df_temp.dropna(subset=['Depth'])
            df_temp = df_temp.reset_index(drop=True)  # Reset index to ensure alignment

            temp = pd.DataFrame(np.zeros(shape=(len(df_temp), 7)),
                                columns=['start', 'q_c', 'f_s', 'd', 'dz', 'gamma', 'R_f'])
            temp['q_c'] = df_temp['Tip_Resistance']
            temp['f_s'] = df_temp['Sleeve_Friction']
            temp['d'] = df_temp['Depth']
            
            # Calculate dz as the change in depth
            temp['dz'] = temp['d'].diff()
            
            # Calculate R_f and gamma
            temp['R_f'] = 100 * temp['f_s'] / temp['q_c']

            # Calculating soil unit weight from Robertson and Cabal (2010)
            temp['gamma'] = gamma_w * (0.27 * np.log10(temp['R_f'].replace(0, np.nan)) +
                                    0.36 * np.log10(temp['q_c'].replace(0, np.nan) / Pa) + 1.236)
            temp['gamma'] = temp['gamma'].fillna(gamma_w * (0.27 * np.log10(temp['R_f'].replace(0, np.nan)) + 1.236))
            temp['gamma'] = temp['gamma'].fillna(gamma_w * (0.36 * np.log10(temp['q_c'].replace(0, np.nan) / Pa) + 1.236))
            temp['gamma'] = temp['gamma'].fillna(gamma_w * 1.236)  # Fill gamma where R_f or q_c was 0

            temp['dsig_v'] = temp['dz'] * temp['gamma']

            key = list(dict(l.strip().rsplit(maxsplit=1) for l in open(name) if any(l.strip().startswith(i) for i in 'File name:')).values())[0]
            names.append(key)
            d[key] = dict(l.strip().rsplit('\t', maxsplit=1) for l in open(name) \
                            if (any(l.strip().startswith(i) for i in ('"UTM-X', '"UTM-Y', '"Elev', '"Water depth', 'Date')) and len(l.strip().rsplit('\t', maxsplit=1)) == 2))

            cpt[key] = {}
            cpt[key]['CPT_data'] = temp

            for i in d[key]:
                if i.startswith('"UTM-X'):
                    cpt[key]['UTM-X'] = int(d[key][i])
                elif i.startswith('"UTM-Y'):
                    cpt[key]['UTM-Y'] = int(d[key][i])
                elif i.startswith('"Elev'):
                    cpt[key]['Elev'] = float(d[key][i])
                elif i.startswith('"Water depth'):
                    # check if water depth is a number
                    if d[key][i].replace('.', '', 1).isdigit():
                        cpt[key]['Water depth'] = float(d[key][i])
                elif i.startswith('Date'):
                    cpt[key]['Date'] = datetime.datetime.strptime(d[key][i], '%m/%d/%Y')

            if 'Elev' not in cpt[key]:
                cpt[key]['Elev'] = np.nan
            if 'Water depth' not in cpt[key]:
                cpt[key]['Water depth'] = np.nan

    for i in range(len(names)):
        cpt[names[i]]['Lat'], cpt[names[i]]['Lon'] = utm.to_latlon(cpt[names[i]]['UTM-X'], cpt[names[i]]['UTM-Y'], 10,
                                                                    northern=True)
    return cpt

def soil_stress(cpt):
    """
    Return the sigma and sigma v based on CPTu data and calculate static pore water pressure (u_s).
    Parameters:
    - cpt: dict containing CPTu data for each borehole.
    Output:
    - Updated cpt dict with additional data for soil stress and calculated static pore water pressure (u_s).
    """
    # Constants
    g = 9.81
    rho_w = 1  # density of water in Mg/m^3
    gamma_w = rho_w * g / 1000  # in units of MPa

    for _, data in cpt.items():
        h = np.maximum(data['CPT_data']['d'] - data['Water depth']*0.3049, 0)  # water depth is in feet, multiply by 0.3049 to convert to meters
        u_s = gamma_w * h  # static pore water pressure in MPa
        sig_v = np.cumsum(data['CPT_data']['dsig_v'])
        sig_prime_v = sig_v - u_s  # effective vertical stress in MPa using static pore water pressure

        # Store the calculated sig_v and sig_prime_v in the cpt dict
        data['CPT_data']['sig_v'] = sig_v
        data['CPT_data']['sig_prime_v'] = sig_prime_v

        # Save the calculated u_s (static pore water pressure) in the CPTu dictionary
        data['CPT_data']['u_s'] = u_s
        ## correct the tip resistance qc to q_t
        data['CPT_data']['q_t'] = data['CPT_data']['q_c'] + data['CPT_data']['u_s']*(1-0.80) 

    return cpt

def solve_Ic(q_c, sig_v, sig_prime_v, f_s):
    '''solves for the soil behavior index, Ic using an interative n procedure as in Robertson (2009)
    input q_c tip resistance
    input sig_v vertical soil stress
    input sig_prime_v effective vertical soil stress
    output Ic soil behavior type index '''

    Pa = 0.101325
    n = np.full_like(sig_prime_v,1.0)  # initialize n array with same shape as sig_prime_v
    tol = 1e-3
    Q = ((q_c - sig_v) / Pa) * (Pa / sig_prime_v) ** n
    F = 100 * f_s / (q_c - sig_v)
    Q = np.maximum(1, Q)
    F = np.where(F < 0, np.maximum(1, F), F)
    Ic = np.sqrt((3.47 - np.log10(Q)) ** 2 + (1.22 + np.log10(F)) ** 2)
    n2 = 0.381*Ic + 0.05 * (sig_prime_v / Pa) - 0.15
    n2 = np.maximum(n2, 0.5)
    n2 = np.minimum(n2, 1.0)
    while any(abs(n-n2) > tol):
        Q = ((q_c - sig_v) / Pa) * (Pa / sig_prime_v) ** n2
        Q = np.maximum(1, Q)
        Ic = np.sqrt((3.47 - np.log10(Q)) ** 2 + (1.22 + np.log10(F)) ** 2)
        n = n2
        n2 = 0.381 * Ic + 0.05 * (sig_prime_v / Pa) - 0.15
        n2 = np.maximum(n2, 0.5)
        n2 = np.minimum(n2, 1.0)

    return Ic

def calculate_bq(u_a, u_s, qt, sigma_v0):
    return (u_a - u_s) / (qt - sigma_v0)

def calculate_qt(qt, sigma_v0, sigma_v0_prime):
    return (qt - sigma_v0) / sigma_v0_prime

def k_wang2011(cpt):
    """
    Calculate hydraulic conductivity using the Wang et al. (2011) method.
    Parameters:
    - cpt: dict containing CPTu data for each borehole.
    Returns:
    - cpt: updated dict with hydraulic conductivity values for each borehole.
    """
    # Constants
    Pa = 0.101325  # MPa (atmospheric pressure)
    cw = 0.00981  # Unit weight of water (MPa/m)
    cone_radius = 0.0218  # Calculated cone radius from given 15 cm² area in m
    U = 0.02  # Penetration rate in meters per second (2 cm/s) m/s
    # Calculate KD using the Wang 2011 approach
    def calculate_kd(bq, qt):
        """Embedded function to calculate dimensionless hydraulic conductivity index (KD)."""
        if bq * qt <= 0.45:
            return 0.062 / (bq * qt) ** 1.6
        else:
            return 0.044 / (bq * qt) ** 4.91
    for _, data in cpt.items():
        temp = data['CPT_data']
        
        # Calculate Bq, Qt, and k for each depth
        temp['Bq'] = temp.apply(lambda row: (row['u'] - row['u_s']) / (row['q_c'] - row['sig_v']), axis=1)
        temp['qc_sigv'] = temp.apply(lambda row:(row['q_c'] - row['sig_v']), axis=1)
        temp['u_us'] = temp.apply(lambda row:(row['u'] - row['u_s']), axis=1)
        temp['Qt'] = temp.apply(lambda row: (row['q_c'] - row['sig_v']) / row['sig_prime_v'], axis=1)
        
        # Calculate hydraulic conductivity k directly
        temp['test_kd'] = temp.apply(lambda row: (row['Bq']*row['Qt']), axis=1)
        temp['kd_wang2011'] = temp.apply(lambda row: (row['test_kd']**4.91),axis=1)
        temp['k_wang2011'] = temp.apply(lambda row: (calculate_kd(row['Bq'], row['Qt']) * cone_radius * cw * U) / 
                                            (3 * row['sig_prime_v']), axis=1)  ## units are m, MPa/m, m/s, MPa, and end in m/s
        
        # Store the results back into the cpt dictionary
        data['CPT_data'] = temp
    
    return cpt

def k_lee2008(cpt):
    """
    Calculate hydraulic conductivity using the method described by Lee et al. (2008).
    Parameters:
    - cpt: dict containing CPT data for each borehole
    Returns:
    - cpt: updated dict with hydraulic conductivity values for each borehole 
    """
    # Constants
    gamma_w = 0.00981  # Unit weight of water (MN/m^3)
    cone_radius = 0.035  # Example cone radius (m)
    U = 0.02  # Penetration rate in meters per second (2 cm/s)
    def calculate_kd(bq, qt):
        return 1 / (bq * qt)
    for _, data in cpt.items():
        temp = data['CPT_data']
        
        # Calculate Bq, Qt, KD, and k for each depth
        temp['Bq'] = temp.apply(lambda row: calculate_bq(row['u'], row['u_s'], row['q_c'], row['sig_v']), axis=1)
        temp['Qt'] = temp.apply(lambda row: calculate_qt(row['q_c'], row['sig_v'], row['sig_prime_v']), axis=1)
        
        # Calculate hydraulic conductivity k using KD and save as 'k_lee2008'
        temp['k_lee2008'] = temp.apply(lambda row: (calculate_kd(row['Bq'], row['Qt']) * cone_radius * gamma_w * U) / 
                                                (4 * row['sig_prime_v']), axis=1)  # units should be MN/m^3, m, m, MPa, and end in m/s
        # Store the results back into the cpt dictionary
        data['CPT_data'] = temp
    
    return cpt

def k_manassero1994(cpt):
    """
    Calculate hydraulic conductivity using the method described by Manassero et al. (1994).
    Parameters:
    - cpt: dict containing CPT data for each borehole
    Returns:
    - cpt: updated dict with hydraulic conductivity values for each borehole
    """
    # Constants for Manassero's empirical equation
    A = 2.61  # Empirical constant for Bq-qt averaging method
    B = -10.93  # Empirical constant for Bq-qt averaging method
    
    # Function to calculate Bk
    def calculate_Bk(qt, fs, delta_u):
        return (qt ** 2) / (100 * fs * delta_u)
    
    # Calculate hydraulic conductivity (k) based on Bk
    def calculate_k(Bk):
        return 10 ** (A * (Bk ** 0.5) + B)  # Hydraulic conductivity in m/s

    for _, data in cpt.items():
        temp = data['CPT_data']
        
        # Calculate Bk and hydraulic conductivity (k_manassero1994) for each depth
        temp['Bk'] = temp.apply(lambda row: calculate_Bk(row['q_c'], row['f_s'], row['u'] - row['u_s']), axis=1)
        temp['k_manassero1994'] = temp.apply(lambda row: calculate_k(row['Bk']), axis=1)
        
        # Store the results back into the cpt dictionary
        data['CPT_data'] = temp
    
    return cpt

def calc_SBT(cpt, pa=0.1013):
    '''
    function calc_SBT to determine soil behavior type from Robertson 2010
    inputs:
        cpt: dictionary of cpt data
        pa: atmospheric pressure in MPa (default = 0.1013)
    outputs:
        cpt: dictionary with SBT and SBT description added to each data set
    '''
    for _, data in cpt.items():
       
        # Calculate I_SBT using NumPy for logarithms and square root
        log_qc_pa = np.log10(data['CPT_data']['q_c'] / pa)
        log_Rf = np.log10(data['CPT_data']['R_f'])
        I_SBT = np.sqrt((3.47 - log_qc_pa)**2 + (log_Rf + 1.22)**2)
        
        # Determine SBT based on I_SBT thresholds to function for an array
        sbt = np.zeros_like(I_SBT)
        sbt_descr = np.empty_like(I_SBT, dtype=object)
        sbt[(I_SBT <= 1.31)] = 7
        sbt_descr[(I_SBT <= 1.31)] = "Dense sand to gravelly sand"
        sbt[(I_SBT > 1.31) & (I_SBT <= 2.05)] = 6
        sbt_descr[(I_SBT > 1.31) & (I_SBT <= 2.05)] = "Sands: clean sands to silty sands"
        sbt[(I_SBT > 2.05) & (I_SBT <= 2.60)] = 5
        sbt_descr[(I_SBT > 2.05) & (I_SBT <= 2.60)] = "Sand mixtures: silty sand to sandy silt"
        sbt[(I_SBT > 2.60) & (I_SBT <= 2.95)] = 4
        sbt_descr[(I_SBT > 2.60) & (I_SBT <= 2.95)] = "Silt mixtures: clayey silt & silty clay"
        sbt[(I_SBT > 2.95) & (I_SBT <= 3.60)] = 3
        sbt_descr[(I_SBT > 2.95) & (I_SBT <= 3.60)] = "Clays: clay to silty clay"
        sbt[(I_SBT > 3.60)] = 2
        sbt_descr[(I_SBT > 3.60)] = "Clay - organic soil"
        # Save the SBT and SBT description to the CPT data
        data['SBT'] = sbt
        data['SBT_descr'] = sbt_descr
    return

def k_from_Ic(Ic_values):
    '''
    function k_from_Ic to solve k from Ic values using Robertson 2010
    inputs:
        Ic_values: array of Ic values
    outputs:
        k: array of k values
    '''
    k = np.zeros_like(Ic_values)
    
    # Case 1: When 1.0 < Ic <= 3.27
    mask_case1 = (Ic_values > 1.0) & (Ic_values <= 3.27)
    k[mask_case1] = 10 ** (0.952 - 3.04 * Ic_values[mask_case1])
    
    # Case 2: When 3.27 < Ic < 4.0
    mask_case2 = (Ic_values > 3.27) & (Ic_values < 4.0)
    k[mask_case2] = 10 ** (-4.52 - 1.37 * Ic_values[mask_case2])
    
    return k

def setup_grid(utmX0=558400, utmY0=4178000, utmX1= 568500, utmY1 = 4183800, width=100, geoplot='Alameda_shape.geojson'):
    '''function to set up a grid of points
    input: utmX0 is the minimum utmX value (default Alameda)
    input: utmY0 is the minimum utmY value (default Alameda)
    input: utmX1 is the maximum utmX value (default Alameda)
    input: utmY1 is the maximum utmY value (default Alameda)
    input: width is the width of the grid in meters (default 100)
    input: geoplot is the geojson of the area of interest (default Alameda)
    output: points is a pandas dataframe with the lat, lon, utmX, and utmY of each point
    '''
    
    ## make a dataframe of points with lat, lon, utmX, utmY
    points = pd.DataFrame(columns=['lat', 'lon', 'utmX', 'utmY'])
    utmX = utmX0
    utmY = utmY0
    while utmX <= utmX1:
        while utmY <= utmY1:
            lat, lon = utm.to_latlon(utmX, utmY, 10, northern=True)
            points = pd.concat([points, pd.DataFrame([[lat, lon, utmX, utmY]], columns=['lat', 'lon', 'utmX', 'utmY'])])
            utmY += width
        utmX += width
        utmY = utmY0

    # cut off points outside of the area of interest using a shapefile
    gdf = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points['lon'], points['lat']))

    gdf.set_crs(epsg=4326, inplace=True)
    Alameda = gpd.read_file(geoplot)
    points = gpd.sjoin(gdf, Alameda)
    points.reset_index(inplace=True, drop=True)

    return points

def setup_grid_no_crop(utmX0=558400, utmY0=4178000, utmX1= 568500, utmY1 = 4183800, width=100):
    '''function to set up a grid of points without cropping to a shapefile
    input: utmX0 is the minimum utmX value (default Alameda)
    input: utmY0 is the minimum utmY value (default Alameda)
    input: utmX1 is the maximum utmX value (default Alameda)
    input: utmY1 is the maximum utmY value (default Alameda)
    input: width is the width of the grid in meters (default 100)
    output: points is a geopandas geodataframe with the lat, lon, utmX, and utmY of each point
    '''
    
    ## make a dataframe of points with lat, lon, utmX, utmY
    points = pd.DataFrame(columns=['lat', 'lon', 'utmX', 'utmY'])
    utmX = utmX0
    utmY = utmY0
    while utmX <= utmX1:
        while utmY <= utmY1:
            lat, lon = utm.to_latlon(utmX, utmY, 10, northern=True)
            points = pd.concat([points, pd.DataFrame([[lat, lon, utmX, utmY]], columns=['lat', 'lon', 'utmX', 'utmY'])])
            utmY += width
        utmX += width
        utmY = utmY0

    gdf = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points['lon'], points['lat']), crs='EPSG:4326')

    return gdf

def SBT_from_Ic(I_c):
    '''
    function calc_SBT to determine soil behavior type from Robertson 2010
    inputs:
        I_c: array of Ic values
    outputs:
        sbt: array of SBT values
        sbt_descr: array of SBT descriptions
    '''
    # Determine SBT based on I_SBT thresholds to function for an array
    sbt = np.zeros_like(I_c)
    sbt_descr = np.empty_like(I_c, dtype=object)
    sbt[(I_c <= 1.31)] = 7
    sbt_descr[(I_c <= 1.31)] = "Dense sand to gravelly sand"
    sbt[(I_c > 1.31) & (I_c <= 2.05)] = 6
    sbt_descr[(I_c > 1.31) & (I_c <= 2.05)] = "Sands: clean sands to silty sands"
    sbt[(I_c > 2.05) & (I_c <= 2.60)] = 5
    sbt_descr[(I_c > 2.05) & (I_c <= 2.60)] = "Sand mixtures: silty sand to sandy silt"
    sbt[(I_c > 2.60) & (I_c <= 2.95)] = 4
    sbt_descr[(I_c > 2.60) & (I_c <= 2.95)] = "Silt mixtures: clayey silt & silty clay"
    sbt[(I_c > 2.95) & (I_c <= 3.60)] = 3
    sbt_descr[(I_c > 2.95) & (I_c <= 3.60)] = "Clays: clay to silty clay"
    sbt[(I_c > 3.60)] = 2
    sbt_descr[(I_c > 3.60)] = "Clay - organic soil"

    return sbt, sbt_descr


def detect_change_points(values, model="l2", penalty=30):
    ''' Apply the PELT change point detection algorithm
    '''
    algo = rpt.Pelt(model=model).fit(values)
    change_points = algo.predict(pen=penalty)
    return change_points


def create_geotiff(points_gdf, data_input, output_filename):
    """
    Create a GeoTIFF either from a column in points_gdf or a constant value.

    Parameters:
    - points_gdf (GeoDataFrame): Input GeoDataFrame with utmX/utmY columns and optionally data columns.
    - data_input (str or float): Column name (str) or constant value (float) for raster values.
    - output_filename (str): Path to the output GeoTIFF file.
    """
    # Extract the bounds of points_gdf
    minx, miny, maxx, maxy = points_gdf.total_bounds

    # Define grid resolution based on unique lat/lon values
    unique_utmx = sorted(points_gdf['utmX'].unique())
    unique_utmy = sorted(points_gdf['utmY'].unique())
    resolution_x = abs(unique_utmx[1] - unique_utmx[0])  # Longitude resolution
    resolution_y = abs(unique_utmy[1] - unique_utmy[0])  # Latitude resolution

    # Determine the number of rows and columns for the raster
    n_cols = int((maxx - minx) / resolution_x) + 1
    n_rows = int((maxy - miny) / resolution_y) + 1

    # Define the affine transform from bounds and resolution
    transform = from_bounds(minx, miny, maxx, maxy, n_cols, n_rows)

    # Initialize the raster data array with nodata value (-9999)
    raster_data = np.full((n_rows, n_cols), -9999, dtype=np.float32)

    # Populate raster data
    for _, row in points_gdf.iterrows():
        # Calculate row and column indices
        col = int((row['utmX'] - minx) / resolution_x)
        row_idx = int((maxy - row['utmY']) / resolution_y)

        # Clamp indices within valid bounds
        col = min(max(col, 0), n_cols - 1)
        row_idx = min(max(row_idx, 0), n_rows - 1)

        # Assign values based on input
        if isinstance(data_input, str):  # If data_input is a column name
            raster_data[row_idx, col] = row[data_input]
        elif isinstance(data_input, (int, float)):  # If data_input is a constant value
            raster_data[row_idx, col] = data_input
        else:
            raise ValueError("data_input must be a column name (str) or a constant value (float or int)")

    # Write the raster data to a single-band GeoTIFF
    with rasterio.open(
        output_filename,
        "w",
        driver="GTiff",
        height=n_rows,
        width=n_cols,
        count=1,
        dtype="float32",
        crs=points_gdf.crs,  # Use the CRS of points_gdf
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(raster_data, 1)

def get_tiff_values_flood(geotiff_file, lon, lat):
    '''
    function get_tiff_values_flood to get the flood depth values from a geotiff file for a list of coordinates
    input the path to the geotiff file and the list of UTM-X and UTM-Y coordinates
    output the list of flood depth values
    '''
    with rasterio.open(geotiff_file) as src:
        rows, cols = zip(*[src.index(x, y) for x, y in zip(lon, lat)])
        values = src.read(1)[rows, cols]

    return values

def assign_elevation(df, elev_tif_path):
    ''' 
    function assign_elevation to assign elevations to a df
    input the dataframe and the path to folder with elevation geotiffs
    output the dataframe with elevations
    '''
    elevs = get_tiff_values_flood(elev_tif_path, df['Longitude'], df['Latitude'])
    df['elevation'] = elevs

    return df

def get_groundwater_values(geotiff_folder,points):
    ''' Function to load groundwater depths at each point from the geotiff file
    points is a dataframe with lat and lon columns
    geotiff_folder is the path to the geotiff folder
    output is an updated dataframe with the depths at each point
    '''
    ## the geotiffs are stored with as County_wt_[tide]_noghb_Kh[Kh.]p[.Kh]_slr[slr.]p[.slr]m.tif
    # open each file in the folder
    for filename in filter(lambda x: x[-4:] == '.tif', os.listdir(geotiff_folder)):
        # extract the tide, Kh, and slr values from the filename
        tide = str(filename.split('_')[2])
        slr = float(str(filename[-9]) + '.' + str(filename[-7:-5]))
        Kh = float(str(filename[-16]) + '.' + str(filename[-14]))
        vals = []
        for index, row in points.iterrows():
            vals.append(get_tiff_value(geotiff_folder + filename, points['lat'][index], points['lon'][index]))
        points['slr'+str(slr)+'_Kh'+str(Kh)+'_'+tide] = vals
    return points

def get_tiff_value(geotiff_file, lat, lon):
    with rasterio.open(geotiff_file) as src:
        # Convert latitude and longitude to the corresponding pixel coordinates
        row, col = src.index(lon, lat)
        
        # Read the water depth value at the pixel coordinates
        value = src.read(1, window=((row, row+1), (col, col+1)))

    return value[0][0]

def ms_layer_assignment(depths, SBTs, fill_flag):
    '''function ms_layer_assignment to assign layers to a borehole
    input: depths is a list of depths
    input: SBTs is a list of soil behavior types
    input: fill_flag is a flag that is 1 if within Artificial Fill and 0 otherwise
    output: layers_dict is a dictionary with the layers assigned to the borehole
    '''
    if len(depths) >= 3 + fill_flag:
        # may combine some layers
        if 6 not in SBTs:
            # print('no sand in', bh)
            if fill_flag:
                layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:]), 'Merritt Sand': np.nan}
            else:
                ## maybe should discard this data (not definitive that the whole profile is YBM)
                # layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths), 'Merritt Sand': np.nan}
                layers_dict = {'Fill': 0, 'Young Bay Mud': np.nan, 'Merritt Sand': np.nan}
        else:
            # need the possibility to combine layers and find all sand layers to assign the thickest as MS
            sand_ind = np.where(np.array(SBTs) >= 6)[0]
            if fill_flag:
                if len(sand_ind) == 1:
                    if sand_ind[0] == 0:
                        layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:]), 'Merritt Sand': np.nan}
                    else:
                        layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:sand_ind[0]]), 'Merritt Sand': depths[sand_ind[0]], 'Below Merritt Sand': np.sum(depths[sand_ind[0]+1:])}
                elif len(sand_ind) == 2:
                    if sand_ind[0] == 0:
                        layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:sand_ind[1]]), 'Merritt Sand': depths[sand_ind[1]], 'Below Merritt Sand': np.sum(depths[sand_ind[1]+1:])}
                    else:
                        # find the thickest sand layer
                        ms_ind = sand_ind[np.argmax(np.array(depths)[np.array(sand_ind)])]
                        # since there are only two, this time we do not need to separate if ms_ind is high or low
                        btwn_d = np.sum(depths[sand_ind[0]+1:sand_ind[1]])
                        out_d = depths[sand_ind[0]] + depths[sand_ind[1]]
                        if btwn_d > out_d:
                            # just ms_ind is MS
                            layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:ms_ind]), 'Merritt Sand': depths[ms_ind], 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                        else:
                            layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:sand_ind[0]]), 'Merritt Sand': np.sum(depths[sand_ind[0]:sand_ind[1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[1]+1:])}
                else: # len(sand_ind) > 2
                    # check if the first is fill
                    if sand_ind[0] == 0:
                        sand_ind = sand_ind[1:]
                        # fill will be the first layer
                        ms_ind = sand_ind[np.argmax(np.array(depths)[np.array(sand_ind)])]
                        # check where ms_ind is
                        if ms_ind == 0:
                            btwn_d = np.sum(depths[ms_ind+1:sand_ind[1]])
                            out_d = depths[sand_ind[0]] + depths[sand_ind[1]]
                            if btwn_d > out_d:
                                layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:ms_ind]), 'Merritt Sand': depths[ms_ind], 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                            else:
                                layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:sand_ind[0]]), 'Merritt Sand': np.sum(depths[sand_ind[0]:sand_ind[1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[1]+1:])}
                        elif ms_ind == sand_ind[-1]:
                            btwn_d = np.sum(depths[np.array(sand_ind)[-2]+1:ms_ind])
                            out_d = depths[sand_ind[-2]] + depths[ms_ind]
                            if btwn_d > out_d:
                                layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:ms_ind]), 'Merritt Sand': np.sum(depths[ms_ind]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                            else:
                                layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:sand_ind[-2]]), 'Merritt Sand': np.sum(depths[sand_ind[-2]:ms_ind]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                        else:
                            ## ms_ind is in the middle
                            # check one below and one above
                            max_ind = np.where(sand_ind == ms_ind)[0][0]
                            btwn_d = np.sum(depths[np.array(sand_ind)[max_ind-1]+1:ms_ind])
                            btwn_d2 = np.sum(depths[ms_ind+1:np.array(sand_ind)[max_ind+1]-1])
                            if btwn_d > btwn_d2:
                                out_d = depths[np.array(sand_ind)[max_ind-1]] + depths[ms_ind]
                                layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:sand_ind[max_ind-1]]), 'Merritt Sand': np.sum(depths[sand_ind[max_ind-1]:ms_ind+1]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                            else:
                                out_d = depths[ms_ind] + depths[np.array(sand_ind)[max_ind+1]]
                                layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:ms_ind]), 'Merritt Sand': np.sum(depths[ms_ind:sand_ind[max_ind+1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[max_ind+1]+1:])}
                    else: ## sand is not the first layer
                        ms_ind = sand_ind[np.argmax(np.array(depths)[np.array(sand_ind)])]
                        # check where ms_ind is
                        if ms_ind == 0:
                            btwn_d = np.sum(depths[ms_ind+1:sand_ind[1]])
                            out_d = depths[sand_ind[0]] + depths[sand_ind[1]]
                            if btwn_d > out_d:
                                layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:ms_ind]), 'Merritt Sand': depths[ms_ind], 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                            else:
                                layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:sand_ind[0]]), 'Merritt Sand': np.sum(depths[sand_ind[0]:sand_ind[1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[1]+1:])}
                        elif ms_ind == sand_ind[-1]:
                            btwn_d = np.sum(depths[sand_ind[-2]+1:ms_ind])
                            out_d = depths[sand_ind[-2]] + depths[ms_ind]
                            if btwn_d > out_d:
                                layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:ms_ind]), 'Merritt Sand': np.sum(depths[ms_ind]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                            else:
                                layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:sand_ind[-2]]), 'Merritt Sand': np.sum(depths[sand_ind[-2]:ms_ind+1]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                        else:
                            ## ms_ind is in the middle.. check one below and one above
                            max_ind = np.where(sand_ind == ms_ind)[0][0]
                            btwn_d = np.sum(depths[np.array(sand_ind)[max_ind-1]+1:ms_ind])
                            btwn_d2 = np.sum(depths[ms_ind+1:np.array(sand_ind)[max_ind+1]])
                            if btwn_d > btwn_d2:
                                out_d = depths[np.array(sand_ind)[max_ind-1]] + depths[ms_ind]
                                layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:sand_ind[max_ind-1]]), 'Merritt Sand': np.sum(depths[sand_ind[max_ind-1]:ms_ind+1]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                            else:
                                out_d = depths[ms_ind] + depths[np.array(sand_ind)[max_ind+1]]
                                layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:ms_ind]), 'Merritt Sand': np.sum(depths[ms_ind:sand_ind[max_ind+1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[max_ind+1]+1:])}
            else:  ## not in fill
                if len(sand_ind) == 1:
                    layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:sand_ind[0]]), 'Merritt Sand': depths[sand_ind[0]], 'Below Merritt Sand': np.sum(depths[sand_ind[0]+1:])}
                elif len(sand_ind) == 2:
                    # check if between is more than outside
                    btwn_d = np.sum(depths[sand_ind[0]+1:sand_ind[1]])
                    out_d = depths[sand_ind[0]] + depths[sand_ind[1]]
                    if btwn_d > out_d:
                        ms_ind = sand_ind[np.argmax(np.array(depths)[np.array(sand_ind)])]
                        layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:ms_ind]), 'Merritt Sand': np.sum(depths[ms_ind]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                    else:
                        layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:sand_ind[0]]), 'Merritt Sand': np.sum(depths[sand_ind[0]:sand_ind[1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[1]+1:])}
                else: # len(sand_ind) > 2
                    ms_ind = sand_ind[np.argmax(np.array(depths)[np.array(sand_ind)])]
                    max_ind = np.where(sand_ind == ms_ind)[0][0]
                    btwn_d = np.sum(depths[np.array(sand_ind)[max_ind-1]+1:ms_ind])
                    btwn_d2 = np.sum(depths[ms_ind+1:np.array(sand_ind)[max_ind+1]])
                    if btwn_d > btwn_d2:
                        out_d = depths[np.array(sand_ind)[max_ind-1]] + depths[ms_ind]
                        layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:sand_ind[max_ind-1]]), 'Merritt Sand': np.sum(depths[sand_ind[max_ind-1]:ms_ind+1]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                    else:
                        out_d = depths[ms_ind] + depths[sand_ind[max_ind+1]]
                        layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:ms_ind]), 'Merritt Sand': np.sum(depths[ms_ind:sand_ind[max_ind+1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[max_ind+1]+1:])}

    else: # len(depths) < 3 + fill_flag:
        ## this is where we have fewer layers than expected
        if 6 not in SBTs:
            if len(depths) == 1:
                ## so that there will not be an entire CPT assigned to fill
                layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.nan, 'Merritt Sand': np.nan}
            # print('no sand in', bh)
            elif fill_flag:
                layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:]), 'Merritt Sand': np.nan}
            else:
                layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths), 'Merritt Sand': np.nan}
        else:
            MS_ind = np.where(np.array(SBTs) == 6)[0][-1]
            if fill_flag:
                if len(depths) <=2:
                    layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.nan, 'Merritt Sand': np.nan}
                    # continue
                elif MS_ind == 0:
                    ## there is only one layer with SBT 6, assume it is fill
                    layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:]), 'Merritt Sand': np.nan}
                else:
                    if len(depths) > MS_ind+1:
                        layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:MS_ind]), 'Merritt Sand': depths[MS_ind], 'Below Merritt Sand': np.sum(depths[MS_ind+1:])}
                    else:
                        layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:MS_ind]), 'Merritt Sand': depths[MS_ind]}
            else:
                if len(depths) <=1:
                    layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.nan, 'Merritt Sand': np.nan}
                    # continue
                else:
                    if len(depths) > MS_ind+1:
                        layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[:MS_ind]), 'Merritt Sand': depths[MS_ind], 'Below Merritt Sand': np.sum(depths[MS_ind+1:])}
                    else:
                        layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[:MS_ind]), 'Merritt Sand': depths[MS_ind]}
    return layers_dict

def msaf_layer_assignment(depths, SBTs, fill_flag, limit=5):
    '''function msaf_layer_assignment to assign layers to a borehole
    where fill layer will be limited to a given depth
    input: depths is a list of depths
    input: SBTs is a list of soil behavior types
    input: fill_flag is a flag that is 1 if within Artificial Fill and 0 otherwise
    input: limit is the depth limit of the fill layer (default 5m)
    output: layers_dict is a dictionary with the layers assigned to the borehole
    '''
    if len(depths) >= 3 + fill_flag:
        # may combine some layers
        if 6 not in SBTs:
            # print('no sand in', bh)
            if fill_flag:
                if depths[0] <= limit:
                    layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:]), 'Merritt Sand': np.nan}
                else:
                    ## no sand i detected and the fill is too deep, discard
                    layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.nan, 'Merritt Sand': np.nan}
            else:
                ## maybe should discard this data (not definitive that the whole profile is YBM)
                # layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths), 'Merritt Sand': np.nan}
                layers_dict = {'Fill': 0, 'Young Bay Mud': np.nan, 'Merritt Sand': np.nan}
        else:
            # need the possibility to combine layers and find all sand layers to assign the thickest as MS
            sand_ind = np.where(np.array(SBTs) >= 6)[0]
            if fill_flag:
                if len(sand_ind) == 1:
                    if sand_ind[0] == 0:
                        if depths[0] <= limit:
                            layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:]), 'Merritt Sand': np.nan}
                        else:
                            layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.nan, 'Merritt Sand': np.nan}
                    else:
                        if depths[0] <= limit:
                            layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:sand_ind[0]]), 'Merritt Sand': depths[sand_ind[0]], 'Below Merritt Sand': np.sum(depths[sand_ind[0]+1:])}
                        else: 
                            layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.sum(depths[:sand_ind[0]]), 'Merritt Sand': depths[sand_ind[0]], 'Below Merritt Sand': np.sum(depths[sand_ind[0]+1:])}
                elif len(sand_ind) == 2:
                    if sand_ind[0] == 0:
                        if depths[0] <= limit:
                            layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:sand_ind[1]]), 'Merritt Sand': depths[sand_ind[1]], 'Below Merritt Sand': np.sum(depths[sand_ind[1]+1:])}
                        else:
                            layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.sum(depths[1:sand_ind[1]]), 'Merritt Sand': depths[sand_ind[1]], 'Below Merritt Sand': np.sum(depths[sand_ind[1]+1:])}
                    else:
                        # find the thickest sand layer
                        ms_ind = sand_ind[np.argmax(np.array(depths)[np.array(sand_ind)])]
                        # since there are only two, this time we do not need to separate if ms_ind is high or low
                        btwn_d = np.sum(depths[sand_ind[0]+1:sand_ind[1]])
                        out_d = depths[sand_ind[0]] + depths[sand_ind[1]]
                        if depths[0] <= limit:
                            if btwn_d > out_d:
                                # just ms_ind is MS
                                layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:ms_ind]), 'Merritt Sand': depths[ms_ind], 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                            else:
                                layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:sand_ind[0]]), 'Merritt Sand': np.sum(depths[sand_ind[0]:sand_ind[1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[1]+1:])}
                        else:
                            if btwn_d > out_d:
                                # just ms_ind is MS
                                layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.sum(depths[0:ms_ind]), 'Merritt Sand': depths[ms_ind], 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                            else:
                                layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.sum(depths[0:sand_ind[0]]), 'Merritt Sand': np.sum(depths[sand_ind[0]:sand_ind[1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[1]+1:])}

                else: # len(sand_ind) > 2
                    # check if the first is fill
                    if sand_ind[0] == 0:
                        sand_ind = sand_ind[1:]
                        # fill will be the first layer
                        ms_ind = sand_ind[np.argmax(np.array(depths)[np.array(sand_ind)])]
                        # check where ms_ind is
                        if ms_ind == 0:
                            btwn_d = np.sum(depths[ms_ind+1:sand_ind[1]])
                            out_d = depths[sand_ind[0]] + depths[sand_ind[1]]
                            if depths[0] <= limit:
                                if btwn_d > out_d:
                                    layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:ms_ind]), 'Merritt Sand': depths[ms_ind], 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                                else:
                                    layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:sand_ind[0]]), 'Merritt Sand': np.sum(depths[sand_ind[0]:sand_ind[1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[1]+1:])}
                            else:
                                if btwn_d > out_d:
                                    layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.sum(depths[0:ms_ind]), 'Merritt Sand': depths[ms_ind], 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                                else:
                                    layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.sum(depths[0:sand_ind[0]]), 'Merritt Sand': np.sum(depths[sand_ind[0]:sand_ind[1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[1]+1:])}
                        elif ms_ind == sand_ind[-1]:
                            btwn_d = np.sum(depths[np.array(sand_ind)[-2]+1:ms_ind])
                            out_d = depths[sand_ind[-2]] + depths[ms_ind]
                            if depths[0] <= limit:
                                if btwn_d > out_d:
                                    layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:ms_ind]), 'Merritt Sand': np.sum(depths[ms_ind]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                                else:
                                    layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:sand_ind[-2]]), 'Merritt Sand': np.sum(depths[sand_ind[-2]:ms_ind]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                            else:
                                if btwn_d > out_d:
                                    layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.sum(depths[0:ms_ind]), 'Merritt Sand': np.sum(depths[ms_ind]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                                else:
                                    layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.sum(depths[0:sand_ind[-2]]), 'Merritt Sand': np.sum(depths[sand_ind[-2]:ms_ind]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                        else:
                            ## ms_ind is in the middle
                            # check one below and one above
                            max_ind = np.where(sand_ind == ms_ind)[0][0]
                            btwn_d = np.sum(depths[np.array(sand_ind)[max_ind-1]+1:ms_ind])
                            btwn_d2 = np.sum(depths[ms_ind+1:np.array(sand_ind)[max_ind+1]-1])
                            if depths[0] <= limit:
                                if btwn_d > btwn_d2:
                                    out_d = depths[np.array(sand_ind)[max_ind-1]] + depths[ms_ind]
                                    layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:sand_ind[max_ind-1]]), 'Merritt Sand': np.sum(depths[sand_ind[max_ind-1]:ms_ind+1]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                                else:
                                    out_d = depths[ms_ind] + depths[np.array(sand_ind)[max_ind+1]]
                                    layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:ms_ind]), 'Merritt Sand': np.sum(depths[ms_ind:sand_ind[max_ind+1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[max_ind+1]+1:])}
                            else:
                                if btwn_d > btwn_d2:
                                    out_d = depths[np.array(sand_ind)[max_ind-1]] + depths[ms_ind]
                                    layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.sum(depths[0:sand_ind[max_ind-1]]), 'Merritt Sand': np.sum(depths[sand_ind[max_ind-1]:ms_ind+1]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                                else:
                                    out_d = depths[ms_ind] + depths[np.array(sand_ind)[max_ind+1]]
                                    layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.sum(depths[0:ms_ind]), 'Merritt Sand': np.sum(depths[ms_ind:sand_ind[max_ind+1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[max_ind+1]+1:])}
                    else: ## sand is not the first layer
                        ms_ind = sand_ind[np.argmax(np.array(depths)[np.array(sand_ind)])]
                        # check where ms_ind is
                        if ms_ind == 0:
                            btwn_d = np.sum(depths[ms_ind+1:sand_ind[1]])
                            out_d = depths[sand_ind[0]] + depths[sand_ind[1]]
                            if btwn_d > out_d:
                                layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:ms_ind]), 'Merritt Sand': depths[ms_ind], 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                            else:
                                layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:sand_ind[0]]), 'Merritt Sand': np.sum(depths[sand_ind[0]:sand_ind[1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[1]+1:])}
                        elif ms_ind == sand_ind[-1]:
                            btwn_d = np.sum(depths[sand_ind[-2]+1:ms_ind])
                            out_d = depths[sand_ind[-2]] + depths[ms_ind]
                            if btwn_d > out_d:
                                layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:ms_ind]), 'Merritt Sand': np.sum(depths[ms_ind]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                            else:
                                layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:sand_ind[-2]]), 'Merritt Sand': np.sum(depths[sand_ind[-2]:ms_ind+1]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                        else:
                            ## ms_ind is in the middle.. check one below and one above
                            max_ind = np.where(sand_ind == ms_ind)[0][0]
                            btwn_d = np.sum(depths[np.array(sand_ind)[max_ind-1]+1:ms_ind])
                            btwn_d2 = np.sum(depths[ms_ind+1:np.array(sand_ind)[max_ind+1]])
                            if btwn_d > btwn_d2:
                                out_d = depths[np.array(sand_ind)[max_ind-1]] + depths[ms_ind]
                                layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:sand_ind[max_ind-1]]), 'Merritt Sand': np.sum(depths[sand_ind[max_ind-1]:ms_ind+1]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                            else:
                                out_d = depths[ms_ind] + depths[np.array(sand_ind)[max_ind+1]]
                                layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:ms_ind]), 'Merritt Sand': np.sum(depths[ms_ind:sand_ind[max_ind+1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[max_ind+1]+1:])}
            else:  ## not in fill
                if len(sand_ind) == 1:
                    layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:sand_ind[0]]), 'Merritt Sand': depths[sand_ind[0]], 'Below Merritt Sand': np.sum(depths[sand_ind[0]+1:])}
                elif len(sand_ind) == 2:
                    # check if between is more than outside
                    btwn_d = np.sum(depths[sand_ind[0]+1:sand_ind[1]])
                    out_d = depths[sand_ind[0]] + depths[sand_ind[1]]
                    if btwn_d > out_d:
                        ms_ind = sand_ind[np.argmax(np.array(depths)[np.array(sand_ind)])]
                        layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:ms_ind]), 'Merritt Sand': np.sum(depths[ms_ind]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                    else:
                        layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:sand_ind[0]]), 'Merritt Sand': np.sum(depths[sand_ind[0]:sand_ind[1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[1]+1:])}
                else: # len(sand_ind) > 2
                    ms_ind = sand_ind[np.argmax(np.array(depths)[np.array(sand_ind)])]
                    max_ind = np.where(sand_ind == ms_ind)[0][0]
                    btwn_d = np.sum(depths[np.array(sand_ind)[max_ind-1]+1:ms_ind])
                    btwn_d2 = np.sum(depths[ms_ind+1:np.array(sand_ind)[max_ind+1]])
                    if btwn_d > btwn_d2:
                        out_d = depths[np.array(sand_ind)[max_ind-1]] + depths[ms_ind]
                        layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:sand_ind[max_ind-1]]), 'Merritt Sand': np.sum(depths[sand_ind[max_ind-1]:ms_ind+1]), 'Below Merritt Sand': np.sum(depths[ms_ind+1:])}
                    else:
                        out_d = depths[ms_ind] + depths[sand_ind[max_ind+1]]
                        layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[0:ms_ind]), 'Merritt Sand': np.sum(depths[ms_ind:sand_ind[max_ind+1]+1]), 'Below Merritt Sand': np.sum(depths[sand_ind[max_ind+1]+1:])}

    else: # len(depths) < 3 + fill_flag:
        ## this is where we have fewer layers than expected
        if 6 not in SBTs:
            if len(depths) == 1:
                ## so that there will not be an entire CPT assigned to fill
                layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.nan, 'Merritt Sand': np.nan}
            # print('no sand in', bh)
            elif fill_flag:
                if depths[0] <= limit:
                    layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:]), 'Merritt Sand': np.nan}
                else:
                    layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.nan, 'Merritt Sand': np.nan}
            else:
                layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths), 'Merritt Sand': np.nan}
        else:
            MS_ind = np.where(np.array(SBTs) == 6)[0][-1]
            if fill_flag:
                if len(depths) <=2:
                    layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.nan, 'Merritt Sand': np.nan}
                    # continue
                elif MS_ind == 0:
                    ## there is only one layer with SBT 6, assume it is fill
                    if depths[0] <= limit:
                        layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:]), 'Merritt Sand': np.nan}
                    else:
                        layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.nan, 'Merritt Sand': np.nan}
                else:
                    if depths[0] <= limit:
                        if len(depths) > MS_ind+1:
                            layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:MS_ind]), 'Merritt Sand': depths[MS_ind], 'Below Merritt Sand': np.sum(depths[MS_ind+1:])}
                        else:
                            layers_dict = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:MS_ind]), 'Merritt Sand': depths[MS_ind]}
                    else:
                        if len(depths) > MS_ind+1:
                            layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.sum(depths[:MS_ind]), 'Merritt Sand': depths[MS_ind], 'Below Merritt Sand': np.sum(depths[MS_ind+1:])}
                        else:
                            layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.sum(depths[:MS_ind]), 'Merritt Sand': depths[MS_ind]}
            else:
                if len(depths) <=1:
                    layers_dict = {'Fill': np.nan, 'Young Bay Mud': np.nan, 'Merritt Sand': np.nan}
                    # continue
                else:
                    if len(depths) > MS_ind+1:
                        layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[:MS_ind]), 'Merritt Sand': depths[MS_ind], 'Below Merritt Sand': np.sum(depths[MS_ind+1:])}
                    else:
                        layers_dict = {'Fill': 0, 'Young Bay Mud': np.sum(depths[:MS_ind]), 'Merritt Sand': depths[MS_ind]}
    return layers_dict


def align_rasters(target_raster_path, input_raster_path):
    with rasterio.open(target_raster_path) as target:
        # Use the target's transform and dimensions
        target_transform = target.transform
        target_width = target.width
        target_height = target.height

        target_band = target.read(1, masked=True)

        with rasterio.open(input_raster_path) as input_r:
            # Calculate the necessary transform and output dimensions for alignment
            transform, width, height = calculate_default_transform(
                input_r.crs, target.crs, input_r.width, input_r.height,
                *input_r.bounds, dst_width=target_width, dst_height=target_height, dst_transform=target_transform)

            kwargs = input_r.meta.copy()
            kwargs.update({
                'crs': target.crs,
                'transform': target_transform,
                'width': target_width,
                'height': target_height
            })

            # Create an array to hold the reprojected data
            reprojected_band = np.full((target_height, target_width), np.nan, dtype=np.float32)

            # Reproject the data
            reproject(
                source=rasterio.band(input_r, 1),
                destination=reprojected_band,
                src_transform=input_r.transform,
                src_crs=input_r.crs,
                dst_transform=target_transform,
                dst_crs=target.crs,
                resampling=Resampling.nearest)

    return target_band, reprojected_band, target_transform

def get_any_tiff_values(geotiff_file, lats, lons):
    """
    Get flood depth values from a GeoTIFF file for lists of latitude and longitude coordinates.
    
    Parameters:
      geotiff_file (str): Path to the GeoTIFF file.
      lats (list or array): List of latitudes (assumed in EPSG:4326).
      lons (list or array): List of longitudes (assumed in EPSG:4326).
    
    Returns:
      numpy.ndarray: An array of flood depth values corresponding to the provided coordinates.
                     If a coordinate is outside the raster bounds or causes an error, np.nan is returned.
    """
    source_crs = 'EPSG:4326'
    
    with rasterio.open(geotiff_file) as src:
        tif_crs = src.crs
        # Transform coordinates if necessary
        if tif_crs != source_crs:
            transformed_lon, transformed_lat = warp_transform(source_crs, tif_crs, lons, lats)
        else:
            transformed_lon, transformed_lat = lons, lats

        # Read the entire band once
        band = src.read(1)
        values = []
        for x, y in zip(transformed_lon, transformed_lat):
            try:
                # Get pixel indices corresponding to the transformed coordinates
                row, col = src.index(x, y)
                # Check if the indices are within the bounds of the raster
                if 0 <= row < band.shape[0] and 0 <= col < band.shape[1]:
                    values.append(band[row, col])
                else:
                    values.append(np.nan)
            except Exception:
                values.append(np.nan)
    
    return np.array(values)

def create_mask_from_band(file_path):
    """Create a mask from a band where data is not NaN."""
    with rasterio.open(file_path) as src:
        band = src.read(1)
        mask = np.isnan(band)  # True where band is NaN
        return ~mask  # Invert mask: True where band is not NaN

def apply_mask_to_band(file_path, mask):
    """Apply a given mask to a raster band, setting unmasked areas to NaN."""
    with rasterio.open(file_path) as src:
        band = src.read(1)
        band[~mask] = np.nan  # Set areas outside the mask to NaN
        return band