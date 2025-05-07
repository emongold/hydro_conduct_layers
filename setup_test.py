### setup_test.py to set up the test cases for the groundwater inputs
## saves the input parameters to .csv and saves the outputs as geotiffs
# Author: Emily Mongold, Dec. 2024

import numpy as np
import pandas as pd
import geopandas as gpd
from pykrige.ok import OrdinaryKriging
from scipy import stats
from shapely.geometry import Point
import utm
import rasterio
from rasterio.transform import from_bounds
import ruptures as rpt

from regional_K.cpt_functions import setup_cpt, calc_SBT, soil_stress, detect_change_points, k_from_Ic, setup_grid_no_crop, solve_Ic, create_geotiff

names = ['test1', 'test2', 'test3','test4','test5','test6','test7','test8','test9']
methods = ['heuristic','heuristic','heuristic','heuristic','naive','MS','MS','MS','MS_AF']
K_source = ['empirical','low','table','high','empirical','empirical','table','table','empirical']
# add_bhs = [True,True,False,False,False,True]

penalties_0 = [0,0,0,0,0,0,0,0,0]
penalties_1 = [0,0,0,0,24,24,24,16,24]
penalties_2 = [0,0,0,0,44,44,44,44,44]
thresholds_1 = [0,0,0,0,0,0,0,0,0]
thresholds_2 = [0,0,0,0,24,24,24,24,24]

shapefile_path = './deposits_shp/sfq2py.shp'

save_params = pd.DataFrame(columns=['hk_m_day_0','thick_m_0','hk_m_day_1','thick_m_1','hk_m_day_2','thick_m_2','hk_m_day_3','thick_m_3'], index=names)

input_params = pd.DataFrame({'names': names, 'methods': methods, 'K_source': K_source, 
                             'penalties_0': penalties_0, 'penalties_1': penalties_1, 'penalties_2': penalties_2,
                             'thresholds_1': thresholds_1, 'thresholds_2': thresholds_2})
input_params.to_csv('./input_params.csv')

# run the test cases
# points = setup_grid(geoplot='./geojson/alameda_city.geojson')  ## this line crops everything that is not on land
points = setup_grid_no_crop()
points['utmX'] = pd.to_numeric(points['utmX'], errors='coerce')
points['utmY'] = pd.to_numeric(points['utmY'], errors='coerce')
points = points.dropna(subset=['utmX', 'utmY'])

cpt = setup_cpt('./USGS_CPT_data/')
cpt = soil_stress(cpt)
calc_SBT(cpt)
for _, data in cpt.items():
    data['CPT_data']['I_c'] = solve_Ic(data['CPT_data']['q_c'], data['CPT_data']['sig_v'],data['CPT_data']['sig_prime_v'], data['CPT_data']['f_s'])
    data['CPT_data']['k_fromIc'] = k_from_Ic(data['CPT_data']['I_c'])


data = gpd.read_file(shapefile_path)
data.crs = 'EPSG:4326'
cpts_df = pd.DataFrame.from_dict(cpt, orient='index')
cpts_df.drop(['CPT_data'], axis=1, inplace=True)
cpts_gdf = gpd.GeoDataFrame(cpts_df, geometry=gpd.points_from_xy(cpts_df['Lon'], cpts_df['Lat']))
cpts_gdf.crs = 'EPSG:4326'
joined = gpd.sjoin(cpts_gdf, data, how='left')

## load in the borehole lithology data
cgs_data = pd.read_csv('./cgs_bhs_0/CGSBoreholeDB_BoreholeLocations.csv')
lith = pd.read_csv('./cgs_bhs_0/lithology.csv')
lith = lith[lith['remarks'].str.contains('fill', case=False, na=False)]
lith = lith.merge(cgs_data[['well_name', 'latitude', 'longitude']], on='well_name', how='left')
lith_gdf = gpd.GeoDataFrame(lith, geometry=gpd.points_from_xy(lith.longitude, lith.latitude),crs='EPSG:4326')
lith_gdf['Fill'] = lith_gdf['bottom_depth'] * 0.3048 # convert to meters
## drop the 64 from the lith data index, it said 'filled' and was not artificial fill.
lith = lith[lith.index != 64]
lith_gdf = lith_gdf[lith_gdf.index != 64]
lith_gdf['UTM -X'], lith_gdf['UTM -Y'] = zip(*[utm.from_latlon(lat, lon)[:2] for lat, lon in zip(lith_gdf['latitude'], lith_gdf['longitude'])])

for i in range(len(names)):
    method = methods[i]
    if method == 'heuristic':
        layers = pd.read_csv('./layer_thicknesses.csv', index_col=0)
        results_k_values = {}
        cp_layers_dict = {}
        for _,row in layers.iterrows():  # [layers['Filename'].str.startswith('ALC')]
            fill_thickness = row['Fill']
            ybm_thickness = row['Young Bay Mud']
            ms_thickness = row['Merrit sand']
            fill_depth = fill_thickness
            ybm_depth = fill_depth + ybm_thickness  # Young Bay Mud goes from fill_depth to fill_depth + ybm_thickness
            ms_depth = ybm_depth + ms_thickness  # Merritt Sand goes from ybm_depth to ybm_depth + ms_thickness
            bh = row['Filename']
            cp_layers_dict[bh] = {'Fill': fill_thickness, 'Young Bay Mud': ybm_thickness, 'Merritt Sand': ms_thickness}
            if bh in cpt:
                results_k_values[bh] = {}
                fill_values = cpt[bh]['CPT_data'][(cpt[bh]['CPT_data']['d'] <= fill_depth)]['k_fromIc'].dropna()
                results_k_values[bh]['Fill'] = fill_values.mean() if not fill_values.empty else 0  # Default to 0 if no data points
                ybm_values = cpt[bh]['CPT_data'][(cpt[bh]['CPT_data']['d'] > fill_depth) & (cpt[bh]['CPT_data']['d'] <= ybm_depth)]['k_fromIc'].dropna()
                results_k_values[bh]['Young Bay Mud'] = ybm_values.mean() if not ybm_values.empty else 0  # Default to 0 if no data points
                ms_values = cpt[bh]['CPT_data'][(cpt[bh]['CPT_data']['d'] > ybm_depth) & (cpt[bh]['CPT_data']['d'] <= ms_depth)]['k_fromIc'].dropna()
                results_k_values[bh]['Merritt Sand'] = ms_values.mean() if not ms_values.empty else 0  # Default to 0 if no data points
                below_ms_values = cpt[bh]['CPT_data'][(cpt[bh]['CPT_data']['d'] > ms_depth)]['k_fromIc'].dropna()
                results_k_values[bh]['Below Merritt Sand'] = below_ms_values.mean() if not below_ms_values.empty else 0  # Default to 0 if no data points
        results_k_values = pd.DataFrame(results_k_values).transpose()
        results_k_m_day = results_k_values.apply(lambda x: x*86400)
        log_results_k_df = results_k_m_day.map(lambda x: np.log10(x) if x > 0 else np.nan) 
    elif method != 'heuristic':
        pen_0 = penalties_0[i]
        pen_1 = penalties_1[i]
        pen_2 = penalties_2[i]
        threshold1 = thresholds_1[i]
        threshold2 = thresholds_2[i]

        discard_count = 0
        cp_layers_dict = {}
        results_k_values = {}
        for bh in cpt:
            depth = cpt[bh]['CPT_data']['d']
            sbt = cpt[bh]['SBT']  # ['CPT_data'] && .values for corrected data

            # Detect change points
            if max(cpt[bh]['CPT_data']['d']) < threshold1:
                change_points = detect_change_points(sbt, model="l2", penalty=pen_0)[:-1]
            elif max(cpt[bh]['CPT_data']['d']) < threshold2:
                change_points = detect_change_points(sbt, model="l2", penalty=pen_1)[:-1]
            else:
                change_points = detect_change_points(sbt, model="l2", penalty=pen_2)[:-1]
                
            fill_flag = 1 if joined.loc[bh]['PTYPE'] in ['afem','H2O'] else 0

            # Calculate and plot the average k_fromIc between change points
            change_points = [0] + change_points + [len(cpt[bh]['CPT_data']['d']) - 1]
            depths = []
            Ks = []
            SBTs = []
            for cp_ind in range(len(change_points) - 1):
                start_idx = change_points[cp_ind]
                end_idx = change_points[cp_ind + 1]

                avg_k_fromIc = cpt[bh]['CPT_data']['k_fromIc'][start_idx:end_idx].mean()
                avg_SBT = stats.mode(cpt[bh]['SBT'][start_idx:end_idx])[0]  # here also ['CPT_data'] for corrected data
                start_depth = cpt[bh]['CPT_data']['d'][start_idx]
                end_depth = cpt[bh]['CPT_data']['d'][end_idx]
                SBTs.append(avg_SBT)
                depths.append(end_depth - start_depth)
                Ks.append(avg_k_fromIc)
            ## Naive method:
            if method == 'naive':  
                if fill_flag:
                    if len(depths) == 3:
                        cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': depths[1], 'Merritt Sand': depths[2]}
                    elif len(depths) >= 4:
                        cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': depths[1], 'Merritt Sand': depths[2], 'Below Merritt Sand': depths[3]}
                    elif len(depths) ==2:
                        cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': depths[1], 'Merritt Sand': np.nan}
                    else:
                        cp_layers_dict[bh] = {'Fill': depths[0],'Young Bay Mud': np.nan, 'Merritt Sand': np.nan} 
                else:
                    if len(depths) >=3:
                        cp_layers_dict[bh] = {'Fill': 0, 'Young Bay Mud': depths[0], 'Merritt Sand': depths[1], 'Below Merritt Sand': depths[2]}
                    elif len(depths) ==2:
                        cp_layers_dict[bh] = {'Fill': 0, 'Young Bay Mud': depths[0], 'Merritt Sand': depths[1]}
                    else:
                        cp_layers_dict[bh] = {'Fill': 0, 'Young Bay Mud': depths[0], 'Merritt Sand': np.nan}
            elif method == 'MS':
                if len(depths) == 3 + fill_flag:
                    if fill_flag: 
                        cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': depths[1], 'Merritt Sand': depths[2], 'Below Merritt Sand': depths[3]}
                    else:
                        cp_layers_dict[bh] = {'Fill':0, 'Young Bay Mud': depths[0], 'Merritt Sand': depths[1], 'Below Merritt Sand': depths[2]}
                elif len(depths) > 3 + fill_flag:
                    ## this is where we have more layers than expected
                    if fill_flag:
                        # sum of depths from fill to Merritt Sand
                        ybm_depth = sum(depths[1:np.where(np.array(SBTs) == 6)[0][-1]])
                        if len(depths) > np.where(np.array(SBTs) == 6)[0][-1]+1:
                            cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': ybm_depth, 'Merritt Sand': depths[np.where(np.array(SBTs) == 6)[0][-1]], 'Below Merritt Sand': depths[np.where(np.array(SBTs) == 6)[0][-1]+1]}
                        else:
                            cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': ybm_depth, 'Merritt Sand': depths[np.where(np.array(SBTs) == 6)[0][-1]]}
                    else:
                        ybm_depth = sum(depths[:np.where(np.array(SBTs) == 6)[0][-1]])
                        if len(depths) > np.where(np.array(SBTs) == 6)[0][-1]+1:
                            cp_layers_dict[bh] = {'Fill': 0, 'Young Bay Mud': ybm_depth, 'Merritt Sand': depths[np.where(np.array(SBTs) == 6)[0][-1]], 'Below Merritt Sand': depths[np.where(np.array(SBTs) == 6)[0][-1]+1]}
                        else:
                            cp_layers_dict[bh] = {'Fill': 0, 'Young Bay Mud': ybm_depth, 'Merritt Sand': depths[np.where(np.array(SBTs) == 6)[0][-1]]}
                elif len(depths) < 3 + fill_flag:
                    ## this is where we have fewer layers than expected
                    if 6 not in SBTs:
                        print('no sand in', bh)
                        if fill_flag:
                            cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:]), 'Merritt Sand': np.nan}
                        else:
                            cp_layers_dict[bh] = {'Fill': 0, 'Young Bay Mud': np.sum(depths), 'Merritt Sand': np.nan}
                    else:
                        MS_ind = np.where(np.array(SBTs) == 6)[0][-1]
                        if fill_flag:
                            if len(depths) <=2:
                                print('throwing out', bh)
                                discard_count += 1
                                continue
                            elif MS_ind == 0:
                                ## there is only one layer with SBT 6, assume it is fill
                                cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:]), 'Merritt Sand': np.nan}
                            else:
                                if len(depths) > MS_ind+1:
                                    cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:MS_ind]), 'Merritt Sand': depths[MS_ind], 'Below Merritt Sand': np.sum(depths[MS_ind+1:])}
                                else:
                                    cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:MS_ind]), 'Merritt Sand': depths[MS_ind]}
                        else:
                            if len(depths) <=1:
                                print('throwing out', bh)
                                discard_count += 1
                                continue
                            else:
                                if len(depths) > MS_ind+1:
                                    cp_layers_dict[bh] = {'Fill': 0, 'Young Bay Mud': np.sum(depths[:MS_ind]), 'Merritt Sand': depths[MS_ind], 'Below Merritt Sand': np.sum(depths[MS_ind+1:])}
                                else:
                                    cp_layers_dict[bh] = {'Fill': 0, 'Young Bay Mud': np.sum(depths[:MS_ind]), 'Merritt Sand': depths[MS_ind]}
            elif method == 'MS_AF':
                if len(depths) == 3 + fill_flag:
                    if fill_flag: 
                        if depths[0] <= 5:
                            cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': depths[1], 'Merritt Sand': depths[2], 'Below Merritt Sand': depths[3]}
                        else:
                            cp_layers_dict[bh] = {'Fill': np.nan, 'Young Bay Mud': depths[0], 'Merritt Sand': depths[1], 'Below Merritt Sand': depths[2]}
                    else:
                        cp_layers_dict[bh] = {'Fill':0, 'Young Bay Mud': depths[0], 'Merritt Sand': depths[1], 'Below Merritt Sand': depths[2]}
                if len(depths) > 3 + fill_flag:
                    ## we have more layers than expected
                    if fill_flag:
                        # sum of depths from fill to Merritt Sand
                        if len(depths) > np.where(np.array(SBTs) == 6)[0][-1]+1:
                            if depths[0] <= 5:
                                ybm_depth = sum(depths[1:np.where(np.array(SBTs) == 6)[0][-1]])
                                cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': ybm_depth, 'Merritt Sand': depths[np.where(np.array(SBTs) == 6)[0][-1]], 'Below Merritt Sand': depths[np.where(np.array(SBTs) == 6)[0][-1]+1]}
                            else:
                                ybm_depth = sum(depths[:np.where(np.array(SBTs) == 6)[0][-1]])
                                cp_layers_dict[bh] = {'Fill': np.nan, 'Young Bay Mud': ybm_depth, 'Merritt Sand': depths[np.where(np.array(SBTs) == 6)[0][-1]], 'Below Merritt Sand': depths[np.where(np.array(SBTs) == 6)[0][-1]+1]}
                        else:
                            if depths[0] <= 5:
                                ybm_depth = sum(depths[1:np.where(np.array(SBTs) == 6)[0][-1]])
                                cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': ybm_depth, 'Merritt Sand': depths[np.where(np.array(SBTs) == 6)[0][-1]]}
                            else:
                                ybm_depth = sum(depths[:np.where(np.array(SBTs) == 6)[0][-1]])
                                cp_layers_dict[bh] = {'Fill': np.nan, 'Young Bay Mud': ybm_depth, 'Merritt Sand': depths[np.where(np.array(SBTs) == 6)[0][-1]]}
                    else:
                        ybm_depth = sum(depths[:np.where(np.array(SBTs) == 6)[0][-1]])
                        if len(depths) > np.where(np.array(SBTs) == 6)[0][-1]+1:
                            cp_layers_dict[bh] = {'Fill': 0, 'Young Bay Mud': ybm_depth, 'Merritt Sand': depths[np.where(np.array(SBTs) == 6)[0][-1]], 'Below Merritt Sand': depths[np.where(np.array(SBTs) == 6)[0][-1]+1]}
                        else:
                            cp_layers_dict[bh] = {'Fill': 0, 'Young Bay Mud': ybm_depth, 'Merritt Sand': depths[np.where(np.array(SBTs) == 6)[0][-1]]}
                elif len(depths) < 3 + fill_flag:
                    ## case where we have fewer layers than expected
                    try:
                        MS_ind = np.where(np.array(SBTs) == 6)[0][-1]
                    except:
                        print('no sand in', bh)
                        continue
                    if fill_flag:
                        if len(depths) <=2:
                            print('throwing out', bh)
                            discard_count += 1
                            continue
                        elif MS_ind == 0:
                            ## there is only one layer with SBT 6, check its depth
                            if depths[0] <= 5:
                                cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:]),'Merritt Sand': np.nan}
                            else:
                                cp_layers_dict[bh] = {'Fill': np.nan, 'Young Bay Mud': 0, 'Merritt Sand': np.sum(depths[MS_ind]), 'Below Merritt Sand': np.sum(depths[MS_ind+1:])}
                        else:
                            if len(depths) > MS_ind+1:
                                cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:MS_ind]), 'Merritt Sand': depths[MS_ind], 'Below Merritt Sand': np.sum(depths[MS_ind+1:])}
                            else:
                                cp_layers_dict[bh] = {'Fill': depths[0], 'Young Bay Mud': np.sum(depths[1:MS_ind]), 'Merritt Sand': depths[MS_ind]}
                    else: ## not in fill
                        if len(depths) <=1:
                            print('throwing out', bh)
                            discard_count += 1
                            continue
                        else:
                            if len(depths) > MS_ind+1:
                                cp_layers_dict[bh] = {'Fill': 0, 'Young Bay Mud': np.sum(depths[:MS_ind]), 'Merritt Sand': depths[MS_ind], 'Below Merritt Sand': np.sum(depths[MS_ind+1:])}
                            else:
                                cp_layers_dict[bh] = {'Fill': 0, 'Young Bay Mud': np.sum(depths[:MS_ind]), 'Merritt Sand': depths[MS_ind]}
            
            ## still within borehole
            fill_thickness = cp_layers_dict[bh]['Fill']
            ybm_thickness = cp_layers_dict[bh]['Young Bay Mud']
            ms_thickness = cp_layers_dict[bh]['Merritt Sand']
            fill_depth = fill_thickness  # Fill layer goes from 0 to fill_thickness
            ybm_depth = fill_depth + ybm_thickness  # Young Bay Mud goes from fill_depth to fill_depth + ybm_thickness
            ms_depth = ybm_depth + ms_thickness  # Merritt Sand goes from ybm_depth to ybm_depth + ms_thickness
            
            # Calculate average k_fromIc for each layer, excluding NaN values
            results_k_values[bh] = {}
            # Fill layer
            fill_values = cpt[bh]['CPT_data'][(cpt[bh]['CPT_data']['d'] <= fill_depth)]['k_fromIc'].dropna()
            results_k_values[bh]['Fill'] = fill_values.mean() if not fill_values.empty else 0  # Default to 0 if no data points
            # Young Bay Mud layer
            ybm_values = cpt[bh]['CPT_data'][(cpt[bh]['CPT_data']['d'] > fill_depth) & (cpt[bh]['CPT_data']['d'] <= ybm_depth)]['k_fromIc'].dropna()
            results_k_values[bh]['Young Bay Mud'] = ybm_values.mean() if not ybm_values.empty else 0  # Default to 0 if no data points
            # Merritt Sand layer
            ms_values = cpt[bh]['CPT_data'][(cpt[bh]['CPT_data']['d'] > ybm_depth) & (cpt[bh]['CPT_data']['d'] <= ms_depth)]['k_fromIc'].dropna()
            results_k_values[bh]['Merritt Sand'] = ms_values.mean() if not ms_values.empty else 0  # Default to 0 if no data points
            # Below Merritt Sand
            below_ms_values = cpt[bh]['CPT_data'][(cpt[bh]['CPT_data']['d'] > ms_depth)]['k_fromIc'].dropna()
            results_k_values[bh]['Below Merritt Sand'] = below_ms_values.mean() if not below_ms_values.empty else 0  # Default to 0 if no data points
            # Transform data to log scale
        results_k_values = pd.DataFrame(results_k_values).transpose()
        results_k_m_day = results_k_values.apply(lambda x: x*86400)
        log_results_k_df = results_k_m_day.applymap(lambda x: np.log10(x) if x > 0 else np.nan)
    ## back here will be all cases (heuristic, naive, MS, MS_AF)
    if K_source[i] == 'table':
        save_params.loc[names[i],'hk_m_day_0'] = 1e0
        save_params.loc[names[i],'hk_m_day_1'] = 10**(-1.5)
        save_params.loc[names[i],'hk_m_day_2'] = 1e0
        save_params.loc[names[i],'hk_m_day_3'] = 10**(-1.5)
    elif K_source[i] == 'low':
        save_params.loc[names[i],'hk_m_day_0'] = 1e-2
        save_params.loc[names[i],'hk_m_day_1'] = 1e-3
        save_params.loc[names[i],'hk_m_day_2'] = 1e-2
        save_params.loc[names[i],'hk_m_day_3'] = 1e-3
    elif K_source[i] == 'high':
        save_params.loc[names[i],'hk_m_day_0'] = 1e2
        save_params.loc[names[i],'hk_m_day_1'] = 1e0
        save_params.loc[names[i],'hk_m_day_2'] = 1e2
        save_params.loc[names[i],'hk_m_day_3'] = 1e0
    elif K_source[i] == 'empirical':
        layer_names = ['Fill', 'Young Bay Mud', 'Merritt Sand', 'Below Merritt Sand']
        for ind, layer in enumerate(layer_names):
            layer_data = log_results_k_df[layer].dropna()
            save_params.loc[names[i],'hk_m_day_'+str(ind)] = 10**layer_data.mean()  # ensure that units are m/day
    else:
        print('Invalid K_source')
    
    cp_layers_df = pd.DataFrame.from_dict(cp_layers_dict, orient='index')
    ## This is the spatial portion
    if method == 'heuristic':
        ## get Lat and Lon from utm values in layers
        cp_layers_df['Lat'], cp_layers_df['Lon'] = zip(*cp_layers_df.index.map(
            lambda x: utm.to_latlon(
                layers.loc[layers['Filename'] == x, 'UTM -X'].values[0],
                layers.loc[layers['Filename'] == x, 'UTM -Y'].values[0],
                10, 'N')))    
    else:
        cp_layers_df['Lat'] = cp_layers_df.index.map(lambda x: cpt[x]['Lat'])
        cp_layers_df['Lon'] = cp_layers_df.index.map(lambda x: cpt[x]['Lon'])
    gdf_layers_naive = gpd.GeoDataFrame(cp_layers_df, geometry=gpd.points_from_xy(cp_layers_df['Lon'], cp_layers_df['Lat']), crs='EPSG:4326')
    gdf_layers_naive = gpd.sjoin(gdf_layers_naive,data , how='left')
    gdf_layers_naive.loc[~gdf_layers_naive['PTYPE'].isin(['afem']),'Fill'] = 0  ## water also has 0 fill, but measurements are not in water, so they should really be nonzero

    minx, miny, maxx, maxy = gdf_layers_naive.total_bounds
    spacing = 0.008  # Define spacing between grid points for extra zeros outside fill
    x_coords = np.arange(minx, maxx, spacing)
    y_coords = np.arange(miny, maxy, spacing)
    grid_points = [Point(x, y) for x in x_coords for y in y_coords]
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs=gdf_layers_naive.crs)
    filtered_gdf = grid_gdf[~grid_gdf.sjoin(data[data['PTYPE'].isin(['H2O','afem'])], how='left', predicate='intersects').index_right.notnull()]  # only add in 0 in middle of island, impose 0 to water after
    filtered_gdf['Fill'] = 0
    geo_layers_append = pd.concat([gdf_layers_naive, filtered_gdf], ignore_index=True)
    geo_layers_append[['UTM -X', 'UTM -Y']] = geo_layers_append['geometry'].apply( lambda point: pd.Series(utm.from_latlon(point.y, point.x)[:2]))
    ## here add in the borehole data
    geo_layers_append = pd.concat([geo_layers_append, lith_gdf[['UTM -X', 'UTM -Y', 'Fill']]], axis=0)
    geo_layers_append['UTM -X'] = pd.to_numeric(geo_layers_append['UTM -X'], errors='coerce')
    geo_layers_append['UTM -Y'] = pd.to_numeric(geo_layers_append['UTM -Y'], errors='coerce')
    points['utmX'] = pd.to_numeric(points['utmX'], errors='coerce')
    points['utmY'] = pd.to_numeric(points['utmY'], errors='coerce')
    points = points.dropna(subset=['utmX', 'utmY'])

    kriged_depths = {}
    for depth_type in ['Fill', 'Young Bay Mud', 'Merritt Sand']:
        # Ensure depth values are numeric
        geo_layers_append[depth_type] = pd.to_numeric(geo_layers_append[depth_type], errors='coerce')
        
        valid_layers = geo_layers_append.dropna(subset=['UTM -X', 'UTM -Y', depth_type])
        x_coords = valid_layers['UTM -X'].values.astype(float)
        y_coords = valid_layers['UTM -Y'].values.astype(float)
        z_values = valid_layers[depth_type].values.astype(float)
        
        if len(x_coords) > 0 and len(y_coords) > 0 and len(z_values) > 0:
            kriging_model = OrdinaryKriging(
                x_coords, y_coords, z_values,
                variogram_model='exponential', 
                verbose=False,
                nlags = 20, 
                enable_plotting=False
            )
            z_interp, _ = kriging_model.execute(
                'points', points['utmX'].values.astype(float), 
                points['utmY'].values.astype(float))
            
            kriged_depths[depth_type] = z_interp.clip(0)

    # Add kriged results to points DataFrame
    for depth_type, values in kriged_depths.items():
        points[depth_type] = values

    points_gdf = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points['lon'], points['lat']), crs="EPSG:4326")
    # join points_gdf with data to get PTYPE and force fill to 0 in water
    points_gdf = gpd.sjoin(points_gdf, data, how='left')
    points_gdf.loc[points_gdf['PTYPE'].isin(['H2O']),'Fill'] = 0  ## forcing no fill in the Bay

    points_gdf['Yerba Buena Mud'] = 50 - points_gdf['Fill'] - points_gdf['Young Bay Mud'] - points_gdf['Merritt Sand']
    points_gdf.to_crs('EPSG:32610', inplace=True)
    crs = "EPSG:32610"

    layer_names = ['Fill', 'Young Bay Mud', 'Merritt Sand', 'Yerba Buena Mud']
    for lay_ind in range(len(layer_names)):
        layer = layer_names[lay_ind]
        create_geotiff(points_gdf, layer, f"./out_geotiffs/layer{lay_ind}_thick_m_{names[i]}.tif")
        create_geotiff(points_gdf, save_params.loc[names[i],f'hk_m_day_{lay_ind}'], f"./out_geotiffs/layer{lay_ind}_hK_mday_{names[i]}.tif")
