import src as vawf
import datetime
import xarray as xr
import time
import numpy as np
import os
import json

selected_ts = 2
# ------------------------------------------------------------
#                10m Winds
#-------------------------------------------------------------
# Define the local path to store 10-meter wind data
winds10m_file = './data/winds10m.grib2'
# Download 10-meter wind data from the data source
vawf.downloader.download_10m_winds(winds10m_file)
# Open wind data
ifsdata = xr.open_dataset(winds10m_file,engine='cfgrib')
data_date = ifsdata['valid_time'].isel(step=selected_ts).values\
    .astype('datetime64[s]').astype(datetime.datetime).strftime("%B %d, %Y")
data_from = ifsdata.attrs['institution']
data_at = ifsdata.attrs['history'][:16]
# Preprocess wind fields
wind10 = vawf.utilities.preprocess_ifsdata(ifsdata)

# Extract Storms
trackedStorms = vawf.detector.detect_storms_scafet(wind10)
storms = trackedStorms.to_pandas()

# Save storm properties to JSON file for Visulization
ts = np.unique(trackedStorms['timestamp'])
stormProps = storms[storms['timestamp']==ts[selected_ts]]
vawf.utilities.save_storms_json(stormProps, data_from, data_at, data_date,\
                        './outputs/storm_properties.json')

#--------------Plot Wind Speeds---------------------
vawf.plotter.plot_10m_winds(ifsdata.isel(step=selected_ts))
os.remove(winds10m_file)

# ------------------------------------------------------------
#                Upper Level Winds
#-------------------------------------------------------------
# Define the local path to store upper-level wind data
ulwinds_file = './data/ulwinds.grib2'
# Download upper-level wind data from the data source
vawf.downloader.download_upper_level_winds(ulwinds_file)

ifsdata = xr.open_dataset(ulwinds_file)
ifsdata = ifsdata.mean('isobaricInhPa')

# Preprocess wind fields
ulwind = vawf.utilities.preprocess_ifsdata(ifsdata)
# Dectect Jetstreams
jetstreams = vawf.detector.detect_upper_level_jets(ulwind)

# Extract central lines from 2D jet masks
jet_mask = (jetstreams[1]['object'].where(jetstreams[1]['object']>0)*0 +1).fillna(0).astype(bool)
jet_mask = vawf.utilities.rotate_to_180(jet_mask)
jet_lines = vawf.utilities.extract_corelines(jet_mask.isel(time=selected_ts),
                                        min_size=10, smooth_sigma=20, pad=60)
# Save core lines for plotting
vawf.utilities.save_jetlines_geojson(jet_lines, filename="./outputs/jet_lines.geojson")

# Plot the upper level winds using the plotter module
vawf.plotter.plot_upper_winds(ifsdata.isel(step=selected_ts))
os.remove(ulwinds_file)

# ------------------------------------------------------------
#                Precipitation
#-------------------------------------------------------------
# Define the local path to store total precipitation data
precip_file = './data/precip.grib2'
# Download total precipitation data from the data source
vawf.downloader.download_precipitation(precip_file)

ifsdata = xr.open_dataset(precip_file)
precip_data = vawf.utilities.preprocess_ifsdata(ifsdata)

# Plot the precipitation using the plotter module
vawf.plotter.plot_precipitation(ifsdata.isel(step=selected_ts))

# ------------------------------------------------------------
#              Total Column Water Vapor
#-------------------------------------------------------------
# Define the local path to store total column water vapor data
tcwv_file = './data/totcolwv.grib2'
# Download total column water vapor (TCWV) data from the data source
vawf.downloader.download_total_column_wv(tcwv_file)

ifsdata = xr.open_dataset(tcwv_file)
tcw = vawf.utilities.preprocess_ifsdata(ifsdata)

# Plot the total column using the plotter module
vawf.plotter.plot_totcolumn_wv(ifsdata.isel(step=selected_ts))

# ------------------------------------------------------------
#              Atmospheric River Detection
#-------------------------------------------------------------
atmosp_rivers = vawf.detector.detect_atmospheric_rivers(tcw, precip_data)
# Extract central lines from 2D jet masks
ar_mask = (atmosp_rivers[1].where(atmosp_rivers[1]>0)*0 +1).fillna(0).astype(bool)
ar_mask = vawf.utilities.rotate_to_180(ar_mask)
ar_lines = vawf.utilities.extract_corelines(ar_mask.isel(time=selected_ts),
                                        min_size=200, smooth_sigma=20, pad=60)

# Save core lines for plotting
vawf.utilities.save_jetlines_geojson(ar_lines, filename="./outputs/ar_lines.geojson")

# os.remove(tcwv_file)
# os.remove(precip_file)
