# Modules required for computation
import xarray as xr
import numpy as np
from tqdm import tqdm
import cfgrib
import time
from ecmwf.opendata import Client
import metpy.calc as mpcalc
from . import scafet as storm

# Modules required for plotting
import folium
from folium.plugins import Fullscreen
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors

def rotate180(ds):
    """
    Rotates longitude coordinates of an xarray Dataset from [-180, 180] to [0, 360] and sorts them.

    Parameters:
    - ds: xarray.Dataset
        Input dataset with longitude coordinates possibly in the [-180, 180] range.

    Returns:
    - xarray.Dataset
        Dataset with longitudes converted to [0, 360] range and sorted.
    """
    
    # Extract longitude values from the dataset
    lon = ds['lon'].values

    # Convert negative longitudes (e.g., -170) to their equivalent in [0, 360] (e.g., 190)
    lon = xr.where(lon < 0, lon + 360, lon)

    # Extract latitude values (not used here, but extracted — can be removed if unused)
    lat = ds['lat'].values

    # Assign the updated longitudes back to the dataset
    ds = ds.assign_coords({'lon': lon})

    # Sort the dataset along the longitude dimension to ensure longitudes are in increasing order
    return ds.sortby('lon')

def preprocess_ifsdata(ds):
    """
    Preprocesses IFS forecast data to prepare it for analysis.
    
    Steps:
    - Rename latitude and longitude dimensions to 'lat' and 'lon'
    - Flip latitude to go from north to south (if needed)
    - Rotate longitudes from [-180, 180] to [0, 360]
    - Rename time-related dimensions for clarity
    - Load the dataset into memory

    Parameters:
    - ds: xarray.Dataset
        Raw IFS dataset

    Returns:
    - ifsdata_rotated: xarray.Dataset
        Preprocessed and standardized dataset
    """
    # Rename longitude and latitude to standard names
    ifsdata = ds.rename({'longitude': 'lon', 'latitude': 'lat'})

    # Reverse the latitude ordering (from south to north → north to south)
    ifsdata = ifsdata.reindex(lat=list(reversed(ifsdata.lat)))

    # Rotate longitudes from [-180, 180] to [0, 360] and sort
    ifsdata_rotated = rotate180(ifsdata)

    # Rename 'time' to 'forecast-time' (actual forecast issue time)
    ifsdata_rotated = ifsdata_rotated.rename({'time': 'forecast-time'})

    # Rename 'step' to 'time' (lead time becomes the main time coordinate)
    ifsdata_rotated = ifsdata_rotated.rename({'step': 'time'}).load()

    return ifsdata_rotated

def detect_storms_scafet(wind):
    grid_area = xr.open_dataset('resources/grid_area_era5.nc')
            # .sel(latitude=latslice,longitude=lonslice)
    grid_area = grid_area.rename({'longitude':'lon'})
    grid_area = grid_area.rename({'latitude':'lat'})
    grid_area = grid_area.reindex\
        (lat=list(reversed(grid_area.lat)))
    land_mask = xr.open_dataset('resources/land_sea_mask_era5.nc')

    relVor = mpcalc.vorticity(wind['u10'], wind['v10'])
    rv = relVor.metpy.dequantify().to_dataset(name='rv')
    cyc = rv*np.sign(rv.lat)

    smooth_scale = 2e6
    angle_threshold = 45
    shape_index = [0.625,1]
    min_length = 20e3
    min_area = 2e11
    min_duration = 2
    max_distance_per_tstep = 1000e3
    shape_eccentricity = [0.0,1.0]
    lat_mask = [-0,0]
    lon_mask = [360,0]
    
    properties = storm.properties.object_properties2D(grid_area,\
                        land_mask,min_length,min_area,\
                        smooth_scale,angle_threshold,min_duration,max_distance_per_tstep,\
                        shape_index,shape_eccentricity,lon_mask,lat_mask)

    stime = time.time()
    sdetect = storm.shape_analysis.shapeDetector(cyc)
    vor = sdetect.apply_smoother(cyc,properties)
    print('Finished smoothing in {} seconds'.format(time.time()-stime))

    vor = vor.rename({'rv':'mag'})
    cyc = cyc.rename({'rv':'mag'})
    stime = time.time()
    # Select only positive values of cyclonic vorticity
    cyc_sm = vor.where((vor.mag>0)).fillna(0)
    # Detect Ridges
    ridges = sdetect.apply_shape_detection(cyc_sm,properties)
    print('Finished shape extraction in {} seconds'.format(time.time()-stime))

    # Use unsmoothed vorticity as primary field 
    cyc_us = cyc.where((cyc.mag>0)).fillna(0)
    # Define the secondary field as wind speed
    ws = np.sqrt(wind['u10']**2+wind['v10']**2)
    props_mag = xr.concat([ws.expand_dims('Channel'),\
                cyc_us.mag.expand_dims('Channel')], dim='Channel')
    props_mag = props_mag.to_dataset(name='mag')
    
    stime = time.time()
    cycfilter = storm.filtering.filterObjects(ridges)
    filtered = cycfilter.apply_filter(ridges,\
                props_mag,['max_intensity','mean_intensity'],
                [10,3e-5],properties,'ridges')
    print('Finished Filtering in {} seconds'.format(time.time()-stime))

    object_masks = filtered[1]
    object_properties = filtered[0]
    
    stime = time.time()
    # Tracking
    properties.obj['Min_Duration']= min_duration
    properties.obj['Max_Distance']= max_distance_per_tstep/1e3
    
    # Tracking based on centroid of each object
    latlon = ['wclat','wclon']
    
    tracker = storm.tracking.Tracker(latlon,properties)
    tracked = tracker.apply_tracking(object_properties,object_masks)
    print('Finished Tracking in {} seconds'.format(time.time()-stime))

    return tracked[0]