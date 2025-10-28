# Modules required for computation
import xarray as xr
import numpy as np
from tqdm import tqdm
import cfgrib
import time
from ecmwf.opendata import Client
import metpy.calc as mpcalc
import src as vawf

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
    
    properties = vawf.scafet.properties.object_properties2D(grid_area,\
                        land_mask,min_length,min_area,\
                        smooth_scale,angle_threshold,min_duration,max_distance_per_tstep,\
                        shape_index,shape_eccentricity,lon_mask,lat_mask)

    stime = time.time()
    sdetect = vawf.scafet.shape_analysis.shapeDetector(cyc)
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
    cycfilter = vawf.scafet.filtering.filterObjects(ridges)
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
    
    tracker = vawf.scafet.tracking.Tracker(latlon,properties)
    tracked = tracker.apply_tracking(object_properties,object_masks)
    print('Finished Tracking in {} seconds'.format(time.time()-stime))

    return tracked[0]

def detect_upper_level_jets(ulwind):
    grid_area = xr.open_dataset('resources/grid_area_era5.nc')
        # .sel(latitude=latslice,longitude=lonslice)
    grid_area = grid_area.rename({'longitude':'lon'})
    grid_area = grid_area.rename({'latitude':'lat'})
    grid_area = grid_area.reindex\
        (lat=list(reversed(grid_area.lat)))
    land_mask = xr.open_dataset('resources/land_sea_mask_era5.nc')

    smooth_scale = 2.5e6
    angle_threshold = 60
    shape_index = [0.375,1]
    min_length = 2000e3
    min_area = 2e12
    min_duration = 6
    max_distance_per_tstep = 3000e3
    shape_eccentricity = [0.45,1.0]
    lat_mask = [-0,0]
    lon_mask = [360,0]

    properties = vawf.scafet.properties.object_properties2D(grid_area,\
                        land_mask,min_length,min_area,\
                        smooth_scale,angle_threshold,min_duration,max_distance_per_tstep,\
                        shape_index,shape_eccentricity,lon_mask,lat_mask)

    stime = time.time()
    sdetect = vawf.scafet.shape_analysis.shapeDetector(ulwind)
    smoothed = sdetect.apply_smoother(ulwind,properties)
    print('Finished smoothing in {} seconds'.format(time.time()-stime))

    windspeed = vawf.scafet.shape_analysis.calc_magnitude(ulwind.u,ulwind.v)
    stime = time.time()
    # Detect Ridges
    ridges = sdetect.apply_shape_detection(smoothed,properties)
    print('Finished shape extraction in {} seconds'.format(time.time()-stime))

    props_mag = windspeed.to_dataset(name='mag')

    stime = time.time()
    jetfilter = vawf.scafet.filtering.filterObjects(ridges)
    filtered = jetfilter.apply_filter(ridges,\
                props_mag,['mean_intensity'],
                [20],properties,'ridges')
    print('Finished Filtering in {} seconds'.format(time.time()-stime))
    return filtered

def detect_atmospheric_rivers(tcw, precip):
    grid_area = xr.open_dataset('resources/grid_area_era5.nc')
        # .sel(latitude=latslice,longitude=lonslice)
    grid_area = grid_area.rename({'longitude':'lon'})
    grid_area = grid_area.rename({'latitude':'lat'})
    grid_area = grid_area.reindex\
        (lat=list(reversed(grid_area.lat)))
    land_mask = xr.open_dataset('resources/land_sea_mask_era5.nc')

    smooth_scale = 2e6
    angle_threshold = 60
    shape_index = [0.375,1]
    min_length = 2000e3
    min_area = 1e12
    min_duration = 1
    max_distance_per_tstep = 4000e3
    shape_eccentricity = [0.75,1.0]
    lat_mask = [-15,15]
    lon_mask = [360,0]
    
    properties = vawf.scafet.properties.object_properties2D(grid_area,\
                        land_mask,min_length,min_area,\
                        smooth_scale,angle_threshold,min_duration,max_distance_per_tstep,\
                        shape_index,shape_eccentricity,lon_mask,lat_mask)

    stime = time.time()
    sdetect = vawf.scafet.shape_analysis.shapeDetector(tcw)
    smoothed = sdetect.apply_smoother(tcw,properties)
    print('Finished smoothing in {} seconds'.format(time.time()-stime))
    smoothed = smoothed.rename({'tcw':'mag'})

    stime = time.time()
    # Detect Ridges
    ridges = sdetect.apply_shape_detection(smoothed,properties)
    print('Finished shape extraction in {} seconds'.format(time.time()-stime))

    property_fields = xr.concat([tcw['tcw'].expand_dims('Channel'),precip['tp']],\
                        dim='Channel').to_dataset(name='mag')

    stime = time.time()
    arfilter = vawf.scafet.filtering.filterObjects(ridges)
    filtered = arfilter.apply_filter(ridges,\
                property_fields,['mean_intensity','mean_intensity'],
                [5,1],properties,'ridges')
    print('Finished Filtering in {} seconds'.format(time.time()-stime))

    properties.obj['Min_Duration']= min_duration
    properties.obj['Max_Distance']= max_distance_per_tstep/1e3
    
    # Tracking based on centroid of each object
    latlon = ['wclat','wclon']
    object_masks = filtered[1]
    object_properties = filtered[0]
    tracker = vawf.scafet.tracking.Tracker(latlon,properties)
    tracked = tracker.apply_tracking(object_properties,object_masks)
    print('Finished Tracking in {} seconds'.format(time.time()-stime))

    return tracked