# Modules required for computation
import xarray as xr
import numpy as np
from tqdm import tqdm
import cfgrib
import time

def rotate_to_360(ds):
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

def rotate_to_180(ds):
    """
    Rotates longitude coordinates of an xarray Dataset from [0, 360] to [-180, 180] and sorts them.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with longitude coordinates in the [0, 360] range.

    Returns
    -------
    xarray.Dataset
        Dataset with longitudes converted to [-180, 180] range and sorted.
    """
    lon = ds['lon'].values

    # Convert longitudes >180° to their equivalent in [-180, 180]
    lon = xr.where(lon > 180, lon - 360, lon)

    ds = ds.assign_coords({'lon': lon})
    return ds.sortby('lon')

def save_jetlines_geojson(jet_lines, filename="jet_lines.geojson"):
    """
    Save jet corelines as a GeoJSON FeatureCollection of LineStrings,
    suitable for visualization in Three.js or GIS tools.
    """
    features = []
    for idx, line in enumerate(jet_lines, start=1):
        coords = list(zip(line["lon"], line["lat"]))
        features.append({
            "type": "Feature",
            "id": idx,
            "properties": {
                "jet_id": idx,
                "area": line.get("area", None)  # optional attribute
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(filename, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"✅ Saved {len(features)} jet lines to {filename}")

def haversine(lon1, lat1, lon2, lat2):
    """Distance in km between two points given in degrees."""
    R = 6371.0  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def extend_grid(mask, lons, lats, pad=30):
    mask_ext = np.concatenate([mask[:, -pad:], mask, mask[:, :pad]], axis=1)
    lons_ext = np.concatenate([lons[:, -pad:] - 360, lons, lons[:, :pad] + 360], axis=1)
    lats_ext = np.concatenate([lats[:, -pad:], lats, lats[:, :pad]], axis=1)
    return mask_ext, lons_ext, lats_ext, pad

def order_skeleton_points_periodic(skel_ext, region_coords_ext):
    local_mask = np.zeros_like(skel_ext, dtype=bool)
    local_mask[tuple(region_coords_ext.T)] = True
    kernel = np.ones((3,3))
    neighbor_count = convolve(local_mask.astype(int), kernel, mode='wrap')
    endpoints = np.argwhere(local_mask & (neighbor_count==2))
    if len(endpoints)<2:
        return region_coords_ext
    start, end = tuple(endpoints[0]), tuple(endpoints[-1])
    cost = np.where(local_mask, 1.0, np.inf)
    try:
        path,_ = route_through_array(cost, start, end, fully_connected=True)
        return np.array(path)
    except Exception:
        return region_coords_ext

def smooth_path_arc_length(lon_deg, lat_deg, sigma=10, n_points=None):
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    dx = np.diff(lon)
    dy = np.diff(lat)
    dx = (dx + 180) % 360 - 180
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if n_points is None:
        n_points = len(lon)
    s_uniform = np.linspace(0, s[-1], n_points)
    lon_interp = np.interp(s_uniform, s, lon)
    lat_interp = np.interp(s_uniform, s, lat)
    lon_smooth = gaussian_filter1d(lon_interp, sigma=sigma)
    lat_smooth = gaussian_filter1d(lat_interp, sigma=sigma)
    return lon_smooth, lat_smooth

def extract_corelines(mask, lons, lats, min_size=200, smooth_sigma=10, pad=40):
    mask_ext, lons_ext, lats_ext, offset = extend_grid(mask.astype(bool), lons, lats, pad=pad)
    mask_clean = remove_small_objects(mask_ext, min_size=min_size)
    skeleton_ext = skeletonize(mask_clean)
    labeled_ext = label(skeleton_ext, connectivity=2)
    regions = regionprops(labeled_ext)

    jet_lines = []

    for region in regions:
        region_coords_ext = region.coords
        ordered_ext = order_skeleton_points_periodic(skeleton_ext, region_coords_ext)
        i_ext = np.clip(ordered_ext[:,0],0,lons_ext.shape[0]-1).astype(int)
        j_ext = np.clip(ordered_ext[:,1],0,lons_ext.shape[1]-1).astype(int)
        lon_ext = lons_ext[i_ext, j_ext]
        lat_ext = lats_ext[i_ext, j_ext]
        lon_cont = make_continuous_lon(lon_ext)
        lon_smooth, lat_smooth = smooth_path_arc_length(lon_cont, lat_ext, sigma=smooth_sigma, n_points=len(lon_cont))

        # -----------------------
        # Clip longitudes to [-180, 180]
        # -----------------------
        mask_valid = (lon_smooth >= -180) & (lon_smooth <= 180)
        lon_smooth = lon_smooth[mask_valid]
        lat_smooth = lat_smooth[mask_valid]

        jet_lines.append({"lon": lon_smooth, "lat": lat_smooth, "area": region.area})

    return jet_lines

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
    ifsdata_rotated = rotate_to_360(ifsdata)

    # Rename 'time' to 'forecast-time' (actual forecast issue time)
    ifsdata_rotated = ifsdata_rotated.rename({'time': 'forecast-time'})

    # Rename 'step' to 'time' (lead time becomes the main time coordinate)
    ifsdata_rotated = ifsdata_rotated.rename({'step': 'time'}).load()

    return ifsdata_rotated