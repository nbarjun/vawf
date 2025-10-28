import xarray as xr
import numpy as np

# Packages for Plotting
from PIL import Image
import matplotlib.pyplot as plt
from cmcrameri import cm as ccm
from matplotlib.colors import LogNorm
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, FuncFormatter

def plot_precipitation(ds):
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 20})
    
    precip = ds['tp']
    
    # Set min and max for log scale
    vmin, vmax = .05, 150
    precip_clipped = np.clip(precip, vmin, vmax)  # avoid log(0)
    
    # Apply log normalization
    norm = LogNorm(vmin=vmin, vmax=vmax)
    precip_norm = norm(precip_clipped) 
    
    # Pick a colormap
    cmap = plt.get_cmap("viridis")
    rgba_img = (cmap(precip_norm) * 255).astype(np.uint8)  # RGBA array
    
    # Save as PNG for Three.js
    Image.fromarray(rgba_img).save("./outputs/precipitation.png")
    
    # Make Colorbar for Precipitation
    fig, ax = plt.subplots(figsize=(.25, 9),dpi=300)
    
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='vertical',
        label="Precipitation (mm)"
    )
    
    # Set tick locations explicitly in actual values
    tick_values = [0.1, 0.3, 1, 3, 10, 30, 100]
    cb.set_ticks(tick_values)
    
    # Use a formatter to show them as normal numbers
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    
    plt.savefig("./outputs/colorbar_precip.png", dpi=300,\
                transparent=True, bbox_inches= 'tight')

    # Make Colorbar for wind
    fig, ax = plt.subplots(figsize=(7, .3),dpi=300)
    
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal',
        label="Precipitation (mm)"
    )

     # Set tick locations explicitly in actual values
    tick_values = [0.1, 0.3, 1, 3, 10, 30, 100]
    cb.set_ticks(tick_values)
    
    plt.savefig("./outputs/colorbar_precip_p.png", dpi=300,\
            transparent=True, bbox_inches= 'tight')
    return None

def plot_10m_winds(ds):
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 20})

    u = ds["u10"].squeeze()#.isel(time=0)  # zonal wind
    v = ds["v10"].squeeze()#.isel(time=0)  # meridional wind
    speed = np.sqrt(u**2 + v**2)

    # Normalize wind data
    vmin, vmax = 1, 20
    speed_norm = np.clip((speed - vmin) / (vmax - vmin), 0, 1)

    # Pick a colormap
    cmap = plt.get_cmap("inferno")
    rgba_img = (cmap(speed_norm) * 255).astype(np.uint8)  # RGBA array
    
    # Save as PNG for Three.js
    Image.fromarray(rgba_img).save("./outputs/windspeeds_10m.png")

    
    # Make Colorbar for wind
    fig, ax = plt.subplots(figsize=(.25, 9),dpi=300)
    
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='vertical',
        label="Wind speed (m/s)"
    )

    plt.savefig("./outputs/colorbar_wind_10m.png", dpi=300,\
            transparent=True, bbox_inches= 'tight')

    # Make Colorbar for wind
    fig, ax = plt.subplots(figsize=(7, .3),dpi=300)
    
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal',
        label="Wind speed (m/s)"
    )

    plt.savefig("./outputs/colorbar_wind_10m_p.png", dpi=300,\
            transparent=True, bbox_inches= 'tight')
    return None

def plot_upper_winds(ds):
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 20})

    u = ds["u"].squeeze()
    v = ds["v"].squeeze()
    speed = np.sqrt(u**2 + v**2)

    # Normalize wind data
    vmin, vmax = 5, 60
    speed_norm = np.clip((speed - vmin) / (vmax - vmin), 0, 1)

    # Pick a colormap
    cmap = plt.get_cmap("inferno")
    rgba_img = (cmap(speed_norm) * 255).astype(np.uint8)  # RGBA array
    
    # Save as PNG for Three.js
    Image.fromarray(rgba_img).save("./outputs/windspeeds_ul.png")

    
    # Make Colorbar for wind
    fig, ax = plt.subplots(figsize=(.25, 9),dpi=300)
    
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='vertical',
        label="Wind speed (m/s)"
    )

    plt.savefig("./outputs/colorbar_wind_ul.png", dpi=300,\
            transparent=True, bbox_inches= 'tight')

    # Make Colorbar for wind
    fig, ax = plt.subplots(figsize=(7, .3),dpi=300)
    
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal',
        label="Wind speed (m/s)"
    )

    plt.savefig("./outputs/colorbar_wind_ul_p.png", dpi=300,\
            transparent=True, bbox_inches= 'tight')
    return None

def plot_totcolumn_wv(ds):
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 20})
    
    tcw = ds['tcw']
    # Normalize wind data
    vmin, vmax = 5, 75
    tcw_norm = np.clip((tcw - vmin) / (vmax - vmin), 0, 1)

    # Pick a colormap
    cmap = plt.get_cmap(ccm.devon)
    rgba_img = (cmap(tcw_norm) * 255).astype(np.uint8)  # RGBA array
    
    # Save as PNG for Three.js
    Image.fromarray(rgba_img).save("./outputs/totcol_water.png")

    
    # Make Colorbar for wind
    fig, ax = plt.subplots(figsize=(.25, 9),dpi=300)
    
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='vertical',
        label="Total Column Water ($kg m^{-2}$)"
    )

    plt.savefig("./outputs/colorbar_tcw.png", dpi=300,\
            transparent=True, bbox_inches= 'tight')

    # Make Colorbar for wind
    fig, ax = plt.subplots(figsize=(7, .3),dpi=300)
    
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal',
        label="Total Column Water ($kg m^{-2}$)"
    )

    plt.savefig("./outputs/colorbar_tcw_p.png", dpi=300,\
            transparent=True, bbox_inches= 'tight')
    return None