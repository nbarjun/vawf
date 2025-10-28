# Modules required for computation
import xarray as xr
import numpy as np
from tqdm import tqdm
import cfgrib
import time
from ecmwf.opendata import Client

def download_10m_winds(fname):
    """
    Downloads the latest IFS (Integrated Forecasting System) forecast data from ECMWF.

    Parameters:
    - fname: str
        Target filename where the downloaded GRIB data will be saved.

    Returns:
    - data: the response object from the data retrieval client.
    """
    # Initialize the client for ECMWF IFS model data
    client = Client(
        source="ecmwf",          # Data source is ECMWF
        model="aifs-single",     # Single forecast runs produced by the ECMWF Artificial Intelligence Forecasting System
        resol="0p25",            # Horizontal resolution of 0.25 degrees
        preserve_request_order=False,  # Reorder requests for performance (optional)
        infer_stream_keyword=True      # Let client infer the correct stream based on other parameters
    )

    # Retrieve forecast data for 10-meter U and V wind components
    data = client.retrieve(
        time=0,                     # Base time of the forecast (00 UTC)
        type="fc",                  # Forecast type
        step=[0, 12, 24, 36, 48],   # Forecast steps in hours
        param=["10u", "10v"],       # Parameters: 10-meter zonal (u) and meridional (v) winds
        target=fname                # File to save the retrieved data
    )

    return data

def download_precipitation(fname):
    """
    Downloads the latest IFS (Integrated Forecasting System) forecast data from ECMWF.

    Parameters:
    - fname: str
        Target filename where the downloaded GRIB data will be saved.

    Returns:
    - data: the response object from the data retrieval client.
    """
    # Initialize the client for ECMWF IFS model data
    client = Client(
        source="ecmwf",          # Data source is ECMWF
        model="aifs-single",     # Single forecast runs produced by the ECMWF Artificial Intelligence Forecasting System
        resol="0p25",            # Horizontal resolution of 0.25 degrees
        preserve_request_order=False,  # Reorder requests for performance (optional)
        infer_stream_keyword=True      # Let client infer the correct stream based on other parameters
    )

    # Retrieve forecast data for 10-meter U and V wind components
    data = client.retrieve(
        time=0,                     # Base time of the forecast (00 UTC)
        type="fc",                  # Forecast type
        step=[0, 12, 24, 36, 48],   # Forecast steps in hours
        param=["tp"],       # Parameters: 10-meter zonal (u) and meridional (v) winds
        target=fname                # File to save the retrieved data
    )

    return data

def download_lower_level_winds(fname):
    """
    Downloads the latest IFS (Integrated Forecasting System) forecast data from ECMWF.

    Parameters:
    - fname: str
        Target filename where the downloaded GRIB data will be saved.

    Returns:
    - data: the response object from the data retrieval client.
    """
    # Initialize the client for ECMWF IFS model data
    client = Client(
        source="ecmwf",          # Data source is ECMWF
        model="aifs-single",     # Single forecast runs produced by the ECMWF Artificial Intelligence Forecasting System
        resol="0p25",            # Horizontal resolution of 0.25 degrees
        preserve_request_order=False,  # Reorder requests for performance (optional)
        infer_stream_keyword=True      # Let client infer the correct stream based on other parameters
    )

    # Retrieve forecast data for 10-meter U and V wind components
    data = client.retrieve(
        time=0,                     # Base time of the forecast (00 UTC)
        type="fc",                  # Forecast type
        levtype="pl",            # Select pressure levels
        levelist=[1000,925,850,700], # Selected pressure levels
        step=[0, 12, 24, 36, 48],   # Forecast steps in hours
        param=["v","u"],       # Parameters: 10-meter zonal (u) and meridional (v) winds
        target=fname                # File to save the retrieved data
    )

    return data

def download_upper_level_winds(fname):
    """
    Downloads the latest IFS (Integrated Forecasting System) forecast data from ECMWF.

    Parameters:
    - fname: str
        Target filename where the downloaded GRIB data will be saved.

    Returns:
    - data: the response object from the data retrieval client.
    """
    # Initialize the client for ECMWF IFS model data
    client = Client(
        source="ecmwf",          # Data source is ECMWF
        model="aifs-single",     # Single forecast runs produced by the ECMWF Artificial Intelligence Forecasting System
        resol="0p25",            # Horizontal resolution of 0.25 degrees
        preserve_request_order=False,  # Reorder requests for performance (optional)
        infer_stream_keyword=True      # Let client infer the correct stream based on other parameters
    )

    # Retrieve forecast data for 10-meter U and V wind components
    data = client.retrieve(
        time=0,                     # Base time of the forecast (00 UTC)
        type="fc",                  # Forecast type
        levtype="pl",            # Select pressure levels
        levelist=[300,200,100,50], # Selected pressure levels
        step=[0, 12, 24, 36, 48],   # Forecast steps in hours
        param=["v","u"],       # Parameters: 10-meter zonal (u) and meridional (v) winds
        target=fname                # File to save the retrieved data
    )

    return data

def download_total_column_wv(fname):
    """
    Downloads the latest IFS (Integrated Forecasting System) forecast data from ECMWF.

    Parameters:
    - fname: str
        Target filename where the downloaded GRIB data will be saved.

    Returns:
    - data: the response object from the data retrieval client.
    """
    # Initialize the client for ECMWF IFS model data
    client = Client(
        source="ecmwf",          # Data source is ECMWF
        model="aifs-single",     # Single forecast runs produced by the ECMWF Artificial Intelligence Forecasting System
        resol="0p25",            # Horizontal resolution of 0.25 degrees
        preserve_request_order=False,  # Reorder requests for performance (optional)
        infer_stream_keyword=True      # Let client infer the correct stream based on other parameters
    )

    # Retrieve forecast data for 10-meter U and V wind components
    data = client.retrieve(
        time=0,                     # Base time of the forecast (00 UTC)
        type="fc",                  # Forecast type
        step=[0, 12, 24, 36, 48],   # Forecast steps in hours
        param=["tcw"],       # Parameters: 10-meter zonal (u) and meridional (v) winds
        target=fname                # File to save the retrieved data
    )

    return data