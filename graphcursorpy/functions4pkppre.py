#!/usr/bin/env python
# -*- coding=utf8 -*-
# Created Time: 2025-04-22
# Author: Marshon

import os
import numpy as np
from numpy.random import default_rng
from scipy.spatial.distance import cdist
import pandas as pd
import os, math, subprocess, warnings
from multiprocessing import Pool
from tqdm import tqdm
from scipy.signal import hilbert
from obspy import Trace, read
from obspy.core import Stream, UTCDateTime
from obspy.taup import TauPyModel
from vespy.vespagram import vespagram
from vespy.fk import fk_analysis

from graphcursorpy.functions4common import normalize, detect_peaks, DistAz, plot_stacked_map, plot_array_analysis

# Depth grids:
# --------------------------
min_scadp, max_scadp, scadp_interval = 2391, 2891, 50
scadps = np.arange(min_scadp, max_scadp+scadp_interval, scadp_interval)

def screen_predictions(df_array, pred_path, mph,  mpd, config):
    """
    Process and filter predictions from GraphCursor model outputs.
    
    Args:
        df_array (pd.DataFrame): DataFrame containing seismic metadata for one array
        pred_path (str): Path of GraphCursor predictions
        mph (float): Minimum peak height threshold for detection
        mpd (float): Minimum peak distance (in samples)
        config (module): A Python module containing configuration settings. 
        
    Returns:
        dict: Filtered dict with keys:
            'df': Filtered DataFrame
            'dists': Array distances
            'waveforms': Filtered waveforms
            'predictions': Filtered prediction arrays
            'station_locs': Station coordinates
            'times': PKP precursor times
    """

    # Initialize containers using dictionary for better organization
    dict = {
        'predictions': [],
        'waveforms': [],
        'station_locs': [],
        'pkpdf_times': [],
        'dists': [],
        'fnames': [],
        'outputs': []
    }

    # Get common event parameters from first trace
    first_tr = read(df_array.iloc[0]['fname'])[0]
    evla, evlo = first_tr.stats.sac.evla, first_tr.stats.sac.evlo 
    if first_tr.stats.sac.evdp > 1000:
        evdp = first_tr.stats.sac.evdp/1000 # convert to km
    else:
        evdp = first_tr.stats.sac.evdp
    event_time = UTCDateTime(df_array.iloc[0]['event'])

    # Main processing loop
    for _, row in df_array.iterrows():
        # Load prediction
        if config.event_split:
            directory = os.path.dirname(row['fname'])
            event = directory.split('/')[-1]
            pred_file = f"{pred_path}{event}/{os.path.basename(row['fname']).split('.SAC')[0]}.npy"
        else:
            pred_file = f"{pred_path}{os.path.basename(row['fname']).split('.SAC')[0]}.npy"
        pred = np.load(pred_file)
        
        # Detect PKP precursors
        detection = detect_pkppre([pred, row['dist']], mph, mpd, config)
            
        # Load and trim waveform
        tr = read(row['fname'])[0]
        tr_cp = tr.copy()
        tr_cp.trim(tr_cp.stats.starttime + config.pre_window - config.waveform_time, tr_cp.stats.starttime + config.pre_window + 10)
        trimmed_waveform = tr_cp.data[:int(config.waveform_time*config.sr)]
        
        # Store results
        dict['predictions'].append(pred)
        dict['outputs'].append(detection)
        dict['waveforms'].append(trimmed_waveform)
        dict['fnames'].append(row['fname'])
        dict['station_locs'].append([tr.stats.sac.stla, tr.stats.sac.stlo])
        dict['pkpdf_times'].append(tr.stats.starttime + config.pre_window - event_time)
        dict['dists'].append(row['dist'])
    
    # Convert to numpy arrays
    for key in dict:
        dict[key] = np.array(dict[key])
    
    # Apply time filter (-3s threshold)
    time_mask = dict['outputs'][:, 0] <= -3
    filtered_dict = {
        'df_array': df_array.loc[time_mask].reset_index(drop=True),
        'fnames': dict['fnames'][time_mask],
        'dists': dict['dists'][time_mask],
        'waveforms': dict['waveforms'][time_mask],
        'predictions': dict['predictions'][time_mask],
        'station_locs': dict['station_locs'][time_mask],
        'pkpdf_times': dict['pkpdf_times'][time_mask],
        'times': dict['outputs'][time_mask][:, 0],
        'evla': evla,
        'evlo': evlo,
        'evdp': evdp
    }
    
    return filtered_dict


def calculate_centroid(stas):
    """
    Calculate the geographic centroid from a list of (latitude, longitude) tuples.
    This method accounts for the cyclic nature of longitude values (e.g., 179 and -179).

    Args:
        stas (list): A list of tuples containing (latitude, longitude).

    Returns:
        tuple: A tuple containing the centroid's latitude and longitude.
    """
    sum_lat = 0.0
    x, y, z = 0.0, 0.0, 0.0
    count = len(stas)
    
    # Calculate the average latitude and convert longitude into a 3D Cartesian system
    for sta in stas:
        lat = math.radians(sta[0])  # Convert latitude to radians
        lon = math.radians(sta[1])  # Convert longitude to radians

        # Sum up latitudes normally
        sum_lat += lat
        
        # Convert spherical coordinates (longitude) to Cartesian coordinates (x, y, z)
        x += math.cos(lat) * math.cos(lon)
        y += math.cos(lat) * math.sin(lon)
        z += math.sin(lat)

    # Calculate the average latitude and longitude
    avg_lat = math.degrees(sum_lat / count)

    # Convert the averaged Cartesian coordinates back to spherical (longitude)
    avg_x = x / count
    avg_y = y / count
    avg_z = z / count

    # Recalculate longitude and latitude from the average Cartesian coordinates
    avg_lon = math.atan2(avg_y, avg_x)
    avg_lat = math.atan2(avg_z, math.sqrt(avg_x**2 + avg_y**2))

    # Convert the results back to degrees
    centroid_lat = round(math.degrees(avg_lat), 4)
    centroid_lon = round(math.degrees(avg_lon), 4)
    
    return centroid_lat, centroid_lon

def create_grid(center_lat, center_lon, diff = 20, step = 1):
    """
    Create a 40x40 degree grid centered at the given latitude and longitude with 2 degree step,
    with special handling for wraparound at 180 degrees longitude and 90 degrees latitude.

    Args:
        center_lat (float): Center latitude of the grid.
        center_lon (float): Center longitude of the grid.
        diff (float): Half-width of grid in degrees (default 20° creates 40°x40° grid).
        step (float): Grid spacing in degrees (default 1°).

    Returns:
        list of tuples: List containing (latitude, longitude) tuples for each grid point.
    """
    # Input validation
    if not (-90 <= center_lat <= 90):
        raise ValueError(f"center_lat must be between -90 and 90, got {center_lat}")
    if not (-180 <= center_lon <= 180):
        raise ValueError(f"center_lon must be between -180 and 180, got {center_lon}")
    if diff <= 0:
        raise ValueError(f"diff must be positive, got {diff}")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    
    # Generate grid coordinates with proper edge inclusion
    lat_points = np.arange(center_lat - diff, center_lat + diff + step/2, step)
    lon_points = np.arange(center_lon - diff, center_lon + diff + step/2, step)

    # Create grid and adjust for geographic boundaries
    lats, lons = np.meshgrid(lat_points, lon_points, indexing='ij')

    # Apply latitude wrapping (reflect beyond poles)
    lats = np.where(lats > 90, 180 - lats, lats)
    lats = np.where(lats < -90, -180 - lats, lats)

    # Apply longitude wrapping (continuous across dateline)
    lons = np.mod(lons + 180, 360) - 180

    # Combine and format results
    grid_points = np.column_stack([
        np.round(lats.ravel(), 2),
        np.round(lons.ravel(), 2)
    ])

    return grid_points

def calculate_pierce(evla, evlo, evdp, centroid_lat, centroid_lon, dep = 2891):
    '''
    Calculate pierce point locations for PKIKP phase using TauP.

    Args:
        evla (float): Event latitude in degrees (-90 to 90).
        evlo (float): Event longitude in degrees (-180 to 180).
        evdp (float): Event depth in kilometers.
        centroid_lat (float): Station/centroid latitude in degrees.
        centroid_lon (float): Station/centroid longitude in degrees.
        dep (float, optional): Boundary depth in km (default: 2891km (CMB) for ak135 model).

    Returns:
        Tuple containing:
        - source_pierce: [latitude, longitude] of source-side pierce point
        - receiver_pierce: [latitude, longitude] of receiver-side pierce point
    '''
    # Validate input parameters
    if not (-90 <= evla <= 90) or not (-90 <= centroid_lat <= 90):
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not (-180 <= evlo <= 180) or not (-180 <= centroid_lon <= 180):
        raise ValueError("Longitude must be between -180 and 180 degrees")
    if evdp <= 0:
        raise ValueError("Event depth must be positive")
    if dep <= 0:
        raise ValueError("Depth must be positive")

    # Build TauP command
    cmd = (
        f"taup pierce -mod ak135 -h {evdp} -ph PKIKP "
        f"-sta {centroid_lat} {centroid_lon} -evt {evla} {evlo} "
        f"--pierce {dep} --nodiscon"
    )
    # Run TauP and capture output
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=10  # Prevent hanging
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"TauP calculation failed: {e.stderr}") from e
    except subprocess.TimeoutExpired:
        raise RuntimeError("TauP calculation timed out")

    # Parse output
    lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
    
    try:
        # Last two non-empty lines contain pierce points
        source_line = lines[-2].split()
        receiver_line = lines[-1].split()
        
        source_pierce = [float(source_line[-2]), float(source_line[-1])]
        receiver_pierce = [float(receiver_line[-2]), float(receiver_line[-1])]
    except (IndexError, ValueError) as e:
        raise RuntimeError(f"Failed to parse TauP output: {e}\nOutput:\n{result.stdout}") from e

    return source_pierce, receiver_pierce

def detect_pkppre(input, mph, mpd, config):
    '''
    Detect PKP precursor phases in seismic waveform data.

    Args:
        input (list): [pkppre_predict, dist] where:
            - pkppre_predict (array): Prediction array containing probability values (0-1).
            - dist (float): Epicentral distance in degrees.
        mph (float): Minimum peak height threshold for detection
        mpd (float): Minimum peak distance (in samples)
        config (module): A Python module containing configuration settings. 

    Returns:
        list: [pkppre_pick, dist] where:
            - pkppre_pick (float/None): Detected arrival time in seconds (relative to waveform start).
            - dist (float): Original input distance.
            - Returns [np.inf, None] if no detection meets criteria.
    '''

    # Detect peaks in probability time series
    pkppre_predict, dist = input[0], input[1]
    pkppre_ind = detect_peaks(pkppre_predict, mph = mph, mpd = mpd)
    pkppre_picks, pkppre_amps = [], []

    # Process detected peaks
    if (len(pkppre_ind[0]) > 0):
        for idx in pkppre_ind[0]:
            amp = pkppre_predict[idx]
            pick = round(idx/config.sr, 3)
            pkppre_amps.append(amp)
            pkppre_picks.append(pick)
        pkppre_amps = np.array(pkppre_amps)

        # Select highest amplitude peak
        pkppre_pick = float(pkppre_picks[np.argmax(pkppre_amps)])
        output = [pkppre_pick-config.waveform_time, dist]
    else:
        output = [np.inf, dist]
    return output

def calculate_eq2sca_times(df, sca_points, evla, evlo, evdp):
    """
    Optimized function to calculate travel times from seismic source to scatter points.
    
    Args:
        df: DataFrame with precomputed distances and travel times.
        sca_points: List of scatter points (latitude, longitude).
        evla: Latitude of the seismic event.
        evlo: Longitude of the seismic event.
        evdp: Depth of the seismic event.
    
    Returns:
        eq2sca_times: Array of travel times for each scatter point.
    """

    # Convert columns to NumPy arrays for faster processing
    evdp_array = df['evdp'].to_numpy()

    # Find the minimum depth index based on the source depth (evdp)
    min_dep_index = np.argmin(np.abs(evdp_array - evdp))
    min_dep = evdp_array[min_dep_index]

    # Filter the DataFrame to only keep rows with the minimum depth
    df_filtered = df.loc[df['evdp'] == min_dep]

    # Pre-compute the eq2sca_dist column as a NumPy array for fast access
    eq2sca_dist_array = df_filtered['eq2sca_dist'].to_numpy()
    time_array = df_filtered['time'].to_numpy()

    # Initialize an array to store the results for travel times
    eq2sca_times = np.zeros((len(sca_points), len(scadps)))

    for i, sca_point in enumerate(sca_points):
        sca_lat, sca_lon = sca_point[0], sca_point[1]

        # Calculate the distance from the source-side scatter point to the seismic source
        sca_dist = DistAz(sca_lat, sca_lon, evla, evlo).getDelta()

        # Use NumPy to find the index of the closest distance
        res_dist = np.abs(eq2sca_dist_array - sca_dist)
        min_dist = np.min(res_dist)
        min_dist_index = np.where(res_dist == min_dist)[0]

        # Get the corresponding travel time for the closest distance
        eq2sca_times[i, :] = time_array[min_dist_index]

    return eq2sca_times

def calculate_sca2sta_times(df_sca2sta, sca_points, stas):
    """
    Calculate the travel time from source-side scatter to each station.

    Args:
        df_sca2sta: DataFrame with precomputed distances and travel times.
        sca_points: List of scatter points (latitude, longitude).
        stas: List of stations (latitude, longitude).

    Returns:
    - sca2sta_times: A 3D NumPy array.
    """

    # Convert relevant columns to NumPy arrays for faster access
    sca2sta_dist_array = df_sca2sta['sca2sta_dist'].to_numpy()
    time_array = df_sca2sta['time'].to_numpy()

    # Initialize an array to store the results for travel times
    sca2sta_times = np.zeros((len(sca_points), len(scadps), len(stas)))

    for i, sca_point in enumerate(sca_points):
        sca_lat, sca_lon = sca_point[0], sca_point[1]

        for j, sta in enumerate(stas):
            sta_lat, sta_lon = float(sta[0]), float(sta[1])

            # Calculate the distance from the source-side scatter point to the station
            sca2sta_dist = DistAz(sca_lat, sca_lon, sta_lat, sta_lon).getDelta()

            # Use NumPy to find the closest distance index
            res_dist = np.abs(sca2sta_dist_array - sca2sta_dist)
            min_dist = np.min(res_dist)
            min_dist_index = np.where(res_dist == min_dist)[0]

            # Get the corresponding travel time for the closest distance
            sca2sta_times[i, :, j] = time_array[min_dist_index]

    return sca2sta_times

def stack_predictions(sca_points, eq2sca_times, sca2sta_times,
                      stas, pkpdfs, inputs, config, window = 2, norm = True):
    """
    Calculate PKP precursor arrivals and stack predictions for each scatter point.

    Args:
        sca_points (np.ndarray): Array of shape (n_points, 2) containing (lat, lon) of scatter points.
        eq2sca_times (np.ndarray): Array of shape (n_points, n_scadps) with eq-to-scatter times.
        sca2sta_times (np.ndarray): Array of shape (n_points, n_scadps, n_stas) with scatter-to-station times.
        stas (list): List of station identifiers.
        pkpdfs (np.ndarray): Array of PKPdf arrival times for each station.
        inputs (np.ndarray): 2D array (n_stas, waveform_len) of prediction waveforms.
        config (module): A Python module containing configuration settings. 
        window (float): Time window length in seconds for stacking (default: 2.0).
        norm (bool): Whether to normalize individual waveforms before stacking (default: True).

    Returns:
        stacks: Array of stacked predictions for each scatter point.
    """
    # Initialize an array to store the maximum stacked predictions for each scatter point
    stacks = np.full((len(sca_points), len(scadps)), np.nan)  # Default as NaN to mark uncalculated points
    
    # Precompute theo_pkppre for all scatter points and stations
    theo_pkppres = eq2sca_times[:, :, np.newaxis] + sca2sta_times  # Shape (n_sca_points, n_scadps, n_stas)

    # Loop through each scatter point
    for i in range(len(sca_points)):
        for j in range(len(scadps)):
            useful_stas = 0
            stacked_prediction = np.zeros(config.waveform_len)
            for k in range(len(stas)):
                pkpdf = pkpdfs[k]
                if theo_pkppres[i, j, k] > (pkpdf - config.waveform_time) and theo_pkppres[i, j, k] < pkpdf:

                    # A window of 2 s length around the theoretical arrival (from Thomas, 1999)
                    start_idx = int((theo_pkppres[i, j, k] - pkpdf + config.waveform_time - window/2) * config.sr)
                    end_idx = int(start_idx + window * config.sr)
                    if start_idx >= 0 and end_idx < len(inputs[k]):
                        useful_stas += 1
                        prediction4stack = inputs[k][start_idx:end_idx]
                        if norm:
                            prediction4stack = prediction4stack/np.max(np.abs(prediction4stack))
                        else:
                            prediction4stack = prediction4stack
                        stacked_prediction[:len(prediction4stack)] += prediction4stack

            # Store the maximum value of stacked predictions for the current scatter point
            if useful_stas > 0:
                stacks[i, j] = max(stacked_prediction)/len(stas)
    return stacks

def process_scattering_side(pierce_point, eq2sca_df, sca2sta_df, event_loc, station_data, config,
                            window = 2, norm = True):
    """
    Calculate PKP precursor arrivals and stack predictions for each scatter point on one side
    (either source-side or receiver-side).

    Args:
        pierce_point (tuple): (latitude, longitude) of the PKP piercing point at the core-mantle boundary.
        eq2sca_df (pd.DataFrame): Travel time table from earthquake to scatterer.
        sca2sta_df (pd.DataFrame): Travel time table from scatterer to station.
        event_loc (dict): Dictionary with keys 'lat', 'lon', and 'depth' specifying the earthquake location.
        station_data (dict): Dictionary with keys:
            - 'locations' (np.ndarray): Array of station (lat, lon) positions.
            - 'pkpdf_times' (np.ndarray): PKPdf arrival times at each station.
            - 'predictions' (np.ndarray): Prediction waveform array for each station.
        config (object): Configuration object with fields such as 'diff' (grid size) and 'step' (grid step).
        window (float, optional): Time window (in seconds) used for stacking predictions (default is 2).
        norm (bool, optional): Whether to normalize prediction waveforms before stacking (default is True).

    Returns:
        dict: A dictionary with the following keys:
            - 'points' (np.ndarray): Scatter grid points as (lat, lon) coordinates.
            - 'stacks' (np.ndarray): Stacked prediction results for each scatter point.
            - 'travel_times' (np.ndarray): Total travel times (eq-to-sca + sca-to-sta), shape (n_points, n_stas, 1).
    """
    sca_points = create_grid(pierce_point[0], pierce_point[1], config.diff, config.step)
    eq2sca_times = calculate_eq2sca_times(eq2sca_df, sca_points, event_loc['lat'], event_loc['lon'], event_loc['depth'])
    sca2sta_times = calculate_sca2sta_times(sca2sta_df, sca_points, station_data['locations'])
    
    stacks = stack_predictions(
        sca_points, eq2sca_times, sca2sta_times,
        station_data['locations'], station_data['pkpdf_times'], 
        station_data['predictions'], config, window, norm
    )
    
    return {
        'points': sca_points,
        'stacks': stacks,
        'travel_times': eq2sca_times[:, :, np.newaxis] + sca2sta_times
    }

def iterative_centroid(lats, lons, weights=None, max_iter=10, radius_factor=0.3):
    """
    Iteratively compute a robust geographic centroid by only considering nearby points in each step.

    This method starts from a global weighted centroid and iteratively refines it by focusing 
    on local clusters (within a shrinking radius), which helps reduce the influence of outliers.

    Args:
        lats (np.ndarray): Array of latitudes of points.
        lons (np.ndarray): Array of longitudes of points.
        weights (np.ndarray, optional): Optional weights for each point. If None, all points are equally weighted.
        max_iter (int, optional): Maximum number of iterations (default: 10).
        radius_factor (float, optional): Factor to control the local radius as a fraction of the distance standard deviation (default: 0.3).

    Returns:
        tuple: The estimated centroid as a tuple (latitude, longitude).
    """

    points = np.column_stack([lats, lons])
    
    if weights is None:
        weights = np.ones(len(lats))
    
    # Initial centroid (weighted average of all points)
    current_lat = np.average(lats, weights=weights)
    current_lon = np.average(lons, weights=weights)
    
    for i in range(max_iter):
        current_center = np.array([current_lat, current_lon])
        
        # Compute distances from current centroid to all points
        distances = cdist([current_center], points)[0]
        
        # Define search radius based on the standard deviation of distances
        radius = radius_factor * np.std(distances)
        
        # Select points within the radius
        nearby_mask = distances <= radius
        if np.sum(nearby_mask) < 3:  # Too few points; use a more relaxed threshold
            radius = np.percentile(distances, 30)
            nearby_mask = distances <= radius
        
        nearby_lats = lats[nearby_mask]
        nearby_lons = lons[nearby_mask]
        nearby_weights = weights[nearby_mask]
        
        # Recompute the centroid using only nearby points
        new_lat = np.average(nearby_lats, weights=nearby_weights)
        new_lon = np.average(nearby_lons, weights=nearby_weights)
        
        # Check for convergence
        if np.abs(new_lat - current_lat) < 1e-6 and np.abs(new_lon - current_lon) < 1e-6:
            break
            
        current_lat, current_lon = new_lat, new_lon
    
    return current_lat, current_lon

def locate_scatter_migration(dict4location, df_eq2source_sca, df_source_sca2sta, 
                             df_eq2receiver_sca, df_receiver_sca2sta, config, 
                             window = 2, norm = True, plot = False,
                             save_fig = False, output_dir=None):
    """
    Locate scattering points by migrating seismic array data and GraphCursor predictions.
    
    Args:
        dict4location (dict): Contains:
            - df_array: DataFrame with array metadata.
            - evla, evlo, evdp: Event latitude, longitude, depth.
            - station_locs: Station locations.
            - predictions: GraphCursor predictions.
            - pkpdf_times: PKPdf arrival times.
        df_eq2source_sca : pd.DataFrame
            Ray tracing results from event to source-side scattering points
        df_source_sca2sta : pd.DataFrame  
            Ray tracing results from source-side scattering to stations
        df_eq2receiver_sca : pd.DataFrame
            Ray tracing results from event to receiver-side scattering points  
        df_receiver_sca2sta : pd.DataFrame
            Ray tracing results from receiver-side scattering to stations
        config (module): A Python module containing configuration settings. 
        window (float): Time window length in seconds for stacking (default: 2.0).
        norm (bool): Whether to normalize individual waveforms before stacking (default: True).
        plot (bool): Whether to generate scatter location plots (default: False).
        save_fig (bool): Whether to save the figure (default: False).
        output_dir (str): Directory to save figures (required if save_fig=True).
    
    Returns:
        dict: Dictionary containing:
            - dominant_side: 'source' or 'receiver'.
            - secondary_side: 'source' or 'receiver'.
            - source_grids: Grids in source side.
            - receiver_grids: Grids in receiver side.
            - max_scatter: Dict of maximum amplitude scatter point details.
            - min_scatter: Dict of secondary scatter point details.
            - pierce_points: Tuple of (source_pierce, receiver_pierce).
            - metadata: 'array': array_data['array'], 'event_location': event_loc, 'station_centroid': [centroid_lat, centroid_lon]
        }
    """

    # Extract common variables
    array_data = dict4location['df_array'].iloc[0]
    event_loc = {
        'lat': dict4location['evla'],
        'lon': dict4location['evlo'], 
        'depth': dict4location['evdp']
    }
    station_data = {
        'locations': dict4location['station_locs'],
        'predictions': dict4location['predictions'],
        'pkpdf_times': dict4location['pkpdf_times']
    }

    # Calculate station centroid and pierce points
    centroid_lat, centroid_lon = calculate_centroid(station_data['locations'])
    source_pierce, receiver_pierce = calculate_pierce(
        event_loc['lat'], event_loc['lon'], event_loc['depth'],
        centroid_lat, centroid_lon
    )

    # Process both scattering sides
    source_side = process_scattering_side(source_pierce, df_eq2source_sca, df_source_sca2sta, 
                                          event_loc, station_data, config, window, norm)
    receiver_side = process_scattering_side(receiver_pierce, df_eq2receiver_sca, df_receiver_sca2sta, 
                                            event_loc, station_data, config, window, norm)
    source_dp_idx = np.unravel_index(np.nanargmax(source_side['stacks']), source_side['stacks'].shape)[1]
    receiver_dp_idx = np.unravel_index(np.nanargmax(receiver_side['stacks']), receiver_side['stacks'].shape)[1]

    # Determine dominant scattering side
    receiver_max_amp = np.nanmax(receiver_side['stacks'])
    source_max_amp = np.nanmax(source_side['stacks'])
    if receiver_max_amp < source_max_amp:
        dominant_side = 'source'
        secondary_side = 'receiver'
        max_data, min_data = source_side, receiver_side
        dp_idx = source_dp_idx
    else:
        dominant_side = 'receiver'
        secondary_side = 'source'
        max_data, min_data = receiver_side, source_side
        dp_idx = receiver_dp_idx

    # Obtain the scatter location
    secondary_max = np.nanmax(min_data['stacks'][:, dp_idx])
    sca_lat_array = max_data['points'][:, 0][max_data['stacks'][:, dp_idx] >= secondary_max]
    sca_lon_array = max_data['points'][:, 1][max_data['stacks'][:, dp_idx] >= secondary_max]
    weights = max_data['stacks'][:, dp_idx][max_data['stacks'][:, dp_idx] >= secondary_max]
    sca_lat_max, sca_lon_max = iterative_centroid(sca_lat_array, sca_lon_array, weights)
    sca_lat_min = min_data['points'][np.unravel_index(np.nanargmax(min_data['stacks']), min_data['stacks'].shape)[0]][0]
    sca_lon_min = min_data['points'][np.unravel_index(np.nanargmax(min_data['stacks']), min_data['stacks'].shape)[0]][1]

    if plot:
        # Generate plot
        if dominant_side == 'receiver':
            plot_stacked_map(
                source_side['stacks'][:, receiver_dp_idx], receiver_side['stacks'][:, receiver_dp_idx],
                source_side['points'], receiver_side['points'],
                array_data['array'],
                dominant_side,
                sca_lat_max,
                sca_lon_max,
                config,
                save_fig,
                output_dir
            )
        else:
            plot_stacked_map(
                source_side['stacks'][:, source_dp_idx], receiver_side['stacks'][:, source_dp_idx],
                source_side['points'], receiver_side['points'],
                array_data['array'],
                dominant_side,
                sca_lat_max,
                sca_lon_max,
                config,
                save_fig,
                output_dir
            )
    else:
        pass

    return {
        'dominant_side': dominant_side,
        'secondary_side': secondary_side,
        'sca_lat_max': sca_lat_max,
        'sca_lon_max': sca_lon_max,
        'max_amp': np.nanmax(max_data['stacks'][:, dp_idx]),
        'sca_lat_min': sca_lat_min,
        'sca_lon_min': sca_lon_min,
        'min_amp': np.nanmax(min_data['stacks'][:, dp_idx]),
        'dp_idx': dp_idx,
        'depth': config.scadps[dp_idx],
        'source_info': source_side,
        'receiver_info': receiver_side,
        'pierce_points': {
            'source': source_pierce,
            'receiver': receiver_pierce
        },
        'metadata': {
            'array': array_data['array'],
            'event_location': event_loc,
            'station_centroid': [centroid_lat, centroid_lon]
        }
    }

def search_dist_idx(dist):
    """
    Find the index in a distance range that corresponds to the given distance.
    
    Args:
        dist: Target distance to search for (degrees)
        dist_range: Sorted array/list of distance boundaries
        
    Returns:
        Index i where dist_range[i] < dist <= dist_range[i+1]
    """
    dist_range = np.arange(125, 143.5, 0.5)
    # Handle special case for exactly 125°
    if dist == 125:
        return 0
    
    # Check bounds
    if dist <= dist_range[0]:
        raise ValueError(f"Distance {dist}° is below minimum range {dist_range[0]}°")
    if dist > dist_range[-1]:
        raise ValueError(f"Distance {dist}° exceeds maximum range {dist_range[-1]}°")
    
    # Binary search for efficiency with large ranges
    left, right = 0, len(dist_range) - 1
    while left < right:
        mid = (left + right) // 2
        if dist_range[mid] < dist <= dist_range[mid + 1]:
            return mid
        elif dist <= dist_range[mid]:
            right = mid
        else:
            left = mid + 1
    
    return left - 1  # Final case when left == right

def normalize_lon(lon):
    """
    Normalize a longitude value to the range [0, 360) degrees.
    
    Args:
        lon (float): Longitude in degrees, can be in any range.

    Returns:
        float: Normalized longitude in the range [0, 360).
    """
    return lon % 360

def in_eastern_hemisphere(lon):
    """Check if longitude falls within the Eastern Hemisphere (40°E-180°E)"""
    lon = normalize_lon(lon)
    return 40 <= lon <= 180

def in_western_hemisphere(lon):
    """
    Check if a given normalized longitude is in the Eastern Hemisphere,
    defined here as between 40°E (40°) and 180°E (180°).
    
    Args:
        lon (float): Longitude in degrees, normalized to [0, 360).
    
    Returns:
        bool: True if in Eastern Hemisphere, False otherwise.
    """
    lon = normalize_lon(lon)
    return lon >= 180 or lon < 40

def compute_hemisphere_arc_lengths(src_lon, rcv_lon):
    """
    Estimate how much of the great-circle arc between a source and receiver
    falls within the Eastern and Western hemispheres, using sampling in longitude.

    Eastern Hemisphere is defined as: 40°E to 180°E
    Western Hemisphere is defined as: the complement (180°W to 40°E)

    Args:
        src_lon (float): Source longitude in degrees (can be -180 to 180 or 0 to 360).
        rcv_lon (float): Receiver longitude in degrees.

    Returns:
        dict: {
            "arc_east_deg": Arc length in degrees falling in Eastern Hemisphere,
            "arc_west_deg": Arc length in degrees falling in Western Hemisphere,
            "east_ratio": Fraction of total arc in Eastern Hemisphere,
            "west_ratio": Fraction of total arc in Western Hemisphere
        }
    """

    lon1 = normalize_lon(src_lon)
    lon2 = normalize_lon(rcv_lon)
    # Wrap around 360° if necessary
    if abs(lon2 - lon1) > 180:
        if lon1 > lon2:
            lon2 += 360
        else:
            lon1 += 360

    arc_east, arc_west = 0, 0
    n_samples = 100  # sample along the great circle in lon
    lons = np.linspace(lon1, lon2, n_samples)

    for lon in lons:
        lon_norm = normalize_lon(lon)
        if in_eastern_hemisphere(lon_norm):
            arc_east += 1
        else:
            arc_west += 1

    total_arc = arc_east + arc_west
    return {
        "arc_east_deg": arc_east * abs(lon2 - lon1) / n_samples,
        "arc_west_deg": arc_west * abs(lon2 - lon1) / n_samples,
        "east_ratio": arc_east / total_arc,
        "west_ratio": arc_west / total_arc
    }

def calculate_relative_strength(fnames, stack_map, stack_fitted_line, config, ic_attenu = False):
    """
    Compute the relative strength of seismic signals in the PKPpre window by comparing 
    each waveform to a global stack and optionally applying inner core attenuation correction.

    Args:
        fnames (List[str]): List of SAC waveform file paths.
                            Each file must contain:
                                - 'gcarc' (epicentral distance) in SAC headers
                                - waveform data
        stack_map (ndarray): 2D array [time_samples x distance_bins] of stacked amplitude envelopes.
        stack_fitted_line (ndarray): 1D array of fitted PKPdf arrival times (in seconds) for each distance bin.
        config (object): Configuration object with parameters like sr, waveform_time, pre_window, and attenuation settings.
        ic_attenu (bool): Whether to apply inner core attenuation correction based on hemispheric Q differences.

    Returns:
        float: Median relative strength deviation across valid traces (dimensionless).
    """

    # Initialize storage with NaN (preserves float type)
    relative_deviations = []
    valid_traces = 0

    # Time windows in samples (converted once for efficiency)
    # pretime: Start time (in seconds) before the reference time (PKPdf) for noise window
    #          Typical value -50 means start 50 seconds before PKPdf arrival
    # posttime: End time (in seconds) after the reference time for PKPdf signal window
    #           Typical value 10 means extend 10 seconds after PKPdf arrival
    pretime, posttime = -50, 10
    noise_window = slice(
        int((pretime + config.pre_window) * config.sr),
        int((-1 * config.waveform_time + config.pre_window) * config.sr)
    )
    pkpdf_window = slice(
        int(config.pre_window * config.sr),
        int((posttime + config.pre_window) * config.sr)
    )
    pkppre_end = int((config.pre_window) * config.sr)
    end_idx = int(config.waveform_time * config.sr)

    # Correction without inner core attenuation
    correction = 1

    for fname in fnames:
        try:
            # Read trace with validation
            tr = read(fname)[0]
            if not hasattr(tr.stats, 'sac') or not hasattr(tr.stats.sac, 'gcarc'):
                warnings.warn(f"Skipping {fname}: Missing SAC header or gcarc", RuntimeWarning)
                continue

            dist = tr.stats.sac.gcarc
            if np.isnan(dist) or dist < 0:
                warnings.warn(f"Skipping {fname}: Invalid distance {dist}", RuntimeWarning)
                continue
            
            # Calculate the correction for inner core attenuation

            if ic_attenu and (dist >= 130):
                # Calculate the pierce points at inner core boundary
                evla, evlo, evdp = tr.stats.sac.evla, tr.stats.sac.evlo, tr.stats.sac.evdp
                if evdp > 1000:
                    evdp = evdp/1000
                stla, stlo = tr.stats.sac.stla, tr.stats.sac.stlo
                source_point, receiver_point = calculate_pierce(evla, evlo, evdp, stla, stlo, 5153.5)
                source_lon, receiver_lon = source_point[1], receiver_point[1]
                arc_result = compute_hemisphere_arc_lengths(source_lon, receiver_lon)
                correction_east = config.slope_east * dist + config.intercept_east
                correction_west = config.slope_west * dist + config.intercept_west
                correction = (arc_result["east_ratio"] * correction_east +
                            arc_result["west_ratio"] * correction_west)

            # Get stacked reference
            dist_idx = search_dist_idx(dist)
            time_stacked4use = stack_fitted_line[dist_idx]
            stacked4use_idx = int((time_stacked4use + config.waveform_time) * config.sr)
            
            if stacked4use_idx >= stack_map.shape[0] or dist_idx >= stack_map.shape[1]:
                warnings.warn(f"Skipping {fname}: Invalid stack map indices", RuntimeWarning)
                continue
                
            envelope_global_average = stack_map[stacked4use_idx:end_idx, dist_idx]

            # Calculate normalized envelope
            envelope = np.abs(hilbert(tr.data))
            noise_amp = np.nanmean(envelope[noise_window])
            envelope_clean = envelope - noise_amp

            # Normalize by PKPdf amplitude with safety checks
            pkpdf_amp = np.nanmax(envelope_clean[pkpdf_window])
            if pkpdf_amp <= 0:
                warnings.warn(f"Skipping {fname}: Invalid PKPdf amplitude", RuntimeWarning)
                continue

            envelope_norm = envelope_clean / pkpdf_amp

            # Calculate PKPpre relative strength
            pkppre_start = int((time_stacked4use + config.pre_window) * config.sr)
            pkppre_window = slice(pkppre_start, pkppre_end)  # To end of trace
            
            if pkppre_start >= len(envelope_norm):
                warnings.warn(f"Skipping {fname}: PKPpre window out of bounds", RuntimeWarning)
                continue

            envelope_used = envelope_norm[pkppre_window]
            
            if len(envelope_used) != len(envelope_global_average):
                warnings.warn(f"Skipping {fname}: Window length mismatch", RuntimeWarning)
                continue

            # Calculate and store relative deviation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                amp_ratios = envelope_used / envelope_global_average
                relative_deviation = np.nanmedian(amp_ratios) * correction
                
            if not np.isnan(relative_deviation):
                relative_deviations.append(relative_deviation)
                valid_traces += 1

        except Exception as e:
            warnings.warn(f"Error processing {fname}: {str(e)}", RuntimeWarning)
            continue

    # Calculate final statistic
    if not relative_deviations:
        warnings.warn("No valid traces processed", RuntimeWarning)
        return np.nan

    return float(np.nanmedian(relative_deviations))

def random_select_stations(df_array, num_stations = 10, max_dist_diff = 4, max_az_diff = 5,
                           max_attempts = 10000):
    """
    Randomly select a subset of stations meeting specified distance and azimuth constraints.

    Args:
        df_array: DataFrame containing station information with 'dist' and 'azimuth' columns
        num_stations: Minimum number of stations to select (must be <= len(df_array)) (default: 6)
        max_dist_diff: Maximum allowed difference in distance (degrees) between selected stations (default: 5)
        max_az_diff: Maximum allowed difference in azimuth (degrees) between selected stations (default: 5)
        max_attempts: Maximum number of random attempts before giving up (default: 1000)

    Returns:
        DataFrame containing selected stations meeting the conditions, or None if no valid
        subset is found within max_attempts

    Raises:
        ValueError: If input parameters are invalid
    """
    
    # Input validation
    if num_stations <= 0:
        raise ValueError("num_stations must be positive")
    if len(df_array) < num_stations:
        raise ValueError("df_array has fewer elements than requested num_stations")
    if max_dist_diff <= 0 or max_az_diff <= 0:
        raise ValueError("max_dist_diff and max_az_diff must be positive")
    
    required_columns = {'dist', 'azimuth'}
    if not required_columns.issubset(df_array.columns):
        raise ValueError(f"df_array must contain columns: {required_columns}")

    rng = default_rng() # Automatically use system-level random seeds
    valid_candidate_groups = []

    for _ in range(max_attempts):
        # 1. Randomly select a reference station
        ref_idx = rng.choice(len(df_array))
        ref_station = df_array.iloc[ref_idx]
        
        # 2. Find all stations within constraints of reference station
        mask = (
            (np.abs(df_array['dist'] - ref_station['dist']) <= max_dist_diff/2) &
            (np.abs(df_array['azimuth'] - ref_station['azimuth']) <= max_az_diff/2)
        )
        candidates = df_array[mask].copy()
        
        # 3. Check if we have enough stations
        if len(candidates) >= num_stations:
            selected = candidates.sample(n=num_stations, random_state=rng.integers(0, 1e9))
            valid_candidate_groups.append(selected)
    if valid_candidate_groups:
        idx = rng.integers(0, len(valid_candidate_groups))
        return valid_candidate_groups[idx]

    return None

def array_analysis(df_array, order, ssteps, stat, winlen, pre_time,
                   post_time, pws, smin, smax, freqmin, freqmax, config, 
                   fk_stat = 'power', trim_type = None, plot = False):
    """
    Perform array analysis (vespagram and FK analysis) on seismic station data.

    Args:
        df_array: DataFrame containing station metadata with required columns:
                 ['array', 'dist', 'fname', 'back_azimuth']
        order: Order for phase weighting in vespagram
        ssteps: Number of slowness steps in vespagram
        stat: Stationarity parameter for vespagram
        winlen (float): Analysis window length in seconds
        pre_time (float): Start time relative to phase arrival (seconds)
        post_time (float): End time relative to phase arrival (seconds)
        pws (bool): Enable phase-weighted stacking
        smin (float): Minimum slowness for analysis (s/deg)
        smax (float): Maximum slowness for analysis (s/deg)
        freqmin (float): Lower frequency bound for filtering (Hz)
        freqmax (float): Upper frequency bound for filtering (Hz)
        config (module): A Python module containing configuration settings. 
        trim_type (bool): Whether to trim waveforms around PKP precursor or PKIKP 
                          for vespagram and FK analysis.
        plot (bool, optional): Generate summary plots if True. Defaults to True.

    Returns:
        Dictionary containing:
        - 'vespa_slowness': Slowness from vespagram analysis
        - 'fk_slowness': Slowness from FK analysis
        - 'fk_backazimuth': Bac-azimuth from from FK analysis
        - 'stla', 'stlo': Location of reference station
        - 'vespagram_data': Vespagram analysis results
        - 'fk_data': FK analysis results
    Raises:
        ValueError: If input parameters are invalid or data loading fails
    """

    # Validate inputs
    required_columns = {'array', 'dist', 'fname', 'back_azimuth'}
    if not required_columns.issubset(df_array.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(df_array.columns)}")
    
    if config is None or not hasattr(config, 'sr') or not hasattr(config, 'pre_window'):
        raise ValueError("Config object must have 'sr' and 'pre_window' attributes")

    try:
        # Initialize parameters
        array_name = df_array.iloc[0]['array']
        sr = config.sr
        dist_const = 111.19  # Conversion factor from deg to km
        s_list = np.linspace(smin/dist_const, smax/dist_const, ssteps)
        npts = int((post_time - pre_time) * sr + 1)

        # Find reference station (closest to median distance)
        med_dist = df_array['dist'].median()
        closest_idx = (df_array['dist'] - med_dist).abs().idxmin()
        ref_fname = df_array.loc[closest_idx, 'fname']
        ref_baz = df_array.loc[closest_idx, 'back_azimuth']

        # Load and prepare reference trace
        try:
            ref_tr = read(ref_fname)[0]
            ref_start = ref_tr.stats.starttime + config.pre_window + pre_time
            ref_end = ref_tr.stats.starttime + config.pre_window + post_time
            npts = int((ref_end - ref_start) * sr + 1)
        except Exception as e:
            raise RuntimeError(f"Failed to load reference trace {ref_fname}: {str(e)}")

        # Process all traces
        tr_list, waveform_list, dist_list = [], [], []
        for _, row in df_array.iterrows():
            try:
                tr = read(row['fname'])[0]
                tr_cp = tr.copy()
                pkppre_end = tr_cp.stats.starttime + config.pre_window - 3
                pkpdf_start = tr_cp.stats.starttime + config.pre_window
                pkpdf_end = tr_cp.stats.starttime + config.pre_window + 7
                
                # Prepare trace data container
                stats = tr.stats.copy()
                stats.npts = npts
                stats.starttime = ref_start
                data = np.zeros(npts)

                # Fill data based on requested type
                if trim_type == 'pkppre':
                    tr_cp.trim(ref_start, pkppre_end)
                    tr_cp.taper(max_percentage=0.05)
                    data[:len(tr_cp.data)] = normalize(tr_cp.data)
                elif trim_type == 'pkpdf':
                    start_idx = int((pkpdf_start - ref_start)*config.sr)
                    tr_cp.trim(pkpdf_start, pkpdf_end)
                    tr_cp.taper(max_percentage=0.1)
                    data[start_idx:(start_idx+len(tr_cp.data))] = normalize(tr_cp.data)
                else:
                    tr_cp.trim(ref_start, ref_end)
                    tr_cp.taper(max_percentage=0.1)
                    data[:len(tr_cp.data)] = normalize(tr_cp.data)

                trace = Trace(data=data, header=stats)
                tr_list.append(trace)
                waveform_list.append(trace.data)
                dist_list.append(row['dist'])
            except Exception as e:
                raise RuntimeError(f"Failed to process trace {row['fname']}: {str(e)}")

        # Sort traces by distance with reference trace first
        sorted_indices = np.argsort([abs(d - med_dist) for d in dist_list])
        sorted_tr_list = [tr_list[i] for i in sorted_indices]
        st = Stream(traces=sorted_tr_list)
        stla, stlo = st[0].stats.sac.stla, st[0].stats.sac.stlo

        # Perform array analyses
        vespagram_data = vespagram(
            st, smin/dist_const, smax/dist_const, ssteps, ref_baz, winlen,
            stat=stat, phase_weighting=pws, n=order
        )
        fk_data = fk_analysis(st, smax/dist_const, freqmin, freqmax, ref_start, ref_end, fk_stat)
        
        # Find peak slowness from vespagram
        m = np.argmax(abs(vespagram_data))
        r, _ = divmod(m, vespagram_data.shape[1])
        vespa_slowness = s_list[r] * dist_const

        # Find peak slowness and back-azimuth from FK
        fkmax = np.unravel_index(np.argmax(fk_data), (len(st), len(st)))
        slow_x = np.linspace(-smax, smax, len(st))
        slow_y = np.linspace(-smax, smax, len(st))
        slow_x_max = slow_x[fkmax[1]]
        slow_y_max = slow_y[fkmax[0]]

        fk_slowness = np.hypot(slow_x_max, slow_y_max)
        fk_backazimuth = np.degrees(np.arctan2(slow_x_max, slow_y_max))
        if fk_backazimuth < 0:
            fk_backazimuth += 360

        # Generate plots if requested
        if plot:
            plot_array_analysis(
                waveform_list, dist_list, pre_time, post_time,
                vespagram_data, fk_data, smin, smax, vespa_slowness,
                fk_slowness, fk_backazimuth, array_name, stat
            )
        
        array_result = {'vespa_slowness': vespa_slowness,
                        'fk_slowness': fk_slowness,
                        'ref_baz': ref_baz,
                        'fk_baz': fk_backazimuth,
                        'stla': stla, 'stlo': stlo,
                        'vespagram_data': vespagram_data,
                        'fk_data': fk_data}

        return array_result

    except Exception as e:
        raise RuntimeError(f"Array analysis failed: {str(e)}") from e

def calculate_theoretical_slowness(sca_lat, sca_lon, sca_dp, stla, stlo, loc):
    """
    Calculate seismic wave slowness (ray parameter) using ObsPy TauPyModel.

    Args:
        sca_lat (float): Latitude of the scatterer (°).
        sca_lon (float): Longitude of the scatterer (°).
        sca_dp (float): Depth of the scatterer (km).
        stla (float): Station latitude (°).
        stlo (float): Station longitude (°).
        loc (str): Either 'receiver' (P-wave) or 'source' (PKP-wave).

    Returns:
        float: Ray parameter (s/deg), or np.nan if not found.
    """
    
    model = TauPyModel(model="ak135")
    distance_deg = DistAz(sca_lat, sca_lon, stla, stlo).getDelta()

    if loc == 'receiver':
        phase = "p"
        src_depth = sca_dp
        rcv_depth = 0
    else:
        phase = "PKP"
        src_depth = sca_dp
        rcv_depth = 0

    try:
        arrivals = model.get_travel_times(
            source_depth_in_km=src_depth,
            distance_in_degree=distance_deg,
            receiver_depth_in_km=rcv_depth,
            phase_list=[phase]
        )

        if arrivals:
            return arrivals[0].ray_param_sec_degree
        else:
            return np.nan
    except Exception as e:
        print(f"Error calculating slowness: {e}")
        return np.nan

    
def _process_grid_point(args):
    """
    Worker function for parallel processing of single grid point
    
    Args:
        args: Tuple containing:
            - grid_point: (latitude, longitude) coordinates
            - stla: Station latitude (degrees)
            - stlo: Station longitude (degrees) 
            - scatter_depth: Scattering depth (km)
            - loc_type: 'source' or 'receiver'
    
    Returns:
        Tuple: (scala, scalo, baz, slowness, location) or (None,)*5 if error
        
    Note:
        Designed to run in isolated worker processes
    """
    grid_point, stla, stlo, scatter_depth, loc_type = args
    try:
        scala, scalo = grid_point
        baz = DistAz(stla, stlo, scala, scalo).getBaz()
        slowness = calculate_theoretical_slowness(scala, scalo, scatter_depth, stla, stlo, loc=loc_type)
        return (scala, scalo, baz, slowness, loc_type)
    except Exception as e:
        print(f"Error processing {grid_point}: {str(e)}")
        return (None,)*5
    

def slowness_analysis(sca_infos, array_result, baz_range = 5, weights = (0.5, 0.5), 
                      show_progress = True):
    """
    Args:
        sca_infos: Dictionary containing:
            - 'source_grids': List of (lat,lon) tuples for source-side grids
            - 'receiver_grids': List of (lat,lon) tuples for receiver-side grids
            - 'max_scatter': Dictionary with 'depth' key (km)
        array_result: Dictionary containing:
            - 'fk_slowness': Observed slowness from FK (s/deg)
            - 'vespa_slowness': Observed slowness from vespagram (s/deg)
            - 'fk_backazimuth': Observed backazimuth (degrees)
            - 'stla': Reference station latitude
            - 'stlo': Reference station longitude
        baz_range: Back-azimuth range to filter grids.
        weights: Weights for (slowness_diff, baz_diff) in residual calculation
        show_progress: Whether to display tqdm progress bars
        
    Returns:
        pd.DataFrame: Contains all grid points with calculated residuals
    """
    # Determine scatter depths based on dominant side
    if sca_infos['dominant_side'] == 'source':
        source_scatter_depth = float(sca_infos['max_scatter']['depth'])
        receiver_scatter_depth = float(sca_infos['min_scatter']['depth'])
    else:
        source_scatter_depth = float(sca_infos['min_scatter']['depth'])
        receiver_scatter_depth = float(sca_infos['max_scatter']['depth'])
    stla, stlo = float(array_result['stla']), float(array_result['stlo'])

    # Filter grids and check whether their travel times smaller than PKIKP
    filtered_source_grids, filtered_receiver_grids = [], []
    for idx, source_grid in enumerate(sca_infos['source_grids']):
        source_scala, source_scalo = source_grid
        baz = DistAz(array_result['stla'], array_result['stlo'], source_scala, source_scalo).getBaz()
        if (abs(baz - array_result['ref_baz']) <= baz_range) and (sca_infos['source_stacks'][idx] > 0):
            filtered_source_grids.append(source_grid)
    for idx, receiver_grid in enumerate(sca_infos['receiver_grids']):
        receiver_scala, receiver_scalo = receiver_grid
        baz = DistAz(array_result['stla'], array_result['stlo'], receiver_scala, receiver_scalo).getBaz()
        if (abs(baz - array_result['ref_baz']) <= baz_range) and (sca_infos['receiver_stacks'][idx] > 0):
            filtered_receiver_grids.append(receiver_grid)
    
    try:
        tasks = [
            *( (pt, stla, stlo, source_scatter_depth, 'source') 
            for pt in filtered_source_grids ),
            *( (pt, stla, stlo, receiver_scatter_depth, 'receiver') 
            for pt in filtered_receiver_grids )
        ]

        # Process tasks serially
        if show_progress:
            results = [ _process_grid_point(task) for task in tqdm(tasks, desc="Calculating slowness by Taup") ]
        else:
            results = [ _process_grid_point(task) for task in tasks ]
    finally:
        pass

    
    # Filter valid results and create DataFrame
    valid_results = [r for r in results if r[0] is not None]
    df = pd.DataFrame(valid_results, 
                     columns=['scala', 'scalo', 'baz', 'slowness', 'location'])
    
    # Calculate weighted residuals
    slowness = (array_result['fk_slowness'] + array_result['vespa_slowness'])/2
    df['slowness_res'] = np.abs(df['slowness'] - slowness)
    df['baz_res'] = np.abs(df['baz'] - array_result['fk_baz'])
    df['total_res'] = weights[0]*df['slowness_res'] + weights[1]*df['baz_res']
    
    return df