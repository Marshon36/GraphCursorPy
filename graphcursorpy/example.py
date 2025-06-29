#!/usr/bin/env python
# -*- coding=utf8 -*-
# Created Time: 2025-04-22
# Author: Marshon

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__name__)))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import config
from graphcursorpy.functions4AI import run_graphcursor_predictions
from graphcursorpy.functions4pkppre import screen_predictions, locate_scatter_migration, calculate_relative_strength
from graphcursorpy.functions4common import plot_waveforms_predictions, plot_stacked_map

# Load precomputed arrival time catalog:
# -------------------------
# Earthquake to scattering source paths
df_eq2source_sca = pd.read_csv('precomputed_files/eq2source_sca_catalog_scadp_ak135.csv')

# Scattering source to station paths
df_source_sca2sta = pd.read_csv('precomputed_files/source_sca2sta_catalog_scadp_ak135.csv')

# Earthquake to receiver scattering paths
df_eq2receiver_sca = pd.read_csv('precomputed_files/eq2receiver_sca_catalog_scadp_ak135.csv')

# Receiver scattering to station paths
df_receiver_sca2sta = pd.read_csv('precomputed_files/receiver_sca2sta_catalog_scadp_ak135.csv')

# Load preprocessed stacked envelope for calculating the strength of scatters:
# ------------------------------
stack_map = np.load('precomputed_files/global_stacked_envelope.npy')
stack_fitted_line = np.load('precomputed_files/global_stacked_envelope_fitted_line.npy')

def main():
    """Main processing pipeline for seismic scatter detection"""
    
    # ======================
    # 1. Data Preparation
    # ======================
    # Load seismic array metadata
    array_id = '2009-04-09T08:10:54.020Z_0'  # Unique array identifier
    data_fname = 'example_data/'
    df = pd.read_csv(f'{data_fname}{array_id}.csv')  # Load station metadata
    pred_path = 'predictions/'
    
    # # Uncomment to generate predictions (first-time run)
    # run_graphcursor_predictions(
    #     df=df,
    #     arrays=[array_id],
    #     config=config,
    #     model_path='models/GraphCursor.pt',
    #     output_dir=pred_path
    # )

    # ======================
    # 2. Data Visualization
    # ======================
    df_array = df[df['array'] == array_id].copy().reset_index(drop=True)
    plot_waveforms_predictions(df_array, pred_path, config)  # visualize waveforms

    # ======================
    # 3. Signal Detection
    # ======================
    # Screen stations based on prediction quality
    mpd, mph = config.mpd, config.mph
    filtered_dict = screen_predictions(
        df_array, 
        pred_path,
        mph,  # Minimum peak probability
        mpd,  # Minimum peak separation
        config # A Python module containing configuration settings
    )
    
    # Calculate array geometry statistics
    dist_span = filtered_dict['df_array']['dist'].max() - filtered_dict['df_array']['dist'].min()
    az_span = filtered_dict['df_array']['azimuth'].max() - filtered_dict['df_array']['azimuth'].min()
    print(f"\n  {' ORIGINAL ARRAY GEOMETRY ':=^60}")
    print(f"  {'Distance Span:':<20}{dist_span:>8.3f}°")
    print(f"  {'Azimuth Span:':<20}{az_span:>8.3f}°")

    # ======================
    # 4. Scatter Location
    # ======================
    if len(filtered_dict['df_array']) >= config.min_stations:
        # 4.1 Migration-based location
        print(f"\n{' MIGRATION-BASED LOCATION START ':*^80}")
        sca_infos = locate_scatter_migration(filtered_dict, df_eq2source_sca, df_source_sca2sta, 
                                             df_eq2receiver_sca, df_receiver_sca2sta, config, plot=True)
        print(f"  {' PRIMARY SCATTER LOCATION ':=^60}")
        print(f"  {'Location:':<20}{sca_infos['dominant_side'].capitalize()} side")
        print(f"  {'Longitude:':<20}{sca_infos['sca_lon_max']:>8.3f}°")
        print(f"  {'Latitude:':<20}{sca_infos['sca_lat_max']:>8.3f}°")
        print(f"  {'Depth:':<20}{sca_infos['depth']:>8} km")

        print(f"  {' SECONDARY SCATTER LOCATION ':=^60}")
        print(f"  {'Location:':<20}{sca_infos['secondary_side'].capitalize()} side")
        print(f"  {'Longitude:':<20}{sca_infos['sca_lon_min']:>8.3f}°")
        print(f"  {'Latitude:':<20}{sca_infos['sca_lat_min']:>8.3f}°")
        print(f"{' MIGRATION-BASED LOCATION END ':*^80}")

        # 4.2 Scatter strength calculation
        print(f"\n{' SCATTER STRENGTH CALCULATION START ':*^80}")
        relative_strength = calculate_relative_strength(filtered_dict['fnames'], stack_map, stack_fitted_line, config)
        relative_strength_attenu = calculate_relative_strength(
                    filtered_dict['fnames'], stack_map, stack_fitted_line, config, True)
        print(f"  {' RELATIVE SCATTER STRENGTH ':=^60}")
        print(f"  {'Strength:':<20}{relative_strength:>8.3f}")
        print(f"  {'Strength after inner core attenuation correction:':<20}{relative_strength_attenu:>8.3f}")
        print(f"{' SCATTER STRENGTH CALCULATION END ':*^80}")

if __name__ == '__main__':
    main()