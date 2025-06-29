#!/usr/bin/env python
# -*- coding=utf8 -*-
# Created Time: 2025-04-22
# Author: Marshon

import numpy as np

'''
Parameters for seismic waveform processing and deep learning model configuration.
'''

# File format
event_split = False # Waveforms of each event occupies a folder, if False: waveforms of all events occupies a folder

# Location grids parameters
diff = 20
step = 1
grid_len = len(np.arange(-diff, diff + step/2, step))
min_scadp, max_scadp, scadp_interval = 2391, 2891, 50
scadps = np.arange(min_scadp, max_scadp+scadp_interval, scadp_interval)
dist_range = np.arange(125, 143.5, 0.5)

# Waveform window parameters
pre_window = 70    # Time (in seconds) BEFORE theoretical PKP precursor arrival for the starttime of input waveform
post_window = 20   # Time (in seconds) BEFORE theoretical PKP precursor arrival for the endtime of input waveform
waveform_len = 400 # Total length of input waveform (in samples)
waveform_time = 20 # Time duration of input waveform (in seconds)
delta_t = 5        # Time increment step (in seconds) for window sliding or processing

# Signal processing parameters
sigma = 1        # Standard deviation for Gaussian label
sr = 20            # Sampling rate of waveforms (in Hz, samples per second)

# Deep learning training parameters
epochs = 200       # Number of complete passes through the training dataset
gnn_batch_size = 8     # Number of samples processed before GNN model update
unet_batch_size = 256     # Number of samples processed before U-Net model update
learning_rate = 0.01  # Step size at each iteration while moving toward a minimum loss

# Detection threshold
mph = 0.2       # Probability threshold for PKP precursor detection (0-1 range)
mpd = int(2*sigma*sr) # Minimum peak distance for PKP precursor detection
min_stations = 10
dist_cons = 4
az_cons = 5

# Colors used for plotting
c1="#C72228"
c2="#F98F34"
c3="#0C4E9B"
c4='#F5867F'
c5='#FFBC80'
c6='#6B98C4'

# Parameters for correct inner core attenuation
slope_east, intercept_east = -0.012459, 2.615225
slope_west, intercept_west = 0.013899, -0.802666