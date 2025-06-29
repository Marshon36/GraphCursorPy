#!/usr/bin/env python
# -*- coding=utf8 -*-
# Created Time: 2025-04-22
# Author: Marshon

import numpy as np
import pandas as pd
from obspy import read
from obspy.core import UTCDateTime
import scipy.stats as stats
import random, os

from graphcursorpy.functions4common import DistAz, normalize

import torch
import torch.nn as nn
import torch.utils.data as tdata
import torch_geometric.nn as gnn
import torch_geometric.data as gdata
from torch_geometric.loader import DataLoader

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model_name, model_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_name = model_name
        self.model_path = model_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), f'{self.model_path}{self.model_name}.pt')
        self.val_loss_min = val_loss

def normalize_dist(dist):
    """
    Normalize inverse distances using exponential decay to create edge attributes in [0,1] range.
    
    Applies the function: edge_attr = 1 - exp(-1/dist) to transform distances into a 
    normalized similarity measure where:
    - 0 distance → maximum similarity (1.0)
    - ∞ distance → minimum similarity (0.0)
    
    Args:
        dist (float): Distance between stations (must be ≥ 0)
        
    Returns:
        float: Normalized edge attribute value between 0 and 1
    """

    if dist == 0:
        # Handle zero distance case (perfect similarity)
        edge_attr = 1
    else:
        # Apply exponential decay normalization
        edge_attr = 1 - np.exp(-1/dist)
    return edge_attr

def cal_edge(pos_init, locations, norm=True):
    """
    Calculate edges and their attributes (distance or normalized distance) for a graph, 
    where nodes are connected based on their geographic positions (latitude, longitude).

    Args:
        pos_init (int): Number of nodes.
        locations (array-like): List or array of node locations (latitude, longitude).
        norm (bool): Whether to normalize the distance or not.

    Returns:
        edges (np.ndarray): Array of edge pairs.
        edges_attr (np.ndarray): Array of edge attributes (e.g., 1/distance or normalized distance).
    """
    
    # Precompute the number of edges we will have
    num_edges = pos_init * (pos_init - 1)
    
    # Initialize arrays for edges and edge attributes
    edges = np.zeros((num_edges, 2), dtype=int)
    edges_attr = np.zeros((num_edges, 1))

    index = 0  # Keep track of the current position in the arrays
    
    for i in range(pos_init):
        lat1, lon1 = locations[i]
        for j in range(pos_init):
            if i != j:
                lat2, lon2 = locations[j]
                
                # Calculate distance between node i and node j using DistAz
                dist = DistAz(lat1, lon1, lat2, lon2).getDelta()
                
                # Calculate edge attribute (normalized or inverse of distance)
                if norm:
                    edge_attr = normalize_dist(dist)
                else:
                    edge_attr = 1 / dist

                # Add the edge and the attribute to the arrays
                edges[index] = [i, j]
                edges_attr[index] = edge_attr
                index += 1
    return edges, edges_attr

def make_label(PKPpre_idx, config):
    """
    Create a Gaussian-shaped label centered at the PKPpre arrival index.

    The label is a 2D array with two rows:
    - Row 0: Gaussian peak centered at PKPpre_idx (main target)
    - Row 1: Complement of the Gaussian (residual)

    Args:
        PKPpre_idx (int): The index of the PKPpre arrival.
        config (module): A Python module containing configuration settings. 

    Returns:
        label (np.ndarray): 2 x N array with target and residual labels
    """
    sigma = config.sigma                 # Standard deviation of Gaussian (in seconds)
    sr = config.sr                       # Sampling rate (samples per second)
    total_time = config.pre_window + config.post_window  # Total time window in seconds
    total_samples = int(total_time * sr)  # Total number of samples

    # Initialize the label array (1D)
    PKPpre_target = np.zeros(total_samples)

    # Define window around PKPpre arrival: +/- 3 sigma (in samples)
    half_window_samples = int(3 * sigma * sr)
    start_idx = max(PKPpre_idx - half_window_samples, 0)
    end_idx = min(PKPpre_idx + half_window_samples, total_samples)

    # Generate Gaussian curve with peak at PKPpre_idx
    x = np.linspace(-3 * sigma, 3 * sigma, end_idx - start_idx)
    gaussian = stats.norm.pdf(x, loc=0, scale=sigma)
    gaussian /= gaussian.max()  # Normalize to peak = 1

    # Insert Gaussian into target array
    PKPpre_target[start_idx:end_idx] = gaussian

    # Construct 2-row label: [target, residual]
    residual = 1.0 - PKPpre_target
    label = np.vstack([PKPpre_target, residual])  # Shape: (2, N)

    return label

def custom_cross_entropy_loss(output, target, eps=1e-5):
    """
    Custom cross entropy loss function for regression problems where both
    output and target are arrays representing probability distributions.

    Args:
        output (Tensor): Predicted probability distribution (after softmax).
        target (Tensor): Ground truth probability distribution.

    Returns:
        loss (Tensor): Computed cross-entropy loss.
    """
    loss = target * torch.log(output + eps)
    loss = loss.mean(-1) # Mean along sample dimension 
    loss = loss.sum(-1)  # Sum along pick dimension
    loss = loss.mean()   # Mean over batch axis
    return -loss

class DownsampleLayer_T(nn.Module):
    def __init__(self,in_ch,out_ch):
        """
        Downsample layer used in a U-Net-like architecture. This layer consists of two convolutional
        blocks followed by batch normalization and LeakyReLU activations, and a downsampling (pooling) operation.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels after the downsampling layer.
        """
        super(DownsampleLayer_T, self).__init__()

        # Two Conv1D layers followed by BatchNorm and LeakyReLU activation
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv1d(in_channels=in_ch,out_channels=out_ch,kernel_size=7,stride=1,padding=3),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=7, stride=1,padding=3),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Max pooling layer for downsampling the spatial dimensions
        self.downsample=nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=2,padding = 2)
        )

    def forward(self,x):
        """
        Forward pass of the DownsampleLayer_T.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_ch, sequence_length).

        Returns:
            out (torch.Tensor): Output after Conv1D + BatchNorm + ReLU layers, 
                              passed to the next deeper layer (residual path).
            out_2 (torch.Tensor): Downsampled output after max pooling, passed to the next layer.
        """
        # Apply two convolutional layers, each followed by batch normalization and LeakyReLU
        out=self.Conv_BN_ReLU_2(x)

        # Downsample the output using max pooling
        out_2=self.downsample(out)

        return out,out_2
    
    
class UpSampleLayer_T(nn.Module):
    def __init__(self,in_ch,out_ch):
        """
        Up-sample layer used in a U-Net-like architecture. This layer consists of two convolutional
        blocks followed by batch normalization and LeakyReLU activations, and a transposed convolution 
        for upsampling (increasing the resolution).

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels after the upsampling layer.
        """
        super(UpSampleLayer_T, self).__init__()

        # Two Conv1D layers followed by BatchNorm and LeakyReLU activation
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=7, stride=1,padding=3),
            nn.BatchNorm1d(out_ch*2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=7, stride=1,padding=3),
            nn.BatchNorm1d(out_ch*2),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Transposed convolution layer for upsampling (increasing the resolution)
        self.upsample=nn.Sequential(
            nn.ConvTranspose1d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=7,stride=2,padding=3,output_padding=1),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self,x,out):
        """
        Forward pass of the UpSampleLayer_T.

        Args:
            x (torch.Tensor): Input tensor from the previous layer (with smaller resolution).
            out (torch.Tensor): Feature map from the corresponding downsampling layer for concatenation (skip connection).

        Returns:
            cat_out (torch.Tensor): Concatenated output of the upsampled feature and the feature map from the encoder.
        """
        # Apply two convolutional layers, each followed by batch normalization and LeakyReLU
        x_out=self.Conv_BN_ReLU_2(x)

        # Upsample the output using a transposed convolution
        x_out=self.upsample(x_out)

        # Concatenate the upsampled feature map with the corresponding downsampled feature map (skip connection)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out

class UNet(torch.nn.Module):
    """
    U-Net architecture adapted for 1D time series data (e.g., seismic waveforms).
    This model is typically used for sequence labeling or segmentation tasks like phase picking.

    Architecture:
        - Downsampling (encoder) layers using custom temporal convolution blocks.
        - Upsampling (decoder) layers with skip connections to preserve high-resolution features.
        - Final convolutional layer maps features to output classes (e.g., PKPpre, Noise).
    
    Args:
        drop (float): Dropout probability applied at the bottleneck and output stages.
    """
    def __init__(self, drop = 0):
        super(UNet, self).__init__()
        self.dropprob = drop

        # Define the number of channels for the downsampling and upsampling layers
        out_channels = [4, 8, 16, 32, 64]
        result_channel = 2

        # Downsample layers (encoder part of the U-Net)
        self.d1=DownsampleLayer_T(1,out_channels[0])
        self.d2=DownsampleLayer_T(out_channels[0],out_channels[1])
        self.d3=DownsampleLayer_T(out_channels[1],out_channels[2])
        self.d4=DownsampleLayer_T(out_channels[2],out_channels[3])

        # Upsample layers (decoder part of the U-Net)
        self.u1=UpSampleLayer_T(out_channels[3],out_channels[3])
        self.u2=UpSampleLayer_T(out_channels[4],out_channels[2])
        self.u3=UpSampleLayer_T(out_channels[3],out_channels[1])
        self.u4=UpSampleLayer_T(out_channels[2],out_channels[0])
        
        # Output CNN layers
        self.final_conv = nn.Sequential(nn.Conv1d(out_channels[1],result_channel,kernel_size=7, stride=1, padding=3))
        self.Softmax = nn.Softmax(dim=1)
        self.droplayer = nn.Dropout(p=self.dropprob)
  
    def forward(self, x):
        # Downsampling path (encoder)
        out_1,out=self.d1(x)
        out_2,out=self.d2(out)
        out_3,out=self.d3(out)
        out_4,out=self.d4(out)
        out=self.droplayer(out)

        # Upsampling path (decoder)
        out=self.u1(out,out_4)
        out=self.u2(out,out_3)
        out=self.u3(out,out_2)
        out=self.u4(out,out_1)
        out=self.droplayer(out)

        # Pass through the first convolution layer to reduce channels and apply Softmax activation
        unet_out=self.final_conv(out)
        unet_out=self.Softmax(unet_out)
        return unet_out

class UNetDataset(tdata.Dataset):
    """
    PyTorch Dataset class for U-Net-based seismic signal detection.

    Supports waveform loading, normalization, label creation, and optional data augmentation.
    
    Modes:
        - 'train', 'valid', 'test': waveform and PKPpre label provided
        - 'predict': waveform only, without label

    Args:
        df (pd.DataFrame): Input metadata table containing at least:
                           - 'fname': path to SAC file
                           - 'dist': epicentral distance
                           - 'pkppre': arrival time string or placeholder
        mode (str): Mode indicator ['train', 'valid', 'test', 'predict']
        config (module): Configuration module with fields like:
                         - sr, waveform_time, pre_window, post_window, waveform_len, sigma
        data_aug (bool): Whether to apply time-shift augmentation (train mode only)
    """
    def __init__(self, df, mode, config, data_aug = True):
        self.df = df
        self.data_length = len(df)
        self.mode = mode
        self.config = config
        self.data_aug = data_aug

    def _random_cut_data(self, waveform, label, pkppre_arrival):
        # Compute slide time for data augmentation
        if pkppre_arrival > 0:
            min_slide = pkppre_arrival - self.config.waveform_time + self.config.sigma
            max_slide = min([pkppre_arrival - self.config.sigma, 
                             self.config.pre_window + self.config.post_window - self.config.waveform_time - self.config.sigma])
            slide_time = random.uniform(min_slide, max_slide)
        else:
            slide_time = random.uniform(0, 3) + self.config.pre_window - self.config.waveform_time
        slide_idx = int(slide_time*self.config.sr)
        waveform = waveform[slide_idx:slide_idx+self.config.waveform_len]
        label = label[:, slide_idx:slide_idx+self.config.waveform_len]
        return waveform, label

    def __getitem__(self, idx):
        fname_z = self.df.iloc[idx]['fname']
        dist = float(self.df.iloc[idx]['dist'])
        data_len = int((self.config.pre_window + self.config.post_window)*self.config.sr)
        if os.path.exists(fname_z):
            if (self.mode == 'train') or (self.mode == 'valid') or (self.mode == 'test'):
                '''
                Train, valid and test mode: with PKPpre labels.
                '''

                # Read waveform data and detrend
                tr_z = read(fname_z)[-1]
                tr_z.detrend('linear')
                tr_z.detrend('demean')
                data = tr_z.data[:data_len]

                # Normalize the waveform
                waveform = normalize(data)

                # Create label for waveforms with pkppre
                pkppre = self.df.iloc[idx]['pkppre']
                if len(str(pkppre)) > 3:
                    pkppre_arrival = UTCDateTime(pkppre) - tr_z.stats.starttime
                    pkppre_index = int(pkppre_arrival*self.config.sr)
                    label = make_label(pkppre_index, self.config)
                # Create label for waveforms without pkppre
                else:
                    pkppre_arrival = 0
                    PKPpre_target = np.zeros((1, data_len))
                    residual = np.ones((1, data_len))
                    label = np.concatenate((PKPpre_target, residual))
                
                # Data augmentation in 'train' mode
                if self.mode == 'train':
                    if self.data_aug:
                        waveform, label = self._random_cut_data(waveform, label, pkppre_arrival)
                    else:
                        slide_time = self.config.pre_window - self.config.waveform_time
                        slide_idx = int(slide_time*self.config.sr)
                        waveform = waveform[slide_idx:slide_idx+self.config.waveform_len]
                        label = label[:, slide_idx:slide_idx+self.configwaveform_len]
                else:
                    slide_time = self.config.pre_window - self.config.waveform_time
                    slide_idx = int(slide_time*self.config.sr)
                    waveform = waveform[slide_idx:slide_idx+self.config.waveform_len]
                    label = label[:, slide_idx:slide_idx+self.config.waveform_len]
                waveform = waveform.reshape((1, len(waveform)))
                
                sample = {'fname': fname_z, 'waveform': waveform,
                        'label': label, 'dist': dist}
            
            else:
                '''
                Predict mode: without PKPpre labels.
                '''
                # Read waveform data and detrend
                tr_z = read(fname_z)[-1]
                tr_z.detrend('linear')
                tr_z.detrend('demean')
                
                # Trim waveform and normalization
                pkpdf_idx = self.config.pre_window * self.config.sr
                waveform = tr_z.data[int(pkpdf_idx-self.config.waveform_len):int(pkpdf_idx)]
                
                # Normalize the waveform
                waveform = normalize(waveform)
                waveform = waveform.reshape((1, len(waveform)))
                sample = {'fname': fname_z, 'waveform': waveform, 'dist': dist}
            return sample
    
    def __len__(self):
        return self.data_length

class GNN_layer_GATv2(torch.nn.Module):
    """
    A single GATv2-based GNN layer with edge attributes and optional dropout.

    Args:
        in_ch (int): Input feature dimension.
        out_ch (int): Output feature dimension.
        edge_dim (int): Dimensionality of edge attributes.
        head (int): Number of attention heads.
        drop (float): Dropout rate.
    """
    def __init__(self,in_ch,out_ch,edge_dim,head,drop):
        super(GNN_layer_GATv2, self).__init__()
        self.Graph_agg = gnn.Sequential('x, edge_index, edges_attr', [
            (gnn.GATv2Conv(in_channels = in_ch, 
                   out_channels = out_ch, edge_dim = edge_dim,
                   heads = head, dropout = drop, concat = False), 'x, edge_index, edges_attr -> x'),
        ])

        # self.bn = nn.BatchNorm1d(out_ch),
        self.ac = nn.LeakyReLU(negative_slope=0.2)
        
    def forward(self,x,edge_index,edges_attr):
        out = self.Graph_agg(x, edge_index, edges_attr)
        out = self.ac(out)
        return out
    
class GNN(torch.nn.Module):
    """
    GNN is a dual-stream Graph Neural Network model designed for binary classification 
    of seismic waveform data (e.g., PKPpre vs. noise). It operates directly on waveform features 
    using a graph-based architecture.

    Architecture:
        - Two independent GATv2 layers for each class (PKPpre / Noise).
        - Post-processing via a 1D CNN layer.
        - Softmax normalization across class channels.

    Attributes:
        config (object): Configuration object containing waveform parameters.
    """
    def __init__(self, config):
        super(GNN, self).__init__()
        self.config = config

        result_channel = 2
        self.cnn2 = nn.Conv1d(result_channel, result_channel, kernel_size=7, stride=1, padding=3)

        # Graph Neural Network layers
        self.gnn1 = GNN_layer_GATv2(self.config.waveform_len,self.config.waveform_len,1,2,0)
        self.gnn2 = GNN_layer_GATv2(self.config.waveform_len,self.config.waveform_len,1,2,0)
        
        # Activation functions
        self.ReLU = nn.LeakyReLU(negative_slope=0.2)
        self.Softmax = nn.Softmax(dim=1)
  
    def forward(self, data):
        """
        Forward pass of the GraphUnet model.
        
        Parameters:
        - data: A batch containing the input features, edge indices, positions, and edge attributes.
        
        Returns:
        - unet_out: Output from the U-Net architecture.
        - gnn_out: Output from the Graph Neural Network layers.
        """
        # Extract the necessary data from the input
        x, edge_index, edges_attr = data.x, data.edge_index, data.edges_attr
        
        # Process the first class (PKPpre) through the first GNN layer
        gnn_pkppre_out = self.gnn1(x.squeeze(1), edge_index, edges_attr)
        
        # Process the second class (Noise) through the first GNN layer
        gnn_noise_out = self.gnn2(x.squeeze(1), edge_index, edges_attr)

        # Concatenate the GNN outputs for both classes
        gnn_out = torch.cat((gnn_pkppre_out.unsqueeze(1), gnn_noise_out.unsqueeze(1)), dim=1)
        
        # Apply the second CNN layer and Softmax for final predictions
        gnn_out = self.cnn2(gnn_out)
        gnn_out = self.Softmax(gnn_out)
        return gnn_out
    
class GraphCursor(torch.nn.Module):
    """
    GraphCursor is a hybrid neural network model that combines a U-Net with 
    dual Graph Attention Network (GATv2) layers to detect seismic phases 
    from waveform inputs.

    Architecture overview:
        - UNet: Extracts multiscale features from input waveforms.
        - GNN (x2): Processes spatial-temporal relationships using graph structure.
        - CNN + Softmax: Refines and outputs class probabilities.

    Attributes:
        config (object): Configuration object with waveform-related settings.
        res (any): Custom resolution identifier or input metadata.
        dropprob (float): Dropout probability for regularization in UNet.
    """
    def __init__(self, config, res, drop = 0):
        super(GraphCursor, self).__init__()
        self.config = config
        self.res = res
        self.dropprob = drop

        result_channel = 2

        # UNet layers
        self.unet = UNet(self.dropprob)
        
        # CNN layer
        self.cnn2 = nn.Conv1d(result_channel,result_channel,kernel_size=7, stride=1, padding=3)

        # Projection layer
        self.proj = nn.Conv1d(1, 2, kernel_size=1)

        # Graph Neural Network layers
        self.gnn1 = GNN_layer_GATv2(self.config.waveform_len, self.config.waveform_len, 1, 2, 0)
        self.gnn2 = GNN_layer_GATv2(self.config.waveform_len, self.config.waveform_len, 1, 2, 0)
        
        # Activation functions
        self.Softmax = nn.Softmax(dim=1)
  
    def forward(self, data):
        """
        Forward pass of the GraphCursor model.
        
        Args:
            data: A batch containing the input features, edge indices, positions, and edge attributes.
        
        Returns:
            unet_out: Output from the U-Net architecture.
            gnn_out: Output from the Graph Neural Network layers.
        """
        # Extract the necessary data from the input
        x, edge_index, edges_attr = data.x, data.edge_index, data.edges_attr
        
        # Pass through the UNet layer
        unet_out=self.unet(x)
        gnn_input = unet_out

        # Process the first class (PKPpre) through the first GNN layer
        gnn_pkppre_out = self.gnn1(gnn_input[:, 0, :].squeeze(1), edge_index, edges_attr)
        
        # Process the second class (Noise) through the first GNN layer
        gnn_noise_out = self.gnn2(gnn_input[:, 1, :].squeeze(1), edge_index, edges_attr)
        
        # Concatenate the GNN outputs for both classes
        gnn_out = torch.cat((gnn_pkppre_out.unsqueeze(1), gnn_noise_out.unsqueeze(1)), dim=1)
        
        # Apply the second CNN layer and Softmax for final predictions
        gnn_out = self.cnn2(gnn_out)
        gnn_out = self.Softmax(gnn_out)

        return unet_out, gnn_out

class GraphCursorDataset(gdata.Dataset):
    def __init__(self, df, arrays, mode, config, data_aug = True, add_noise = False, noise_ratio = 0):
        """
        GraphCursorDataset is designed to handle different modes (train, valid, test, predict) 
        for GraphCursor that use seismic waveform data and location-based graph information.

        Args:
            df (pd.DataFrame): 
                Contains seismic waveform metadata with columns for:
                Required:
                - event: Event identifiers (e.g., event_id)
                - array: Array identifiers (e.g., array_id)
                - fname: SAC waveform paths
                - dist: Epicentral distances
                Optional:
                - Phase timing information (is needed in train, valid and test mode)
                - Azimuth, back-azimuth, pkppre_SNR, pkpdf_SNR

            arrays (list):
                Seismic array names used to filter and group stations 
                (e.g., ['TA_20200101', 'TA_20200102'] for temporary arrays)

            mode (str):
                Operational mode specifying dataset usage:
                - 'train': Training mode (enables data augmentation)
                - 'valid': Validation mode 
                - 'test': Testing mode
                - 'predict': Inference/prediction mode

            config (module): A Python module containing configuration settings. 

            data_aug (bool, optional):
                When True (default), applies data augmentation techniques:
                - Waveform time-warping (training mode only)
                - Random noise injection 
                - Amplitude scaling
                Disable for validation/testing/prediction.
        """
        gdata.Dataset.__init__(self)
        self.df = df
        self.arrays = arrays
        self.data_length = len(arrays)
        self.mode = mode
        self.config = config
        self.data_aug = data_aug
        self.add_noise = add_noise
        self.noise_ratio = noise_ratio

    def _remove_duplicate_nodes(self, waveforms, noises, labels, dists, nodes, locations, mode):
        """
        Remove duplicate nodes (stations) based on their locations to avoid multiple 
        entries for the same station.

        Args:
            waveforms, noises, labels, dists, nodes, locations: Data for the current arrays.

        Returns:
            unique_waveforms, unique_noises, unique_labels, unique_dists, unique_nodes, unique_locations
        """
        unique_waveforms, unique_noises, unique_labels, unique_dists, unique_nodes, unique_locations = [], [], [], [], [], []
        seen = set()
        
        for i, loc in enumerate(locations):
            loc_tuple = tuple(loc)  # Convert [lat, lon] to a tuple for hashing
            if loc_tuple not in seen:
                seen.add(loc_tuple)
                if mode in ['train', 'valid', 'test']:
                    unique_waveforms.append(waveforms[i])
                    unique_noises.append(noises[i])
                    unique_labels.append(labels[i])
                    unique_dists.append(dists[i])
                    unique_nodes.append(nodes[i])
                    unique_locations.append(loc)
                else:
                    unique_waveforms.append(waveforms[i])
                    unique_dists.append(dists[i])
                    unique_nodes.append(nodes[i])
                    unique_locations.append(loc)
        return unique_waveforms, unique_noises, unique_labels, unique_dists, unique_nodes, unique_locations

    def _random_cut_data(self, waveforms, labels, pkppre_arrivals):
        """
        Perform time-window cropping with NumPy for batch waveforms and labels.

        Args:
            waveforms (np.ndarray): shape (N, 1, T)
            labels (np.ndarray): shape (N, 2, T)
            pkppre_arrivals (list[float]): list of PKPpre arrival times in seconds

        Returns:
            cut_waveforms (np.ndarray): shape (N, 1, waveform_len)
            cut_labels (np.ndarray): shape (N, 2, waveform_len)
        """
        sr = self.config.sr
        waveform_time = self.config.waveform_time
        waveform_len = self.config.waveform_len
        sigma = self.config.sigma
        pre_window = self.config.pre_window
        post_window = self.config.post_window

        if len(pkppre_arrivals) > 0:
            min_slide = max(pkppre_arrivals) - waveform_time + sigma
            max_slide = min([min(pkppre_arrivals) - sigma,
                            pre_window + post_window - waveform_time - sigma])
            slide_time = random.uniform(min_slide, max_slide)
        else:
            slide_time = random.uniform(0, 3) + pre_window - waveform_time

        slide_idx = int(slide_time * sr)

        # Perform slicing using NumPy
        cut_waveforms = np.array(waveforms)[:, :, slide_idx:(slide_idx + waveform_len)]
        cut_labels = np.array(labels)[:, :, slide_idx:(slide_idx + waveform_len)]

        return cut_waveforms, cut_labels

    def _random_remove_data(self, waveforms, labels, dists, nodes, locations):
        """
        Randomly remove some waveforms and labels for data augmentation.

        Args:
            waveforms (np.ndarray): shape (N, 1, T)
            labels (np.ndarray): shape (N, 2, T)
            dists (list or np.ndarray): shape (N,)
            nodes (list or np.ndarray): shape (N,) or (N, D)
            locations (list or np.ndarray): shape (N, 2)

        Returns:
            Filtered arrays after random removal.
        """
        N = len(dists)
        if N > 1:
            remove_num = random.randint(0, N // 2)
            indices_to_remove = sorted(random.sample(range(N), remove_num))
            indices_to_keep = np.setdiff1d(np.arange(N), indices_to_remove)

            # Use numpy to index remaining elements
            waveforms = waveforms[indices_to_keep]
            labels = labels[indices_to_keep]
            dists = np.array(dists)[indices_to_keep]
            nodes = np.array(nodes)[indices_to_keep]
            locations = np.array(locations)[indices_to_keep]

        return waveforms, labels, dists, nodes, locations

    def __getitem__(self, index):
        """
        Retrieve data based on index. Depending on the mode, this function processes 
        waveform data, generates PKPpre labels (if applicable), and calculates edges 
        and edge attributes for the graph representation.
        
        Args:
            index (int): The index of the arrays to be retrieved.
        
        Returns:
            data (gdata.Data): Processed data in PyTorch Geometric Data format.
        """

        if self.mode in ['train', 'valid', 'test']:
            array = self.arrays[index]
            df_array = pd.DataFrame(self.df[self.df['array'] == array])
            df_array.index = range(len(df_array))
            data_len = int((self.config.pre_window + self.config.post_window)*self.config.sr)
            '''
            Train, valid and test mode: with PKPpre labels.
            '''
            # Process each waveform in the arrays
            waveforms, labels, dists, locations, nodes = [], [], [], [], []
            pkppre_arrivals = []
            for i in range(len(df_array)):
                fname_z = df_array.iloc[i]['fname']
                pkppre = df_array.iloc[i]['pkppre']
                dist = float(df_array.iloc[i]['dist'])

                # Read waveform data and apply basic preprocessing
                tr_z = read(fname_z)[-1]
                tr_z.detrend('linear')
                tr_z.detrend('demean')
                data = tr_z.data[:data_len]
                if np.std(data) != 0:
                    data_norm = normalize(data)

                    # Extract signal
                    signal = np.zeros(data_len)
                    signal[self.config.waveform_len:] = data_norm[self.config.waveform_len:]

                    # Get station latitude and longitude
                    stla, stlo = float(tr_z.stats.sac.stla), float(tr_z.stats.sac.stlo)

                    # Create label for waveforms with pkppre
                    if len(str(pkppre)) > 3:
                        pkppre_arrival = UTCDateTime(pkppre) - tr_z.stats.starttime
                        pkppre_arrivals.append(pkppre_arrival)
                        pkppre_index = int(pkppre_arrival*self.config.sr)
                        label = make_label(pkppre_index, self.config)
                    # Create label for waveforms without pkppre
                    else:
                        PKPpre_target = np.zeros((1, data_len))
                        residual = np.ones((1, data_len))
                        label = np.concatenate((PKPpre_target, residual))
                    
                    # Store the processed data
                    waveforms.append([signal])
                    labels.append(label)
                    dists.append(dist)
                    nodes.append(dist/180) # Normalize distance
                    locations.append([stla, stlo])
                    
            if self.mode == 'train':
                # Data augmentation in 'train' mode
                if self.data_aug:
                    waveforms, labels = self._random_cut_data(waveforms, labels, pkppre_arrivals)
                    waveforms, labels, dists, nodes, locations = self._random_remove_data(
                        waveforms, labels, dists, nodes, locations)
                else:
                    # No data augmentation
                    pkpdf_idx = int(self.config.pre_window*self.config.sr)
                    waveforms = np.array(waveforms)[:, :, int(pkpdf_idx-self.config.waveform_len):pkpdf_idx]
                    labels = np.array(labels)[:, :, int(pkpdf_idx-self.config.waveform_len):pkpdf_idx]
                
                # Convert to NumPy arrays
                waveforms = np.array(waveforms)
                labels = np.array(labels)
                nodes = np.array(nodes)

                # Calculate edges and their attributes between nodes (stations)
                edges, edges_attr = cal_edge(np.array(waveforms).shape[0], locations)

                # Convert to PyTorch tensors
                waveforms_torch = torch.tensor(waveforms, dtype=torch.float, requires_grad=True)
                labels_torch = torch.tensor(labels, dtype=torch.float, requires_grad=True)
                nodes_torch = torch.tensor(nodes, dtype=torch.float, requires_grad=True)
                edge_index_torch = torch.tensor(edges, dtype=torch.int64).t().contiguous()
                edges_attr_torch = torch.tensor(edges_attr, dtype=torch.float, requires_grad=True)
                
                # Return data in PyTorch Geometric format
                data = gdata.Data(x = waveforms_torch, edge_index = edge_index_torch, y = labels_torch,
                                pos = nodes_torch, edges_attr = edges_attr_torch)
            else:
                slide_time = self.config.pre_window - self.config.waveform_time
                slide_idx = int(slide_time*self.config.sr)
                new_waveforms, new_labels = [], []
                for i in range(len(waveforms)):
                    waveform = waveforms[i][0][slide_idx:slide_idx+self.config.waveform_len]
                    new_waveforms.append([waveform])
                    label = labels[i]
                    label = label[:, slide_idx:slide_idx+self.config.waveform_len]
                    new_labels.append(label)

                # Convert to NumPy arrays
                waveforms = np.array(new_waveforms)
                labels = np.array(new_labels)
                nodes = np.array(nodes)

                # Calculate edges and their attributes between nodes (stations)
                edges, edges_attr = cal_edge(np.array(waveforms).shape[0], locations)

                # Convert to PyTorch tensors
                waveforms_torch = torch.tensor(waveforms, dtype=torch.float, requires_grad=True)
                labels_torch = torch.tensor(labels, dtype=torch.float, requires_grad=True)
                nodes_torch = torch.tensor(nodes, dtype=torch.float, requires_grad=True)
                edge_index_torch = torch.tensor(edges, dtype=torch.int64).t().contiguous()
                edges_attr_torch = torch.tensor(edges_attr, dtype=torch.float, requires_grad=True)
                
                # Return data in PyTorch Geometric format
                data = gdata.Data(x = waveforms_torch, edge_index = edge_index_torch, y = labels_torch,
                                pos = nodes_torch, edges_attr = edges_attr_torch)

    
        else:
            '''
            Predict mode: without PKPpre labels.
            '''
            array = self.arrays[index]
            df_array = pd.DataFrame(self.df[self.df['array'] == array])
            df_array.index = range(len(df_array))
            data_len = int((self.config.pre_window + self.config.post_window)*self.config.sr)
            fnames, waveforms, dists, locations, nodes = [], [], [], [], []
            
            # Process each waveform for prediction
            for i in range(len(df_array)):
                fname_z = df_array.iloc[i]['fname']
                dist = float(df_array.iloc[i]['dist'])

                # Read waveform data and apply basic preprocessing
                tr_z = read(fname_z)[-1]
                tr_z.detrend('linear')
                tr_z.detrend('demean')

                # Get station latitude and longitude
                stla, stlo  = float(tr_z.stats.sac.stla), float(tr_z.stats.sac.stlo)

                # Trim and normalize the waveform based on slide time
                pkpdf_idx = self.config.pre_window * self.config.sr
                waveform_z = tr_z.data[int(pkpdf_idx-self.config.waveform_len):int(pkpdf_idx)]

                if self.add_noise:
                    noise_start = int((self.config.pre_window - 40) * self.config.sr)
                    noise_end = int((self.config.pre_window - 20) * self.config.sr)
                    noise_window = tr_z.data[noise_start:noise_end]
                    waveform_z = waveform_z + self.noise_ratio * noise_window

                if np.std(waveform_z) != 0:
                    waveform_z = normalize(waveform_z)

                    # Store the processed data
                    fnames.append(fname_z)
                    waveforms.append([waveform_z])
                    dists.append(dist)
                    nodes.append(dist/180)
                    locations.append([stla, stlo])

            # Convert to NumPy arrays
            waveforms = np.array(waveforms)
            nodes = np.array(nodes)

            # Calculate edges and their attributes between nodes (stations)
            edges, edges_attr = cal_edge(np.array(waveforms).shape[0], locations)
            
            # Convert to PyTorch tensors
            waveforms = torch.tensor(waveforms, dtype=torch.float, requires_grad=True)
            nodes = torch.tensor(nodes, dtype=torch.float, requires_grad=True)
            edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()
            edges_attr = torch.tensor(edges_attr, dtype=torch.float, requires_grad=True)
            
            # Return data in PyTorch Geometric format
            data = gdata.Data(x = waveforms, edge_index = edge_index,
                            pos = nodes, edges_attr = edges_attr)

        return data

    def __len__(self):
        return self.data_length
    
def run_graphcursor_predictions(df, arrays, config, model_path='models/GraphCursor_v0.pt', 
                                output_dir='predictions/'):
    """
    Run GraphCursor model predictions on seismic data and save results as .npy files.
    
    Args:
        df (pd.DataFrame): 
            Contains seismic waveform metadata with columns for:
            Required:
            - event: Event identifiers (e.g., event_id)
            - array: Array identifiers (e.g., array_id)
            - fname: SAC waveform paths
            - dist: Epicentral distances
            Optional:
            - Phase timing information (is needed in train, valid and test mode)
            - Azimuth, back-azimuth, pkppre_SNR, pkpdf_SNR
        arrays (list): List of array names to process
        config (module): A Python module containing configuration settings. 
        model_path (str): Path to the pre-trained GraphCursor model
        output_dir (str): Directory to save prediction outputs
        
    Returns:
        None (saves prediction files to disk)
        
    Example:
        df = pd.read_csv('2009-04-09T08:10:54.020Z_0.csv')
        arrays = ['NET_A', 'NET_B']
        run_graphcursor_predictions(df, arrays)
    """

    # Set up device and load model
    device = torch.device("cpu")
    
    try:
        # Initialize model
        model = GraphCursor(config, False)
        
        # Load pre-trained weights with device handling
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Create prediction dataset and loader
        predict_dataset = GraphCursorDataset(df, arrays, 'predict', config)
        predict_loader = DataLoader(predict_dataset, batch_size=1)
        
        # Process each array
        with torch.no_grad():
            for idx, predict_batch in enumerate(predict_loader):
                array = arrays[idx]
                print(f'Processing {array}')
                
                # Filter current array data
                df_array = df[df['array'] == array].copy()
                df_array.reset_index(drop=True, inplace=True)
                predict_batch = predict_batch.to(device)
                
                # Get predictions
                _, gnn_pred = model(predict_batch)
                gnn_pred = gnn_pred.cpu().numpy()
                pkppre_predicts = gnn_pred[:, 0, :]  # Extract PKP precursors
                
                # Save predictions
                for i in range(pkppre_predicts.shape[0]):
                    fname = df_array.iloc[i]['fname']
                    basename = os.path.basename(fname)
                    output_fname = f"{basename.split('.SAC')[0]}.npy"
                    if config.event_split:
                        directory = os.path.dirname(fname)
                        event = directory.split('/')[-1]
                        if not os.path.exists(os.path.join(output_dir, event)):
                            os.mkdir(os.path.join(output_dir, event))
                        np.save(os.path.join(output_dir, event, output_fname), pkppre_predicts[i])
                    else:
                        np.save(os.path.join(output_dir, output_fname), pkppre_predicts[i])
                    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise


def split_dataset(df, seed=42):
    """
    Split the dataset into training, validation, and test sets based on unique events.

    Args:
        df (pd.DataFrame): The complete dataset, must contain a column named 'event'.
        config (object): A config object, currently unused but kept for extensibility.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (df_train, df_val, df_test), three DataFrames with no overlapping events.
    """
    # Remove rows containing infinity values
    df = df[~df.isin([float('inf')]).any(axis=1)].reset_index(drop=True)

    # Get unique event IDs
    unique_events = df['event'].drop_duplicates()

    # Randomly select 90% of events for training + validation
    train_val_events = unique_events.sample(frac=0.9, random_state=seed)

    # From that, select 80% for training, remaining 20% for validation
    train_events = train_val_events.sample(frac=0.8, random_state=seed)
    val_events = train_val_events.drop(train_events.index)

    # Remaining 10% of events go to test set
    test_events = unique_events.drop(train_val_events.index)

    # Subset the DataFrame based on event labels
    df_train = df[df['event'].isin(train_events)]
    df_val = df[df['event'].isin(val_events)]
    df_test = df[df['event'].isin(test_events)]

    return df_train, df_val, df_test