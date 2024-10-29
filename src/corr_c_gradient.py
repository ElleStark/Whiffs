# Script to compute correlations at various lag times for a given location
# Uses changes to log intensity of the odor field vs FTLE 
# Elle Stark November 2024

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd

def main():
# Subset data by index as needed
    integration_time = 62

    xmin = 1400
    xmax = 1401
    ymin = 600
    ymax = 601
    tmin = integration_time + 38
    tmax = tmin + 501

    xrange = slice(xmin, xmax)
    yrange = slice(ymin, ymax)
    time_lims = slice(tmin, tmax)

    # Obtain datasets from h5 file
    filename = 'D:/singlesource_2d_extended/Re100_0_5mm_50Hz_singlesource_2d.h5'
    with h5py.File(filename, 'r') as f:
        # x and y grids for plotting

        # Odor and flow cue data
        odor_data = f.get(f'Odor Data/c')[time_lims, xmin-1:xmax+1, (1200-(ymax+1)):(1200-(ymin-1))].transpose(0, 2, 1)

    #     # Spatial resolution & time array
        dt_freq = f.get('Model Metadata/timeResolution')[0].item()
        dt = 1 / dt_freq  # convert from Hz to seconds
        dx = f.get('Model Metadata/spatialResolution')[0].item()
        time_array = f.get('Model Metadata/timeArray')[time_lims]

    file2 = 'D:/singlesource_2d_extended/FTLE_extendedsim_T1_25_180s.h5'
    # For FTLE, need to adjust time indices above by integration time indexes
    with h5py.File(file2, 'r') as f2:
        flow_data = f2.get('FTLE_back_1_25s_finegrid')[(tmin-integration_time):(tmax-integration_time), ymin*2:ymax*2, xmin*2:xmax*2]

    flow_data = flow_data[:, 1, 1]
    flow_data_gradient = np.gradient(flow_data)
    # Normalize 0 to 1
    flow_data_gradient = (flow_data_gradient - (np.min(flow_data_gradient))) / (np.max(flow_data_gradient - np.min(flow_data_gradient)))
    flow_data = (flow_data - np.min(flow_data)) / (np.max(flow_data) - np.min(flow_data))
    flow_data[np.where(flow_data==0)] = 0.1
    flow_data_log = np.log10(flow_data)
    flow_data_gradient[np.where(flow_data_gradient==0)]=10**(-1)
    flow_gradient_log = np.log10(flow_data_gradient)
    # flow_gradient_log = np.nan_to_num(flow_gradient_log, nan=0)

    # Compute time gradient of concentration signal
    odor_data = odor_data[:, 1, 1]
    grad_t = np.gradient(odor_data)
    # Normalize 0 to 1
    hline = (10**(-4) - np.min(grad_t)) / (np.max(grad_t) - np.min(grad_t))
    grad_t = (grad_t - np.min(grad_t)) / (np.max(grad_t) - np.min(grad_t))
    
    grad_t[np.where(grad_t==0)]=10**(-1)
    grad_t_norm = grad_t/10**(-2)
    # grad_t_norm = np.nan_to_num(grad_t_norm, nan=10**(-5))
    log_gradient = np.log10(grad_t)
    hline = np.log10(hline)
    # log_gradient = np.nan_to_num(log_gradient, nan=0)

    # QC: PLOT line plots of odor gradient & FTLE
    # fig, ax = plt.subplots()
    # plt.plot(time_array, log_gradient, label='C time gradient', color='#1984c5')
    # plt.plot(time_array, flow_gradient_log, label='FTLE time gradient, T=1.25s', color='#de6e56')
    # # plt.plot(time_array, flow_data_log, label='log FTLE, T=1.25s')
    # plt.hlines(hline, tmin*dt, tmax*dt, color='#1984c5', linestyle='dashed')
    # # plt.yscale("log")
    # plt.ylim((-0.55), (0))
    # ax.axes.fill_between(np.squeeze(time_array), log_gradient, hline, where=(log_gradient > hline), color='#1984c5', alpha=0.4)
    # plt.title('Time series of log normalized FTLE gradient and log normalized C gradient, x=0.7m y=0m, t=[2, 12]s')
    # plt.legend()
    # plt.show()

    # Compute Pearson correlation for time series of log normalized signal gradients

    # First version: lag of 0
    


if __name__=='__main__':
    main()



