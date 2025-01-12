# Script to compute the relative timing of flow cue and odor GRADIENT ridges
# Elle Stark January 2025

import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger('GradientTiming')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
logger.addHandler(handler)
INFO = logger.info
WARN = logger.warning
DEBUG = logger.debug

def main():
    # load data, subset data by index as needed
    integration_T_idx = 62
    
    xmin = 399
    xmax = 401
    ymin = 99
    ymax = 101
    tmin = integration_T_idx + 5000
    tmax = tmin + 500
    time = 800

    xrange = slice(xmin, xmax)
    yrange = slice(ymin, ymax)
    time_lims = slice(tmin, tmax)
    
    # Obtain datasets from h5 file
    filename = 'D:/singlesource_2d_extended/Re100_0_5mm_50Hz_singlesource_2d.h5'
    with h5py.File(filename, 'r') as f:
        # x and y grids for plotting
        x_grid = f.get(f'Model Metadata/xGrid')[xrange, yrange]
        y_grid = f.get(f'Model Metadata/yGrid')[xrange, yrange]

        # Odor and detectable flow cue data
        odor_gradient = f.get('Odor Data/c_grad_spatial')[yrange, xrange, time_lims].transpose(2, 0, 1)
        # odor_data = f.get(f'Odor Data/c')[time_lims, xrange, yrange]
        # u_data = f.get(f'Flow Data/u')[time_lims, xrange, yrange]

        # Spatial resolution & time array
        dt_freq = f.get('Model Metadata/timeResolution')[0].item()
        dt = 1 / dt_freq  # convert from Hz to seconds
        time_vec = f.get('Model Metadata/timeArray')[time_lims]
        dx = f.get('Model Metadata/spatialResolution')[0].item()

    file2 = 'D:/singlesource_2d_extended/FTLE_extendedsim_T1_25_180s.h5'
    # For FTLE, need to adjust time indices above by integration time 
    with h5py.File(file2, 'r') as f2:
        flow_data = f2.get('FTLE_back_1_25s_finegrid')[tmin-integration_T_idx:tmax-integration_T_idx, ymin*2:ymax*2, xmin*2:xmax*2]
        # strain = f.get('maxPstrain')[time, xlims, ylims]

    # QC plot: time series of flow & odor data at select points
    DEBUG(f'odor gradient dims: {odor_gradient.shape}')
    testpt_x = 1
    testpt_y = 1
    flow_ts = flow_data[:, testpt_y*2, testpt_x*2]
    flow_ts = (flow_ts - np.min(flow_ts)) / (np.max(flow_ts)-np.min(flow_ts))
    odor_ts = odor_gradient[:, testpt_y, testpt_x]
    odor_ts = (odor_ts - np.min(odor_ts)) / (np.max(odor_ts) - np.min(odor_ts))
    
    # Plot each time series
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.plot(time_vec, flow_ts, label='FTLE')
    plt.plot(time_vec, odor_ts, label='odor gradient')

    # Plotting parameters
    plt.title(f'FTLE and odor gradients, [x, y] = [{round((xmin+1)*dx, 2)}, {round(0.3-(ymin+1)*dx, 2)}], t={round(tmin*dt, 1)} to {round(tmax*dt, 1)} s')
    plt.xlabel('time (s)')
    plt.ylabel('normalized value')
    plt.legend()
    plt.savefig(f'ignore/plots/c_grad_ts/FTLEodorgrad_x{round((xmin+1)*dx, 2)}_y{round(0.3-(ymin+1)*dx, 2)}]_t{round(tmin*dt, 1)}to{round(tmax*dt, 1)}s.png', dpi=300)
    plt.show()
 
    # define flow cue 


    # define odor cue






    # Find odor cue ridge indexes for each location
    

    # Select local window for each ridge
    w_dur = 0.50  # duration in sec
    w_idx_dur = w_dur/dt

    # Find timing of local max flow cue in each window


    # Display distribution of timing for location, if desired


    # Summarize distribution with characteristic statistic(s)

    
    # Compute for many locations


    # Display heat map of results



if __name__=='__main__':
    main()


