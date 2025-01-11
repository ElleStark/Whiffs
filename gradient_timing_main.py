# Script to compute the relative timing of flow cue and odor GRADIENT ridges
# Elle Stark January 2025

import h5py
import numpy as np


def main():
    # load data, subset data by index as needed
    xmin = 795
    xmax = 805
    ymin = 100
    ymax = 1100
    tmin = 30
    tmax = 9001
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
        odor_data = f.get(f'Odor Data/c')[time_lims, xrange, yrange]
        # u_data = f.get(f'Flow Data/u')[time_lims, xrange, yrange]

        # Spatial resolution & time array
        dt_freq = f.get('Model Metadata/timeResolution')[0].item()
        dt = 1 / dt_freq  # convert from Hz to seconds

    file2 = 'D:/singlesource_2d_extended/FTLE_extendedsim_T1_25_180s.h5'
    # For FTLE, need to adjust time indices above by integration time 
    with h5py.File(file2, 'r') as f2:
        flow_data = f2.get('FTLE_back_1_25s_finegrid')[tmin-62:tmax-62, ymin*2:ymax*2, xmin*2:xmax*2]
        # strain = f.get('maxPstrain')[time+62, xlims, ylims]

    # define flow cue 


    # define odor cue
    odor = np.flipud(odor)

    # compute odor gradient
    

    # QC plot: time series of flow & odor cues 


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


