# Script to compute the relative timing of flow cue and odor GRADIENT ridges
# Elle Stark January 2025

from src import datafield 
from matplotlib import colors
import h5py
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle


logger = logging.getLogger('GradientTiming')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
logger.addHandler(handler)
INFO = logger.info
WARN = logger.warning
DEBUG = logger.debug

def main():
    save_files = True  # Save data?
    
    # load data, subset data by index as needed
    integration_T_idx = 62
    
    xmin = 600
    xmax = 901
    ymin = 0
    ymax = 1201
    tmin = integration_T_idx 
    tmax = 9000

    xrange = slice(xmin, xmax)
    yrange = slice(ymin, ymax)
    time_lims = slice(tmin, tmax)
    
    # Obtain datasets from h5 file
    filename = 'D:/singlesource_2d_extended/Re100_0_5mm_50Hz_singlesource_2d.h5'
    with h5py.File(filename, 'r') as f:
        # x and y grids for plotting
        x_grid = f.get(f'Model Metadata/xGrid')[xrange, yrange].T
        y_grid = f.get(f'Model Metadata/yGrid')[xrange, yrange].T

        # Odor and/or detectable flow cue data
        odor_gradient = f.get('Odor Data/c_grad_spatial')[yrange, xrange, time_lims]
        # odor_data = f.get(f'Odor Data/c')[time_lims, xrange, yrange]
        # u_data = f.get(f'Flow Data/u')[time_lims, xrange, yrange]

        # Spatial resolution & time array
        dt_freq = f.get('Model Metadata/timeResolution')[0].item()
        dt = 1 / dt_freq  # convert from Hz to seconds
        time_vec = f.get('Model Metadata/timeArray')[time_lims]
        dx = f.get('Model Metadata/spatialResolution')[0].item()

    # file2 = 'D:/singlesource_2d_extended/FTLE_extendedsim_T1_25_180s.h5'
    file2 = 'D:/singlesource_2d_extended/FTLE_extendedsim_T1_25_180s.h5'
    # For FTLE, need to adjust time indices above by integration time 
    with h5py.File(file2, 'r') as f2:
        ftle_data = f2.get('FTLE_back_1_25s_finegrid')[tmin-integration_T_idx:tmax-integration_T_idx, ymin*2:ymax*2, xmin*2:xmax*2]
        # strain = f2.get('maxPstrain')[time_lims, yrange, xrange]

    # define flow cue grid if needed for FTLE data
    FTLE_x = np.linspace(x_grid[0, 0], x_grid[0, -1], ftle_data.shape[2]-1)  # FTLE grid finer than odor grid
    FTLE_y = np.linspace(y_grid[0, 0], y_grid[-1, 0], ftle_data.shape[1]-1)  # FTLE grid finer than odor grid
    FTLE_x, FTLE_y = np.meshgrid(FTLE_x, FTLE_y)

    flow_data = ftle_data
    flow_x = FTLE_x
    flow_y = FTLE_y

    # create instance of class DataField for computations
    flow_cgrad = datafield.DataField(odor_gradient, x_grid, y_grid, dt, dx, flow_data, flow_x, flow_y, dt, round(flow_x[0, 1]-flow_x[0, 0], 5))
    DEBUG(f'fdx: {flow_cgrad.fdx}')

    # QC plot: time series of flow & odor data at select points
    # ftle_cgrad.plot_time_series(time_vec, 1, 1, xmin, ymin)

    # QC plot: spatial snapshot of flow & odor data at a few times
    time = 500
    fig, ax = plt.subplots()
    # FTLE params
    colormap_f = plt.cm.Greys
    vmin_f = 0
    vmax_f = 8
    plt.pcolormesh(flow_x, flow_y, flow_data[time, :-2, :-2], vmin=vmin_f, vmax=vmax_f, cmap=colormap_f)
    # plt.pcolormesh(flow_x, flow_y, flow_data[time, :-1, :-1], cmap=colormap_f)

    # Odor overlay
    odor_gradient[odor_gradient<0.0001] = 0.0001
    colormap = plt.cm.Reds
    vmin = 0.0001
    vmax = 1
    plt.pcolormesh(x_grid, y_grid, (odor_gradient[:-1, :-1, time]), cmap=colormap, norm=colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.4)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    # Find odor cue ridge indexes for each location
    x_locs = np.floor(np.linspace(0, xmax-xmin-1, 100))
    y_locs = np.floor(np.linspace(0, ymax-ymin-1, 400))
    compute_pts = list((int(xloc), int(yloc)) for yloc in y_locs for xloc in x_locs)
    # compute_pts = [(0, 0), (10, 50), (20, 100), (30, 150), (40, 200), (50, 250), (60, 300), (70, 350), (80, 400)]  # list of index tuples for each (x, y) point to be analyzed
    odor_threshold = 10E-5
    # required window size for each ridge
    w_dur = 1  # duration in sec
    w_idx_dur = np.ceil(w_dur/dt)

    # Loop through desired locations for computing relative timing distributions
    counter = 0
    for pt in compute_pts:
        DEBUG(f'point idxs: {pt}')
        odor_ridges = flow_cgrad.find_odor_ridges(odor_threshold, pt, distance=np.ceil(w_idx_dur/2))

        # QC plot: time series of flow & odor data at select points with ridges
        # ftle_cgrad.plot_time_series(time_vec, 1, 1, xmin, ymin, ridges=odor_ridges[0], save=True)

        # Find timing of local max flow cue peaks in each window
        title_id = f'ftle, x{round((xmin+1)*dx, 2)} to{round((xmax)*dx, 2)} y{round(0.3-(ymax)*dx, 2)} to {round(0.3-(ymin)*dx, 2)} t={round(np.min(time_vec), 1)} to {round(np.max(time_vec), 1)} s, window={w_dur} s'
        file_id = f'ftle_x{round((xmin+1)*dx, 2)}to{round((xmax)*dx, 2)}_y{round(0.3-(ymax)*dx, 2)}to {round(0.3-(ymin)*dx, 2)}_t{round(np.min(time_vec), 1)}to{round(np.max(time_vec), 1)}s_wdur{w_dur}s'
        flow_cgrad.find_loc_max_fcue(pt, odor_ridges[0], w_idx_dur, title_id, file_id, corr=True, yidx=pt[1], xidx=pt[0], hist=False, box=False, QC=False, ftle=True)

        DEBUG(f'Number of gradient ridges with corresponding flow cue peaks: {len(flow_cgrad.flow_peaks[pt])}')
        DEBUG(f'average correlation: {np.mean(flow_cgrad.f_o_corrs[pt])}')

        counter += 1

        if (counter % 100) ==0:
            INFO(f'{counter} pts computed. {round(counter/len(compute_pts)*100, 1)}% complete.')


    if save_files:
        # save flow peak and correlation data
        with open(f'ignore/data/FTLE_peaks_{file_id}.pkl', 'wb') as f:
            pickle.dump(flow_cgrad.flow_peaks, f)
            INFO('flow peak data saved.')
        with open(f'FTLE_corr_data_{file_id}.pkl', 'wb') as f:
            pickle.dump(flow_cgrad.f_o_corrs, f)
            INFO('correlation data saved')

    # Summarize distributions with characteristic statistic(s)
    flow_cgrad.compute_timing_centers('mean')
    flow_cgrad.compute_correlation_stats()

    # Display heat maps of results
    flow_cgrad.plot_timing_ctrs_heatmap(w_idx_dur, title_id=title_id)
    flow_cgrad.plot_correlation_heatmap(title_id)

    DEBUG('program done')

if __name__=='__main__':
    main()


