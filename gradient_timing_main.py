# Script to compute the relative timing of flow cue and odor GRADIENT ridges
# Elle Stark January 2025

from src import datafield 
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
    
    xmin = 1449
    xmax = 1451
    ymin = 99
    ymax = 101
    tmin = integration_T_idx 
    tmax = 9000

    xrange = slice(xmin, xmax)
    yrange = slice(ymin, ymax)
    time_lims = slice(tmin, tmax)
    
    # Obtain datasets from h5 file
    filename = 'D:/singlesource_2d_extended/Re100_0_5mm_50Hz_singlesource_2d.h5'
    with h5py.File(filename, 'r') as f:
        # x and y grids for plotting
        x_grid = f.get(f'Model Metadata/xGrid')[xrange, yrange]
        y_grid = f.get(f'Model Metadata/yGrid')[xrange, yrange]

        # Odor and/or detectable flow cue data
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
        ftle_data = f2.get('FTLE_back_1_25s_finegrid')[tmin-integration_T_idx:tmax-integration_T_idx, ymin*2:ymax*2, xmin*2:xmax*2]
        # strain = f.get('maxPstrain')[time, xlims, ylims]

    # define flow cue grid if needed for FTLE data
    FTLE_x = np.linspace(x_grid[0, 0], x_grid[-1, 0], ftle_data.shape[1]-1)  # FTLE grid finer than odor grid
    FTLE_y = np.linspace(y_grid[0, 0], y_grid[0, -1], ftle_data.shape[2]-1)  # FTLE grid finer than odor grid
    FTLE_x, FTLE_y = np.meshgrid(FTLE_x, FTLE_y)

    # create instance of class DataField for computations
    ftle_cgrad = datafield.DataField(odor_gradient, x_grid, y_grid, dt, dx, ftle_data, FTLE_x, FTLE_y, dt, round(FTLE_x[0, 1]-FTLE_x[0, 0], 5))
    DEBUG(f'fdx: {ftle_cgrad.fdx}')

    # QC plot: time series of flow & odor data at select points
    # ftle_cgrad.plot_time_series(time_vec, 1, 1, xmin, ymin)

    # Find odor cue ridge indexes for each location
    compute_pts = [(1, 1)]  # list of tuples for each point to be analyzed
    odor_threshold = 10E-5
    # required window size for each ridge
    w_dur = 1  # duration in sec
    w_idx_dur = w_dur/dt
    for pt in compute_pts:
        odor_ridges = ftle_cgrad.find_odor_ridges(odor_threshold, pt[1], pt[0], distance=int(w_idx_dur/2))

        # QC plot: time series of flow & odor data at select points with ridges
        # ftle_cgrad.plot_time_series(time_vec, 1, 1, xmin, ymin, ridges=odor_ridges[0], save=True)

        # Find timing of local max flow cue peaks in each window
        flow_peaks, corrs = ftle_cgrad.find_loc_max_fcue(pt[1], pt[0], odor_ridges[0], w_idx_dur, corr=True, yidx=pt[1], xidx=pt[0])
        valid_fpeaks = flow_peaks[flow_peaks>-100]
        valid_corrs = corrs[corrs>-100]
        INFO(f'Number of gradient ridges with corresponding flow cue peaks: {len(valid_fpeaks)}')
        INFO(f'average correlation: {np.mean(valid_corrs)}')

        # Box plot of relative flow peak timing
        # plt.boxplot(valid_fpeaks)
        # plt.ylim=(-w_idx_dur/2, w_idx_dur/2)
        # plt.show()

        # Histogram of relative flow peak timing
        plt.hist(valid_fpeaks, bins=15)
        plt.title(f'relative FTLE timing, x{round((xmin+1)*dx, 2)} y{0.3-round((ymin+1)*dx)} t=1.25 to 180 s, window={w_dur} s, n_peaks={len(valid_fpeaks)}')
        plt.savefig(f'ignore/plots/c_grad_ts/FTLEodorgrad_hist_x{round((xmin+1)*dx, 2)}_y{round((ymin+1)*dx)}]_\
                t{round(np.min(time_vec), 1)}to{round(np.max(time_vec), 1)}s_wdur{w_dur}s.png', dpi=300)
        plt.show()

        # Boxplot of correlation values w/ mean & std dev labeled
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.boxplot(valid_corrs)
        plt.title(f'pearson correlations, avg {round(np.mean(valid_corrs), 2)}, std {round(np.std(valid_corrs), 2)}, x{round((xmin+1)*dx, 2)} y{round(0.3-(ymin+1)*dx, 2)} t=1.25 to 180 s, window={w_dur}s')
        # ax.text(1.5, 2.5, f'mean: {np.mean(valid_corrs)}', fontsize=12)
        # ax.text(1.5, 2.5, 'test', fontsize=12)
        #ax.text(2, 2.5, f'std dev: {np.std(valid_corrs)}', fontsize=12)
        plt.savefig(f'ignore/plots/c_grad_ts/FTLEodorgrad_corrs_x{round((xmin+1)*dx, 2)}_y{round(0.3-(ymin+1)*dx, 2)}]_\
                        t{round(np.min(time_vec), 1)}to{round(np.max(time_vec), 1)}s.png', dpi=300)
        plt.show()
    
    
    # Display distribution of timing for location, if desired


    # Summarize distribution with characteristic statistic(s)

    
    # Compute for many locations


    # Display heat map of results



if __name__=='__main__':
    main()


