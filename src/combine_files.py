# Script to combine multiple spatial chunks into a single array, save, and plot a heatmap
# Elle Stark January 2025

import cmasher as cmr
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from scipy import stats

# initialize arrays for combined data
peaks_array = np.empty((501, 401))
corrs_array = np.empty((501, 401))

dt = 0.02  # temporal resolution of data
w_dur = 1  # duration in sec
w_idx_dur = np.ceil(w_dur/dt)

n_idxs = 100  # number of x values per file
start_idx = 0
idx_per_src_idx = 3

# Combine 5 pickled correlation files into single array
peaks_f_list  = ['noWin_peaks_ftle_directOdor_x0.0to0.15_y-0.3to 0.3_t1.2to180.0s_wdur1s_othrs0.001.pkl', 'noWin_peaks_ftle_directOdor_x0.15to0.3_y-0.3to 0.3_t1.2to180.0s_wdur1s_othrs0.001.pkl', 
                 'noWin_peaks_ftle_directOdor_x0.3to0.45_y-0.3to 0.3_t1.2to180.0s_wdur1s_othrs0.001.pkl', 'noWin_peaks_ftle_directOdor_x0.45to0.6_y-0.3to 0.3_t1.2to180.0s_wdur1s_othrs0.001.pkl', 'noWin_peaks_ftle_directOdor_x0.6to0.75_y-0.3to 0.3_t1.2to180.0s_wdur1s_othrs0.001.pkl']

for pfile in peaks_f_list:
    with open(f'ignore/data/{pfile}', 'rb') as pf:
        peaks_dict = pickle.load(pf)
        for pt, peaks_list in peaks_dict.items():
            # Compute desired statistic and save at corresponding location
            if len(peaks_list) > 0:
            # mode_vals = round(stats.mode(abs(peaks_list))[1]/len(peaks_list), 2)
            # median_vals = round(np.median(abs(peaks_list))*dt, 2)
                p_ctr_4pct = np.sum((peaks_list>=-2) & (peaks_list<=2)) / len(peaks_list) * 100
                # count_peaks = len(peaks_list)
                peaks_array[start_idx+int(pt[0]/idx_per_src_idx), int(pt[1]/idx_per_src_idx)] = p_ctr_4pct
    start_idx += n_idxs - 1

# Combine 5 pickled timing files into a single array
corr_f_list = ['noWin_corr_data_ftle_directOdor_x0.0to0.15_y-0.3to 0.3_t1.2to180.0s_wdur1s_othrs0.001.pkl', 'noWin_corr_data_ftle_directOdor_x0.15to0.3_y-0.3to 0.3_t1.2to180.0s_wdur1s_othrs0.001.pkl', 
                 'noWin_corr_data_ftle_directOdor_x0.3to0.45_y-0.3to 0.3_t1.2to180.0s_wdur1s_othrs0.001.pkl', 'noWin_corr_data_ftle_directOdor_x0.45to0.6_y-0.3to 0.3_t1.2to180.0s_wdur1s_othrs0.001.pkl', 'noWin_corr_data_ftle_directOdor_x0.6to0.75_y-0.3to 0.3_t1.2to180.0s_wdur1s_othrs0.001.pkl']

start_idx = 0
for cfile in corr_f_list:
    with open(f'ignore/data/{cfile}', 'rb') as cf:
        corrs_dict = pickle.load(cf)
        for pt, corrs_list in corrs_dict.items():
            corrs_array[start_idx+int(pt[0]/3), int(pt[1]/3)] = np.mean(corrs_list)
    start_idx += n_idxs - 1


### PLOTTING ###

# Timing centers heatmap
fig, ax = plt.subplots(figsize=(8, 6))
# index of half window duration for plotting colormaps
half_dur = np.ceil(w_idx_dur / 2)
absv = True  # decide if absolute value or raw distance from center is desired

# if using absolute value, create sequential colormap 0 to half duration
if absv:
    # peaks_array = np.abs(peaks_array)
    cmap = cmr.lavender
    vmin = 0
    vmax = 50
    # vmax = (vmin+half_dur)*dt
    # vmax = None
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

# if using positive and negative values, create diverging colormap with 0 at center
else:
    center = 0
    norm = colors.TwoSlopeNorm(vcenter=center, vmin=(center-half_dur)*dt, vmax=(center+half_dur)*dt)
    cmap = cmr.iceburn

plt.pcolormesh(peaks_array.T, cmap=cmap, norm=norm)
plt.colorbar()
ax.set_aspect('equal', adjustable='box')
plt.title(f'percent of FTLE peaks within 0.04s of odor ridges, {w_dur}s windows, threshold=10E-4')
# plt.savefig()
plt.show()

# correlation heatmap 
fig, ax = plt.subplots(figsize=(8, 6))

# create diverging colormap with 0 at center
# norm = colors.TwoSlopeNorm(vcenter=0, vmin=-0.5, vmax=0.5)
# cmap = cmr.guppy_r
norm = colors.Normalize(vmin=0, vmax=0.4)
cmap = cmr.ember
plt.pcolormesh(corrs_array.T, cmap=cmap, norm=norm)
plt.colorbar()
ax.set_aspect('equal', adjustable='box')
plt.title(f'mean correlation between odor and FTLE signals, {w_dur}s windows, threshold=10E-4')
# plt.savefig()
plt.show()


