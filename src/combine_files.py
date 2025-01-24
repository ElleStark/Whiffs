# Script to combine multiple spatial chunks into a single array, save, and plot a heatmap
# Elle Stark January 2025

import cmasher as cmr
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# initialize arrays for combined data
peaks_array = np.empty((501, 400))
corrs_array = np.empty((501, 400))

dt = 0.02  # temporal resolution of data
w_dur = 1  # duration in sec
w_idx_dur = np.ceil(w_dur/dt)

n_idxs = 100  # number of x values per file
start_idx = 0

# Combine 5 pickled correlation files into single array
peaks_f_list  = ['flow_peaks_x0.0to0.15_y-0.3to 0.3_t1.2to180.0s_wdur1s.pkl', 'flow_peaks_x0.15to0.3_y-0.3to 0.3_t1.2to180.0s_wdur1s.pkl', 
                 'flow_peaks_x0.3to0.45_y-0.3to 0.3_t1.2to180.0s_wdur1s.pkl', 'flow_peaks_x0.45to0.6_y-0.3to 0.3_t1.2to180.0s_wdur1s.pkl', 'flow_peaks_x0.6to0.75_y-0.3to 0.3_t1.2to180.0s_wdur1s.pkl']

for pfile in peaks_f_list:
    with open(f'ignore/data/{pfile}', 'rb') as pf:
        peaks_dict = pickle.load(pf)
        for pt, peaks_list in peaks_dict.items():
            # Compute desired statistic and save at corresponding location
            peaks_array[start_idx+int(pt[0]/3), int(pt[1]/3)] = round(np.mean(peaks_list)*dt, 2)
    start_idx += n_idxs

# Combine 5 pickled timing files into a single array
corr_f_list = ['corr_data_x0.0to0.15_y-0.3to 0.3_t1.2to180.0s_wdur1s.pkl', 'corr_data_x0.15to0.3_y-0.3to 0.3_t1.2to180.0s_wdur1s.pkl', 
                 'corr_data_x0.3to0.45_y-0.3to 0.3_t1.2to180.0s_wdur1s.pkl', 'corr_data_x0.45to0.6_y-0.3to 0.3_t1.2to180.0s_wdur1s.pkl', 'corr_data_x0.6to0.75_y-0.3to 0.3_t1.2to180.0s_wdur1s.pkl']

start_idx = 0
for cfile in corr_f_list:
    with open(f'ignore/data/{cfile}', 'rb') as cf:
        corrs_dict = pickle.load(cf)
        for pt, corrs_list in corrs_dict.items():
            corrs_array[start_idx+int(pt[0]/3), int(pt[1]/3)] = np.mean(corrs_list)
    start_idx += n_idxs


### PLOTTING ###

# Timing centers heatmap
fig, ax = plt.subplots(figsize=(8, 6))
# index of half window duration for plotting colormaps
half_dur = np.ceil(w_idx_dur / 2)
absv = True  # decide if absolute value or raw distance from center is desired

# if using absolute value, create sequential colormap 0 to half duration
if absv:
    peaks_array = np.abs(peaks_array)
    cmap = cmr.lavender_r
    vmin = 0
    vmax = (vmin+half_dur)*dt
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

# if using positive and negative values, create diverging colormap with 0 at center
else:
    center = 0
    norm = colors.TwoSlopeNorm(vcenter=center, vmin=(center-half_dur)*dt, vmax=(center+half_dur)*dt)
    cmap = cmr.iceburn

plt.pcolormesh(peaks_array.T, cmap=cmap, norm=norm)
plt.colorbar()
ax.set_aspect('equal', adjustable='box')
plt.title(f'flow peak distance from odor gradient ridge, {w_dur}s windows')
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
plt.title(f'pearson correlation between odor gradient and FTLE signals, {w_dur}s windows')
# plt.savefig()
plt.show()


