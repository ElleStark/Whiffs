# Main script to compute relative timing of flow and odor cue signals in a chaotic flow field
# Uses conditional averaging of whiff-based time windows
# Elle Stark, CU Boulder Environmental Fluid Mechanics Lab, Aug 2024

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd

def main():
# Subset data by index as needed
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

        # Odor and flow cue data
        odor_data = f.get(f'Odor Data/c')[time_lims, xrange, yrange]
        # odor_data = f.get(f'Odor Data/c')[time+30, xrange, yrange].transpose()
        # u_data = f.get(f'Flow Data/u')[time_lims, xrange, yrange]

    #     # Spatial resolution & time array
        dt_freq = f.get('Model Metadata/timeResolution')[0].item()
        dt = 1 / dt_freq  # convert from Hz to seconds
        dx = f.get('Model Metadata/spatialResolution')[0].item()
    #     time_array = f.get('Model Metadata/timeArray')[time_lims]


    file2 = 'D:/singlesource_2d_extended/FTLE_extendedsim_T1_25_180s.h5'
    # For FTLE, need to adjust time indices above by integration time 
    with h5py.File(file2, 'r') as f2:
        flow_data = f2.get('FTLE_back_1_25s_finegrid')[tmin-62:tmax-62, xmin*2:xmax*2, ymin*2:ymax*2]
        # flow_data = f2.get('FTLE_back_0_6s_finegrid')[tmin-30:tmax-30, :, :]
        # flow_data = f2.get('FTLE_back_0_6s_finegrid')[time, ymin*2:ymax*2, xmin*2:xmax*2]

    # # Flip velocity data upside-down for correct computations
    # odor_data = np.flip(odor_data, axis=2)
    # # dudt = np.gradient(u_data, dt, axis=0)
    # # flow_data = dudt

    # # QC Plot: FTLE & odor data
    # # odor_data = odor_data[0, :, :]
    # # flow_data = flow_data[0, :, :]
    # # FTLE_x = np.linspace(x_grid[0, 0], x_grid[-1, 0], len(flow_data[0, 0, :])-1)
    # # FTLE_y = np.linspace(y_grid[0, 0], y_grid[0, -1], len(flow_data[0, :, 0])-1)
    # # FTLE_x, FTLE_y = np.meshgrid(FTLE_x, FTLE_y)
    # # fig, ax = plt.subplots()
    # # # FTLE params
    # # colormap = plt.cm.Greys
    # # vmin = -2
    # # vmax = 8
    # # plt.contourf(FTLE_x, FTLE_y, flow_data[0 , :-1, :-1], 100, vmin=vmin, vmax=vmax, cmap=colormap, alpha=1)
    # # plt.colorbar()

    # # # Odor overlay
    # # colormap = plt.cm.Reds
    # # vmin = 0.0005
    # # vmax = 1
    # # plt.pcolormesh(x_grid, y_grid, odor_data[0, :-1, :-1], cmap=colormap, norm=colors.LogNorm(vmin=vmin, vmax=vmax), alpha=0.3)
    # # plt.colorbar()

    # # ax.set_aspect('equal', adjustable='box')
    # # plt.show()


    # # Normalize flow data array on [0, 1] (min-max normalization)
    # # flow_data = (flow_data - np.min(flow_data)) / (np.max(flow_data) - np.min(flow_data))
    # # Constrain FTLE to positive values
    # # flow_data[np.where(flow_data<=0)] = 0.01
    # # flow_data = flow_data / 8.25
    # flow_data = (flow_data - (-4.5)) / (8.25 - (-4.25))  # global min/max is about -4.5 to 8.25

    # # Define concentration level for start & end of whiff
    whiff_threshold = 0.001

    # CHOOSING X AND Y BASED ON CONTINUOUS INDICES
    x_spacing = 1
    y_spacing = 1
    x = list(range(0, len(odor_data[0, :, 0]), x_spacing))
    y = list(range(0, len(odor_data[0, 0, :]), y_spacing))
    x, y = np.meshgrid(x, y)
    locs_list = list(zip(x.flatten(), y.flatten())) 

    # Initialize dataframe for storing whiff information if needed
    # all_whiffs_df = pd.DataFrame(columns=['x_idx', 'y_idx', 'start_idx', 'end_idx', 'length_idx', 'duration_sec', 'flowcue_start',
    #                                     'flowcue_end', 'E_F_prior', 'E_F_during', 'E_F_after', 'E_F_ratio'])
    
    stats_list = []

    def compute_expected_vals(row):
        # val_list = []
        flowstart = int(row['flowcue_start'])
        flowend = int(row['flowcue_end'])
        whiffstart = int(row['start_idx'])
        whiffend = int(row['end_idx'])

        prior_ts = flow_ts[flowstart:whiffstart]
        # print(prior_ts)

        row['E_F_prior'] = prior_ts.mean()
        row['E_F_during'] = (flow_ts[whiffstart:whiffend]).mean()
        row['E_F_after'] = (flow_ts[whiffend:flowend]).mean()
        row['E_F_ratio'] = np.max([row['E_F_prior'], row['E_F_after']]) / row['E_F_during']

        return row

    # Loop through each location to compute conditional average ratios
    for x, y in locs_list:
        # Obtain time series of data for flow and odor cues
        odor_ts = pd.Series(odor_data[:, x, y])
        flow_ts = pd.Series(flow_data[:, x*2, y*2])
        # print(flow_ts.size)

        odor_df = pd.DataFrame({'C': odor_ts})
        odor_df['next_C'] = odor_df.C.shift(-1)
        odor_df['time'] = time_array

        odor_df['start_whiff'] = ((odor_df.next_C > whiff_threshold) & (odor_df.C <= whiff_threshold))
        odor_df['end_whiff'] = ((odor_df.C > whiff_threshold) & (odor_df.next_C <= whiff_threshold))

        # If no whiffs are present at this location, move to next location
        if (not any(odor_df['start_whiff'])) | (not any(odor_df['end_whiff'])):
            continue

        # Get indices of start and end of each whiff
        start_idxs = odor_df.index[odor_df['start_whiff']].tolist()
        end_idxs = odor_df.index[odor_df['end_whiff']].tolist()

        if start_idxs[-1] > end_idxs[-1]:
            start_idxs = start_idxs[:-1]

        # Now, if lists are empty, move to next location
        if (len(start_idxs) == 0) | (len(end_idxs) == 0):
            continue
        elif end_idxs[0] < start_idxs[0]:
            end_idxs = end_idxs[1:]

        # Ensure equal numbers of start and end whiffs
        for i in range(min(len(start_idxs), len(end_idxs))):
            if end_idxs[i] < start_idxs[i]:
                del end_idxs[i]
            if start_idxs[i] > end_idxs[i]:
                del start_idxs[i]

        try:
            whiff_df = pd.DataFrame({'start_idx': start_idxs, 'end_idx': end_idxs})
        except ValueError as error:
            print(f'whiff start/end indices error at {x}, {y}: {error}')
            continue

        whiff_df.insert(loc=0, column='y_idx', value=y)
        whiff_df.insert(loc=0, column='x_idx', value=x)

        # Calculate duration of whiffs
        whiff_df['length_idx'] = whiff_df['end_idx'] - whiff_df['start_idx']
        whiff_df['duration_sec'] = whiff_df['length_idx'] * dt
    
        # Define ftle start as whiff start - duration of whiff indices
        whiff_df['flowcue_start'] = whiff_df['start_idx'] - (whiff_df['length_idx']).astype(int)
        # Only keep rows from dataframe if ftle_start exists in data range
        whiff_df = whiff_df[whiff_df['flowcue_start'] >= 0]

        # Define ftle end as whiff end + duration of whiff indices
        whiff_df['flowcue_end'] = whiff_df['end_idx'] + (whiff_df['length_idx']).astype(int)
        # Only keep rows from dataframe if ftle_end exists in data range
        whiff_df = whiff_df[whiff_df['flowcue_end'] <= (tmax-tmin)]

        whiff_df = whiff_df.astype({'flowcue_start': int, 'flowcue_end': int})

        # Skip this location if whiff_df is empty:
        if whiff_df.empty:
            continue

        # Compute and store expected value (average) for before, during, and after whiff, along with ratio of (max expected outside)/(expected during)
        whiff_df = whiff_df.apply(compute_expected_vals, axis=1)
        
        # Compute conditional average relative to whiff duration

        # Plot conditional average relative to whiff duration
        # plt.close()
        # boxfig = plt.figure()
        # boxplot = whiff_df.boxplot(column=['E_F_prior', 'E_F_during', 'E_F_after'])
        # plt.plot(1, whiff_df['E_F_prior'].mean(), 'ro')
        # plt.plot(2, whiff_df['E_F_during'].mean(), 'ro')
        # plt.plot(3, whiff_df['E_F_after'].mean(), 'ro')
        # nwhiff = len(whiff_df['E_F_during'])
        # plt.title(f'x={round((xmin+x)*dx, 3)}, y={round(0.3-(ymin+y)*dx, 3)}, whiff={whiff_threshold}, n_whiffs={nwhiff}')
        # boxfig.savefig(f'ignore/plots/FTLE_condAvg_x{round(xmin*dx, 3)}to{round(xmax*dx, 3)}_y{round(ymin*dx-0.3, 3)}to{round(ymax*dx-0.3, 3)}_t{round(tmin*dt, 0)}to{round(tmax*dt, 0)}_wthr{whiff_threshold}.png', dpi=300)
        # plt.show()

        # Use dictionary to track statistics for this location
        stats_dict = {}
        stats_dict.update({'x':x, 'y':y, 'mean_E_ratio': np.mean(whiff_df['E_F_ratio']), 'median_E_F_ratio':np.median(whiff_df['E_F_ratio']), 'sigma_E_F_ratio':np.std(whiff_df['E_F_ratio'])})
        # stats_dict.update({'x':x, 'y':y, 'mean_E_ratio': np.mean(whiff_df['E_F_ratio']), 'median_E_F_ratio':np.median(whiff_df['E_F_ratio'])})
        stats_list.append(stats_dict)

    stats_df = pd.DataFrame(stats_list)
    stats_df.to_pickle(f'ignore/data/meanEratio_x{round(xmin*dx, 3)}to{round(xmax*dx, 3)}_y{round(ymin*dx-0.3, 3)}to{round(ymax*dx-0.3, 3)}_t{round(tmin*dt, 0)}to{round(tmax*dt, 0)}_wthr{whiff_threshold}.pkl')

    # stats_df = pd.read_pickle('ignore/data/meanEratio_x0.398to0.403_y-0.25to0.25_t1.0to180.0_wthr0.001.pkl')

    # PLOTTING
    plt.close()
    vmin = 1
    vmax = 1.1
    plt.scatter(stats_df['x'], stats_df['y'], c=stats_df['mean_E_ratio'], vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(f'ignore/plots/FTLE_meanEratio_x{round(xmin*dx, 3)}to{round(xmax*dx, 3)}_y{round(ymin*dx-0.3, 3)}to{round(ymax*dx-0.3, 3)}_t{round(tmin*dt, 0)}to{round(tmax*dt, 0)}_wthr{whiff_threshold}.png', dpi=300)
    plt.show()

if __name__=='__main__':
    main()
