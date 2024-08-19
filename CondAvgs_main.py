# Main script to compute relative timing of flow and odor cue signals in a chaotic flow field
# Uses conditional averaging of whiff-based time windows
# Elle Stark, CU Boulder Environmental Fluid Mechanics Lab, Aug 2024

import h5py
import numpy as np
import pandas as pd

def main():
# Subset data by index as needed
    xrange = slice(None, None)
    yrange = slice(None, None)
    time_lims = slice(None, None)

    # Obtain datasets from h5 file
    filename = 'D:/singlesource_2d_extended/Re100_0_5mm_50Hz_singlesource_2d.h5'
    with h5py.File(filename, 'r') as f:
        # x and y grids for plotting
        x_grid = f.get(f'Model Metadata/xGrid')[xrange, yrange].T
        y_grid = f.get(f'Model Metadata/yGrid')[xrange, yrange].T

        # Odor and flow cue data
        odor_data = f.get(f'Odor Data/c')[time_lims, xrange, yrange].transpose(0, 2, 1)
        odor_max = f.get(f'Odor Data/odorSourceConcentration')[0].item()
        u_data = f.get(f'Flow Data/u')[time_lims, xrange, yrange].transpose(0, 2, 1)

        # Spatial resolution & time array
        dt_freq = f.get('Model Metadata/timeResolution')[0].item()
        dt = 1 / dt_freq  # convert from Hz to seconds
        dx = f.get('Model Metadata/spatialResolution')[0].item()
        time_array = f.get('Model Metadata/timeArray')[time_lims]

    # Flip velocity data upside-down for correct computation of gradients
    u_data = np.flip(u_data, axis=2)
    dudt = np.gradient(u_data, dt, axis=0)
    flow_data = dudt 

    # Define concentration level for start & end of whiff
    whiff_threshold = 0.05

    # CHOOSING X AND Y BASED ON CONTINUOUS INDICES
    x = list(range(595, 605, 1))
    y = list(range(500, 1000, 1))
    locs_list = list(zip(x, y)) 

    # Initialize dataframe for storing whiff information
    all_whiffs_df = pd.DataFrame(columns=['x_idx', 'y_idx', 'start_idx', 'end_idx', 'length_idx', 'duration_sec', 'ftle_start',
                                        'ftle_end', 'E_F_prior', 'E_F_during', 'E_F_after', 'E_F_ratio'])

    counter = 0

    # Loop through each location to compute conditional average ratios
    for x, y in locs_list:
        # Obtain time series of data for flow and odor cues
        odor_ts = pd.Series(odor_data[:, x, y])
        flow_ts = pd.Series(flow_data[:, x, y])

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
        whiff_df['ftle_start'] = whiff_df['start_idx'] - (whiff_df['length_idx']).astype(int)
        # Only keep rows from dataframe if ftle_start exists in data range
        whiff_df = whiff_df[whiff_df['ftle_start'] >= 0]

        # Define ftle end as whiff end + duration of whiff indices
        whiff_df['ftle_end'] = whiff_df['end_idx'] + 2 * (whiff_df['length_idx']).astype(int)
        # Only keep rows from dataframe if ftle_start exists in data range
        whiff_df = whiff_df[whiff_df['ftle_end'] <= (time_lims[1] - time_lims[0])]

        whiff_df = whiff_df.astype({'ftle_start': int, 'ftle_end': int})

        # Skip this location if whiff_df is empty:
        if whiff_df.empty:
            continue



if __name__=='__main__':
    main()
