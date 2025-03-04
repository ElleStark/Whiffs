"""
Class: DataField
For use in computing relative timing of flow and odor cue ridges
"""
import cmasher as cmr
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import scipy.signal as signal
from scipy import stats

logger = logging.getLogger('GradientTiming')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
logger.addHandler(handler)
INFO = logger.info
WARN = logger.warning
DEBUG = logger.debug


class DataField:
    def __init__(self, odor_array, odor_xmesh, odor_ymesh, odor_dt, odor_dx, flow_array, flow_xmesh, flow_ymesh, flow_dt, flow_dx):
        """_summary_

        Args:
            odor_array (_type_): _description_
            odor_xmesh (_type_): _description_
            odor_ymesh (_type_): _description_
            odor_dt (_type_): _description_
            odor_dx (_type_): _description_
            flow_array (_type_): _description_
            flow_xmesh (_type_): _description_
            flow_ymesh (_type_): _description_
            flow_dt (_type_): _description_
            flow_dx (_type_): _description_
        """
        self.odata = odor_array
        self.ox = odor_xmesh
        self.oy = odor_ymesh
        self.odt = odor_dt
        self.odx = odor_dx
        self.fdata = flow_array
        self.fx = flow_xmesh
        self.fy = flow_ymesh
        self.fdt = flow_dt
        self.fdx = flow_dx

        # Initialize dicts for tracking relevant statistics at each location
        self.flow_peaks = {}  # indices of max peak flow cue within all windows at each location
        self.f_o_corrs = {}  # distribution of correlation values at each location
        self.timing_centers = {}  # central tendency (mean or mode) of timing distribution at each location
        self.mean_corrs = {}  # mean of pearson correlation values across all time at each location
        self.std_corrs = {}  # standard deviation of pearson correlations at each location


    def plot_time_series(self, time_vec, o_xidx, o_yidx, o_xloc, o_yloc, save=False, ridges=None, ridges2=None):
        """_summary_

        Args:
            time_vec (1d array): vector of time values for the time to plot
            xidx (int): x index within the existing data arrays to use for selecting point to plot
            yidx (int): y index within the existing data arrays to use for selecting point to plot
            xloc (int): x index relative to the entire domain, to identify location in title and file name
            yloc (int): y index relative to the entire domain, to identify location in title and file name
            ridges (1d array): array of times for plotting vertical lines at ridge locations
        """

        # QC plot: time series of flow & odor data at select points
        DEBUG(f'odor data dims: {self.odata.shape}')
        testpt_x = o_xidx
        testpt_y = o_yidx
        flow_ts = self.fdata[:, testpt_y*int(self.odx/self.fdx), testpt_x*int(self.odx/self.fdx)]
        flow_ts = (flow_ts - np.min(flow_ts)) / (np.max(flow_ts)-np.min(flow_ts))
        odor_ts = self.odata[testpt_y, testpt_x, :]
        odor_ts = (odor_ts - np.min(odor_ts)) / (np.max(odor_ts) - np.min(odor_ts))
        
        # Plot each time series
        fig, ax = plt.subplots(figsize=(10, 4))
        plt.plot(time_vec, flow_ts, label='FTLE')
        plt.plot(time_vec, odor_ts, label='odor gradient')

        # Plot vertical lines at ridge locations, if any
        if ridges is not None:
            ridge_times = time_vec[0] + ridges*self.odt
            plt.vlines(ridge_times, ymin=0, ymax=1, color='red', linestyle='dashed')

        if ridges2 is not None:
            ridge2_times = time_vec[0] + ridges*self.odt
            plt.vlines(ridge2_times, ymin=0, ymax=1, color='blue', linestyle='dashed')

        # Plotting parameters
        plt.title(f'FTLE and odor gradients, [x, y] = [{round((o_xloc+1)*self.odx, 2)}, {round(max(self.oy[:, 0])-(o_yloc+1)*self.odx, 2)}]\
                  t={round(np.min(time_vec), 1)} to {round(np.max(time_vec), 1)} s')
        plt.xlabel('time (s)')
        plt.ylabel('normalized value')
        plt.legend()
        if save:
            plt.savefig(f'ignore/plots/c_grad_ts/FTLEodorgrad_x{round((o_xloc+1)*self.odx, 2)}_y{round(max(self.oy[:, 0])-(o_yloc+1)*self.odx, 2)}]_\
                        t{round(np.min(time_vec), 1)}to{round(np.max(time_vec), 1)}s.png', dpi=300)
        plt.show()


    def find_odor_ridges(self, o_thrs, pt, timelims=slice(None, None), distance=None, width=None):
        """_summary_

        Args:
            o_thrs (_type_): _description_
            yidx (_type_): _description_
            xidx (_type_): _description_
            timelims (_type_, optional): _description_. Defaults to slice(None, None).
            distance (_type_, optional): _description_. Defaults to None.
            width (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        yidx = pt[1]
        xidx = pt[0]

        # select time series of odor data for finding ridge indexes
        odor_ts = self.odata[yidx, xidx, timelims]

        # identify peaks using scipy signal processing package
        ridge_idxs = signal.find_peaks(odor_ts, height=o_thrs, distance=distance, width=width)

        # return list of indexes with odor ridge times
        return ridge_idxs
    

    def find_loc_max_fcue(self, pt, w_ctrs, w_dur_idx, title_id, file_id, ftle=True, timelims=slice(None, None), corr=False, yidx=None, xidx=None, hist=False, box=False, QC=False):
        
        # Extract x, y index locations and convert to FTLE indexes
        yidx = pt[1]
        xidx = pt[0]

        if ftle:
            f_yidx = pt[1]*2
            f_xidx = pt[0]*2
        else:
            f_yidx = pt[1]
            f_xidx = pt[0]

        # select time series of flow cue data for finding max local peaks
        flow_ts = self.fdata[timelims, f_yidx, f_xidx]
        flow_peaks = np.empty_like(w_ctrs)
        flow_peaks[:] = -999

        if corr:
            odor_ts = self.odata[yidx, xidx, timelims]
            corrs = np.empty(len(w_ctrs))
            corrs[:] = -999
        else:
            corrs = None

        i=0
        for ctr in w_ctrs:
            f_ts = flow_ts[ctr-int(w_dur_idx/2):ctr+int(w_dur_idx/2)+1]
            peak = signal.find_peaks(f_ts, distance=w_dur_idx)[0]
            if len(peak) == 0:
                flow_peaks[i] = -999
            elif len(peak) == 1:
                flow_peaks[i] = peak - int(w_dur_idx/2)
            else:
                WARN('Too many peaks found, check window duration')
            
            if QC:
                # QC: line plot of flow time series for each window with peak overlaid
                plt.plot(f_ts)
                plt.vlines(peak, 0, 1)
                plt.show()

            if (corr & len(peak)) == 1:
                o_ts = odor_ts[ctr-int(w_dur_idx/2):ctr+int(w_dur_idx/2)+1]
                o_ts = (o_ts-min(o_ts)) / (max(o_ts)-min(o_ts))
                w_corr = np.corrcoef(o_ts, f_ts)[0, 1]
                corrs[i] = w_corr

            if QC:
                # QC: plot of odor and flow in window
                plt.plot(o_ts, label='odor gradient')
                plt.plot(f_ts, label='flow')
                plt.legend()
                plt.show()

            i+=1

        self.flow_peaks[pt] = flow_peaks[flow_peaks>-100]
        self.f_o_corrs[pt] = corrs[corrs>-100]
        # prominences = signal.peak_prominences(flow_ts, flow_peaks, wlen=w_dur_idx)

        if hist:
            self.plot_peak_hist(pt, title_id, file_id)
        
        if box:
            self.plot_corr_boxplot(pt, title_id, file_id)


    def plot_peak_hist(self, pt, title_id, file_id, bins=20):
        # Histogram of relative flow peak timing
        plt.hist(self.flow_peaks[pt], bins=bins)
        plt.title(f'relative flow cue timing, {title_id}')
        plt.savefig(f'ignore/plots/c_grad_ts/FTLEodorgrad_hist_{file_id}.png', dpi=300)
        plt.show()


    def plot_corr_boxplot(self, pt, title_id, file_id):
        # Boxplot of correlation values w/ mean & std dev labeled
        mean_corr = np.mean(self.f_o_corrs[pt])
        std_corr = np.std(self.f_o_corrs[pt])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.boxplot(self.f_o_corrs[pt])
        plt.title(f'pearson correlations, avg {round(mean_corr, 2)}, std {round(std_corr, 2)}, {title_id}')
        plt.savefig(f'ignore/plots/c_grad_ts/FTLEodorgrad_corrs_{file_id}.png', dpi=300)
        plt.show()


    def compute_timing_centers(self, type='mean'):

        for pt, peak_times in self.flow_peaks.items():
            # each 'peak_times' is a list of timing of flow cue peaks for a given (x, y) location, 'pt'
            if type=='mean':
                self.timing_centers[pt] = round(np.mean(np.abs(peak_times))*self.fdt, 2)
            elif type=='mode':
                self.timing_centers[pt] = round(stats.mode(np.abs(peak_times))[0]*self.fdt, 2)
            else:
                raise SystemExit('Invalid central tendency type. Options are \'mean\' or \'mode\'.')
            
            DEBUG(f'point {pt} {type} of flow cue peak timing: {self.timing_centers[pt]}')

    
    def plot_timing_ctrs_heatmap(self, w_idx_dur, absv=True, title_id=''):

        # initialize figure
        fig, ax = plt.subplots(figsize=(4, 6))

        x_idxs = [x for x, y in self.timing_centers.keys()]
        y_idxs = [y for x, y in self.timing_centers.keys()]
        x_vals = self.ox[y_idxs, x_idxs]
        y_vals = self.oy[y_idxs, x_idxs]

        t_ctrs = [ctr for ctr in self.timing_centers.values()]

        # index of half window duration for plotting colormaps
        half_dur = np.ceil(w_idx_dur / 2)

        # if using absolute value, create sequential colormap 0 to half duration
        if absv:
            # t_ctrs = np.abs(t_ctrs)
            cmap = cmr.lavender_r
            vmin = 0
            vmax = (vmin+half_dur)*self.odt
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

        # if using positive and negative values, create diverging colormap with 0 at center
        else:
            center = 0
            norm = colors.TwoSlopeNorm(vcenter=center, vmin=(center-half_dur)*self.odt, vmax=(center+half_dur)*self.odt)
            cmap = cmr.iceburn

        plt.scatter(x_vals, y_vals, c=t_ctrs, norm=norm, cmap=cmap)
        plt.colorbar()
        plt.title(f'flow peak distance from odor gradient ridge, \n{title_id}')

        plt.show()

    
    def compute_correlation_stats(self):
        for pt, corr_list in self.f_o_corrs.items():
            self.mean_corrs[pt] = round(np.mean(corr_list), 2)
            self.std_corrs[pt] = round(np.std(corr_list), 2)


    def plot_correlation_heatmap(self, title_id=''):
        # initialize figure
        fig, ax = plt.subplots(figsize=(4, 6))

        x_idxs = [x for x, y in self.f_o_corrs.keys()]
        y_idxs = [y for x, y in self.f_o_corrs.keys()]
        x_vals = self.ox[y_idxs, x_idxs]
        y_vals = self.oy[y_idxs, x_idxs]

        # create diverging colormap with 0 at center
        norm = colors.TwoSlopeNorm(vcenter=0, vmin=-0.5, vmax=0.5)
        cmap = cmr.guppy_r

        # extract correlation values
        corr_vals = [corr for corr in self.mean_corrs.values()]

        plt.scatter(x_vals, y_vals, c=corr_vals, norm=norm, cmap=cmap)
        plt.colorbar()
        plt.title(f'pearson correlations, {title_id}')

        plt.show()

