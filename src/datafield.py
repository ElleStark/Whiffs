"""
Class: DataField
For use in computing relative timing of flow and odor cue ridges
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

logger = logging.getLogger('GradientTiming')
logger.setLevel(logging.DEBUG)
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
        odor_ts = self.odata[:, testpt_y, testpt_x]
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

    def find_odor_ridges(self, o_thrs, yidx, xidx, timelims=slice(None, None), distance=None, width=None):
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
        # select time series of odor data for finding ridge indexes
        odor_ts = self.odata[timelims, yidx, xidx]

        # identify peaks using scipy signal processing package
        ridge_idxs = signal.find_peaks(odor_ts, height=o_thrs, distance=distance, width=width)

        # return list of indexes with odor ridge times
        return ridge_idxs
    
    def find_loc_max_fcue(self, f_yidx, f_xidx, w_ctrs, w_dur_idx, timelims=slice(None, None), corr=False, yidx=None, xidx=None):

        # select time series of flow cue data for finding max local peaks
        flow_ts = self.fdata[timelims, f_yidx, f_xidx]
        flow_peaks = np.empty_like(w_ctrs)
        flow_peaks[:] = -999

        if corr:
            odor_ts = self.odata[timelims, yidx, xidx]
            corrs = np.empty(len(w_ctrs))
            corrs[:] = -999
        else:
            corrs = None

        i=0
        for ctr in w_ctrs:
            f_ts = flow_ts[ctr-int(w_dur_idx/2):ctr+int(w_dur_idx/2)]
            peak = signal.find_peaks(f_ts, distance=w_dur_idx)[0]
            if len(peak) == 0:
                flow_peaks[i] = -999
            elif len(peak) == 1:
                flow_peaks[i] = peak - int(w_dur_idx/2)
            else:
                INFO('Too many peaks found, check window duration')
            
            # QC: line plot of flow time series for each window with peak overlaid
            # plt.plot(f_ts)
            # plt.vlines(peak, 0, 1)
            # plt.show()

            if (corr & len(peak)) == 1:
                o_ts = odor_ts[ctr-int(w_dur_idx/2):ctr+int(w_dur_idx/2)]
                o_ts = (o_ts-min(o_ts)) / (max(o_ts)-min(o_ts))
                w_corr = np.corrcoef(o_ts, f_ts)[0, 1]
                corrs[i] = w_corr

                # QC: plot of odor and flow in window
                # plt.plot(o_ts, label='odor gradient')
                # plt.plot(f_ts, label='flow')
                # plt.legend()
                # plt.show()

            i+=1

        return flow_peaks, corrs

