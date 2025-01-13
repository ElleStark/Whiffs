"""
Class: DataField
For use in computing relative timing of flow and odor cue ridges
"""
import logging
import numpy as np
import matplotlib.pyplot as plt

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
        

    def plot_time_series(self, time_vec, o_xidx, o_yidx, o_xloc, o_yloc):
        """_summary_

        Args:
            time_vec (1d array): vector of time values for the time to plot
            xidx (int): x index within the existing data arrays to use for selecting point to plot
            yidx (int): y index within the existing data arrays to use for selecting point to plot
            xloc (int): x index relative to the entire domain, to identify location in title and file name
            yloc (int): y index relative to the entire domain, to identify location in title and file name
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

        # Plotting parameters
        plt.title(f'FTLE and odor gradients, [x, y] = [{round((o_xloc+1)*self.odx, 2)}, {round(max(self.oy[:, 0])-(o_yloc+1)*self.odx, 2)}], t={round(np.min(time_vec)*self.odt, 1)} to {round(np.max(time_vec)*self.odt, 1)} s')
        plt.xlabel('time (s)')
        plt.ylabel('normalized value')
        plt.legend()
        plt.savefig(f'ignore/plots/c_grad_ts/FTLEodorgrad_x{round((o_xloc+1)*self.odx, 2)}_y{round(max(self.oy[:, 0])-(o_yloc+1)*self.odx, 2)}]_t{round(np.min(time_vec)*self.odt, 1)}to{round(np.max(time_vec)*self.odt, 1)}s.png', dpi=300)
        plt.show()
