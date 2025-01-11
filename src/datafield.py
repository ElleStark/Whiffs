"""
Class: DataField
For use in computing relative timing of flow and odor cue ridges
"""

import numpy as np
import matplotlib.pyplot as plt

class DataField:
    def __init__(self, dataarray, xmesh, ymesh, dt):
        self.data = dataarray
        self.x = xmesh
        self.y = ymesh
        self.dt = dt

        self.xy_gradient = np.empty_like(dataarray)

