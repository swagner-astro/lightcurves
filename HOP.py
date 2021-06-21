'''
HOP Class - Sarah Wagner
========================

Takes start, peak and end time of a flare (flux variation) as determined with the HOP algorithm (see LC_new.py)

'''

import numpy as np
from matplotlib import pyplot as plt
from selfmade_py.LC_new import LightCurve

class Hopject:

    def __init__(self, hop_params, lc):
        self.start_time, self.peak_time, self.end_time = hop_params
        self.lc = lc
        self.dur = self.end_time - self.start_time
        self.rise_time = self.peak_time - self.start_time
        self.decay_time = self. end_time - self. peak_time
        self.asym = (self.rise_time - self.decay_time)/(self.rise_time + self.decay_time)

        self.start_flux = lc.block_val[lc.bb_i_start(self.start_time)]
        self.peak_flux = lc.block_val[lc.bb_i(self.peak_time)]
        self.end_flux = lc.block_val[lc.bb_i_end(self.end_time)]
        self.rise_flux = self.peak_flux - self.start_flux
        self.decay_flux = self.peak_flux - self.end_flux
        self.n_blocks = lc.bb_i_end(self.end_time) - lc.bb_i_start(self.start_time)
        # eg one-block hop: 5 - 3 = 2 
        if lc.z:
            self.z = lc.z
        else:
            self.z = np.nan

        # how many bins; coverage = bins/time e.g. = 1 for daily binning
        # which bins -> mask to then fit and so on

# exponential flare function se my_forschung/coding

    def plot_hop(self):
        self.lc.plot_bblocks()
        plt.xlim(self.start_time - self.dur/2, self.end_time + self.dur/2)
        x = np.linspace(self.start_time, self.end_time)
        y = np.ones(len(x)) * self.peak_flux
        y1 = np.zeros(len(x))
        plt.fill_between(x, y, y1, step="mid", color='lightsalmon', alpha=0.2, label='hop', zorder=0)





