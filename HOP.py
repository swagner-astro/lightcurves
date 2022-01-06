import numpy as np
from matplotlib import pyplot as plt
from lightcurves.LC import LightCurve


class Hopject: 
    '''
    HOP Class
    ==========
    Segment in a light curve, i.e. a group of Bayesian blocks, that represent a flare ( HOP group)
    For definition of start, peak and end of the flare check out the LightCurve class
    '''
    def __init__(self, hop_params, lc): #e.g. Hopject(lc.get_hop_method[0], lc)
        self.start_time, self.peak_time, self.end_time = hop_params
        self.lc = lc
        #self.mask = which bins -> mask to have all HOP bins of lc
        #self.n_bins = how many bins
        #self.coverage = bins/time
        self.n_blocks = lc.bb_i_end(self.end_time) - lc.bb_i_start(self.start_time) 
        # e.g. one-block hop: 5 - 3 = 2 
        if lc.z:
            self.z = lc.z
        else:
            self.z = np.nan

        self.dur = self.end_time - self.start_time
        self.rise_time = self.peak_time - self.start_time
        self.decay_time = self. end_time - self. peak_time
        self.asym = (self.rise_time - self.decay_time)/(self.rise_time + self.decay_time)

        self.start_flux = lc.block_val[lc.bb_i_start(self.start_time)]
        self.peak_flux = lc.block_val[lc.bb_i(self.peak_time)]
        self.end_flux = lc.block_val[lc.bb_i_end(self.end_time)]
        #attention: bb_i has bugs for baseline method
        # in case of baseline 
        self.rise_flux = self.peak_flux - self.start_flux
        self.decay_flux = self.peak_flux - self.end_flux

    #----------------------------------------------------------------------------------------------
    def plot_hop(self):
        """
        Plot the snip of light curve with this flare
        """
        self.lc.plot_bblocks()
        plt.xlim(self.start_time - self.dur/2, self.end_time + self.dur/2)
        x = np.linspace(self.start_time, self.end_time)
        y = np.ones(len(x)) * self.peak_flux
        y1 = np.zeros(len(x))
        plt.fill_between(x, y, y1, step="mid", color='lightsalmon', alpha=0.2, label='hop', zorder=0)
