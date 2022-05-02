
import numpy as np 
from matplotlib import pyplot as plt
import astropy.stats.bayesian_blocks as bblocks
#https://docs.astropy.org/en/stable/api/astropy.stats.bayesian_blocks.html
from lightcurves.HopFinder import *

import logging
logging.basicConfig(level=logging.ERROR)
"""
set logging to the desired level
logging options:
DEBUG:      whatever happens will be thrown at you
INFO:       confirmation that things are working as expected
WARNING:    sth unexpected happened
ERROR:      sth didn't work, abort mission
""" 


def flux_puffer(flux, threshold, threshold_error):
    """
    ATTENTION! This returns artificial flux values! Use cautiously if at all..
    Set every flux bin under threshold to threshold before initializing light curve
    Apply Bayesian blocks -> detect significant variations wrt threshold = flares?
    """
    flux_new = np.where(flux > threshold, flux, threshold)
    flux_error_new = np.where(flux > threshold, flux_error, th_error)
    return(flux_new, flux_error_new)

def fix_data(time, flux, flux_error):
    """
    ATTENTION! this deletes bins, if there is np.nan in flux(_error) or duplicate in time
    """
    flux_ = flux[np.invert(np.isnan(flux)) * np.invert(np.isnan(flux_error))]
    flux_error_ = flux_error[np.invert(np.isnan(flux)) * np.invert(np.isnan(flux_error))]
    time_ = time[np.invert(np.isnan(flux)) * np.invert(np.isnan(flux_error))]
    logging.info('Deleted ' + str(len(flux) - len(flux_)) + ' np.nan values.')
    unique_time, unique_time_id = np.unique(time_, return_index=True)
    good_flux = flux_[unique_time_id]
    good_flux_error = flux_error_[unique_time_id]
    logging.info('Deleted ' + str(len(time_) - len(unique_time)) + ' time duplicates')
    return(unique_time, good_flux, good_flux_error)


def get_gti_iis(time, n_gaps, n_pick):
    # get index of good time intervals (divide LC into secitons in case there are n_gaps gaps in data; like FACT)
    # biggest time gaps (length in s) in chronological order
    diff = np.array([t - s for s, t in zip(time, time[1:])])
    diff1 = np.sort(diff)
    ii = [x for x in range(len(diff)) if diff[x] in diff1[-n_gaps:]] #index of the 10 longest gaps
    GTI_start_ii = np.array(ii)+1
    GTI_start_ii = np.insert(GTI_start_ii,0,0)
    GTI_end_ii = np.array(ii)
    GTI_end_ii = np.append(GTI_end_ii, len(time)-1)
    if n_pick:
        # only consider the n_pick longest gtis 
        gap_len = np.array([t - s for s,t in zip(GTI_start_ii, GTI_end_ii)])
        gap_len1 = np.sort(gap_len)
        ii = [x for x in range(len(gap_len)) if gap_len[x] in gap_len1[-n_pick:]] # n_gaps = considered gaps (longest not gaps)
        GTI_start_ii_ = GTI_start_ii[ii]
        GTI_end_ii_ = GTI_end_ii[ii]
        return GTI_start_ii_, GTI_end_ii_
    else:
        return GTI_start_ii, GTI_end_ii

def make_gti_lcs(lc, n_gaps, n_pick=None):
    """
    Divide one lc with n_gaps gaps into several lcs with good coverage.
    """
    gti_starts, gti_ends = get_gti_iis(lc.time, n_gaps, n_pick)
    if n_pick is None:
        n_pick = n_gaps + 1 #select all 
    chunks = []
    for g in range(n_pick):
        gti_lc = LightCurve(lc.time[gti_starts[g]:gti_ends[g]+1], 
                            lc.flux[gti_starts[g]:gti_ends[g]+1], 
                            lc.flux_error[gti_starts[g]:gti_ends[g]+1],
                            name=lc.name, z=lc.z)
        chunks.append(gti_lc)
    return(np.array(chunks))


#--------------------------------------------------------------------------------------------------
class LightCurve:
    """
    Light Curve Class
    ------------------
    Create a light curve based on input data: time, flux, flux_error
    """
    def __init__(self, time, flux, flux_error, name=None, z=None, telescope=None):
        self.time = np.array(time)
        self.flux = np.array(flux)
        self.flux_error = np.array(flux_error)
        self.name = name
        self.z = z
        self.telescope = telescope
        if len(time) != len(flux) or len(time) != len(flux_error):
            raise ValueError('Input arrays do not have same length')
        if len(flux[np.isnan(flux)]) > 0 or len(flux_error[np.isnan(flux_error)]) > 0:
            raise TypeError('flux or flux_error contain np.nan values')
        if len(time) != len(np.unique(time)):
            raise ValueError('time contains duplicate values')

    def plot_lc(self, data_color='k', ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.errorbar(x=self.time, y=self.flux, yerr=self.flux_error, ecolor=data_color, 
                     elinewidth=1, linewidth=0, marker='+', markersize=3, 
                     color=data_color, **kwargs)

    def plot_hline(self, value, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.hlines(value, xmin=min(self.time), xmax=max(self.time), **kwargs)

    #----------------------------------------------------------------------------------------------
    def get_bblocks(self, gamma_value=None, p0_value=0.05): 
        """
        Bayesian block algorithm (https://ui.adsabs.harvard.edu/abs/2013arXiv1304.2818S/abstract)
        fitness is set to 'measures' since we assume Gaussian error for flux measurements
        from astropy (https://docs.astropy.org/en/stable/api/astropy.stats.bayesian_blocks.html)
        Returns edges of blocks (significant changes of flux) in units of time (e.g. MJD)
        Edges are converted to edge_index (based on position in time array)
        -> See GitHub description and Jupyter Notebook for more information
        block_val are the flux values of all blocks based on the mean of all flux bins within
        block_val_error is the corresponding error computed with Gaussian error propagation
        block_pbin has the same shape as flux and is filled with corrsponding block values
        """
        # get Bayesian block edges for light curve
        self.edges = bblocks(t=self.time, x=self.flux, sigma=self.flux_error, fitness='measures',
                             gamma=gamma_value, p0=p0_value)
        logging.debug('got edges for light curve')

        if len(self.edges) <= 2:
            logging.warning('light curve is constant; only one bayesian block found.')
            self.block_pbin = np.ones(len(self.flux)) * np.mean(self.flux)
            self.block_val = np.array([np.mean(self.flux)])
            self.block_val_error = np.array([np.std(self.flux)])
            self.edge_index = np.array([0, -1])
            self.edges = np.array([self.time[0], self.time[-1]])
            return(self.block_pbin, self.block_val, self.block_val_error, self.edge_index,
                   self.edges)
        # get edge_index
        self.edge_index = np.array([np.where(self.time >= self.edges[i])[0][0] 
                                    for i,_ in enumerate(self.edges)])
        #change last entry such that loop over [j:j+1] gives all blocks
        self.edge_index[-1] += 1

        # determine flux value (mean) and error (Gaussian propagation) for each block
        self.block_val = np.zeros(len(self.edge_index)-1)  
        self.block_val_error = np.zeros(len(self.edge_index)-1) 
        for j in range(len(self.edge_index)-1):
            self.block_val[j] = np.mean(self.flux[self.edge_index[j]: self.edge_index[j+1]])
            self.block_val_error[j] = (np.sqrt(np.sum(self.flux_error[self.edge_index[j]:
                                                     self.edge_index[j+1]]**2))
                                      / (self.edge_index[j+1]-self.edge_index[j]))

        # create block-per-bin array corresponding to flux
        self.block_pbin = np.zeros(len(self.flux))
        for k,_ in enumerate(self.block_val):
            self.block_pbin[self.edge_index[k] : self.edge_index[k+1]] = self.block_val[k]
        logging.debug('got block parameters for light curve')

        return(self.block_pbin, self.block_val, self.block_val_error, self.edge_index, self.edges)

    #----------------------------------------------------------------------------------------------
    def get_bblocks_above(self, threshold, pass_gamma_value=None, pass_p0_value=None):
        """
        ATTENTION! This returns artificial flux values! Use cautiously if at all..
        Note: get_bblocks has to be applied first
        Determine Bayesian blocks for light curve but set all blocks that are lower than threshold
        to that threshold (i.e. set small block_val to threshold and neglect edges under threshold)
        -> leaves only significant variations wrt threshold = flares?
        """ 
        self.block_pbin = np.where(self.block_pbin > threshold, self.block_pbin, threshold)
        self.block_val = np.where(self.block_val > threshold, self.block_val, threshold)
        block_mask = np.ones(len(self.block_val), dtype = bool)
        edge_mask = np.ones(len(self.edges), dtype=bool)
        for i in range(len(self.block_val)-1):
            if self.block_val[i] == threshold and self.block_val[i+1] == threshold:
                block_mask[i+1] = False
                edge_mask[i+1] = False
        self.block_val = self.block_val[block_mask]
        self.block_val_error = self.block_val_error[block_mask]
        self.edge_index = self.edge_index[edge_mask]
        self.edges = self.edges[edge_mask]
        return(self.block_pbin, self.block_val, self.block_val_error, self.edge_index, self.edges)

    #----------------------------------------------------------------------------------------------
    def plot_bblocks(self, bb_color='steelblue', data_color='k', data_label='obs flux',
                     size=1, ax=None):
        if ax is None:
                ax = plt.gca()
        try:   
            ax.step(self.time, self.block_pbin, where='mid', linewidth=1*size, label='bblocks', 
            	     color=bb_color, zorder=1000)
            ax.errorbar(x=self.time, y=self.flux, yerr=self.flux_error, label=data_label, 
            	         ecolor=data_color, elinewidth=1*size, linewidth=0, marker='+', 
                         markersize=3*size, color=data_color)
        except AttributeError:
            raise AttributeError('Initialize Bayesian blocks with .get_bblocks() first!')

    #----------------------------------------------------------------------------------------------
    def bb_i(self, t):
        """
        Convert time to index of corresponding Bayesian block (e.g. block_value of peak_time)
        use bb_i_start/bb_i_end to make sure you get the block left/right outside of hop
        this works fine for flip, halfclap, and sharp but *NOT for BASELINE* (-> block inside hop)
        """
        if t == self.edges[0]:
            return(int(0))
        else:
            block_index = [
                e for e in range(len(self.edges)-1) if t > self.edges[e] and t <= self.edges[e+1]]
            return(int(block_index[0]))

    def bb_i_start(self,t):
        """
        if time = edge -> take block on the left
        ATTENTION: for baseline method this is first block of hop -> use bb_i() instead (works)
        """
        block_index = [
            e for e in range(len(self.edges)-1) if t >= self.edges[e] and t < self.edges[e+1]]
        return(int(block_index[0]))

    def bb_i_end(self,t):
        """
        if time = edge -> take block on the right
        ATTENTION: for baseline method this is last block of hop - use bb_i() instead (TBD)
        """
        block_index = [
            e for e in range(len(self.edges)-1) if t > self.edges[e] and t <= self.edges[e+1]]
        return(int(block_index[0]))
    
    #----------------------------------------------------------------------------------------------
    def find_hop(self, method='half', lc_edges='neglect', baseline=None):
        if method == 'baseline':
            if baseline is None:
                self.baseline = np.mean(self.flux)
            else:
                self.baseline = baseline
            hopfinder = HopFinderBaseline(lc_edges)
        if method == 'half':
            hopfinder = HopFinderHalf(lc_edges)
        if method == 'sharp':
            hopfinder = HopFinderSharp(lc_edges)
        if method == 'flip':
            hopfinder = HopFinderFlip(lc_edges)
        self.hops = hopfinder.find(self)
        return self.hops

    #----------------------------------------------------------------------------------------------
    def plot_hop(self, ax=None, **kwargs):
        """
        Plot shaded area for all hops in light curve
        """
        if self.hops is None:
            return # no hop in this lc
        if ax is None:
            ax = plt.gca()
        for i,hop in enumerate(self.hops):
            x = np.linspace(hop.start_time, hop.end_time)
            y = np.ones(len(x)) * np.max(self.flux)
            y1 = np.min(self.flux)
            if i == 0:
                ax.fill_between(x, y, y1, step="mid", color='lightsalmon', alpha=0.2,
                                 label='hop', zorder=0)
            if i == 1:
                ax.fill_between(x, y, y1, step="mid", color='orchid', alpha=0.2, label='hop',
                                 zorder=0)
            elif i % 2:
                ax.fill_between(x, y, y1, step="mid", color='orchid', alpha=0.2, zorder=0)
            elif i != 0:
                ax.fill_between(x, y, y1, step="mid", color='lightsalmon', alpha=0.2, zorder=0)
        #ax.set_title(lc.name, hop.method)

    #----------------------------------------------------------------------------------------------
    def plot_all_hop(self, gamma_value=None, p0_value=0.05, lc_edges='neglect'):
        """
        Plot all HOP methods in one figure for comparison
        """
        fig = plt.figure(0,(15,9))
        plt.suptitle('All HOP methods', fontsize=16)

        ax0 = fig.add_subplot(511)
        self.find_hop('baseline')
        self.plot_bblocks()
        self.plot_hop()
        plt.ylabel('baseline')

        ax1 = fig.add_subplot(512)
        self.find_hop('half')
        self.plot_bblocks()
        self.plot_hop()
        plt.ylabel('half')

        ax2 = fig.add_subplot(513)
        self.find_hop('flip')
        self.plot_bblocks()
        self.plot_hop()
        plt.ylabel('flip')

        ax3 = fig.add_subplot(514)
        self.find_hop('sharp')
        self.plot_bblocks()
        self.plot_hop()
        plt.ylabel('sharp')
        fig.subplots_adjust(hspace=0)


