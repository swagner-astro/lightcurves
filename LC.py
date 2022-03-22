
import numpy as np 
from matplotlib import pyplot as plt
import astropy.stats.bayesian_blocks as bblocks
#https://docs.astropy.org/en/stable/api/astropy.stats.bayesian_blocks.html

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

#--------------------------------------------------------------------------------------------------
class LightCurve:
    """
    Light Curve Class
    ------------------
    Create a light curve based on input data: time, flux, flux_error
    Determine Bayesian block representation of light curve.
    Characterize flares (start, peak and end time) based on blocks with four methods:
        1. baseline:
            Original method as described in Meyer et al. 2019
            https://ui.adsabs.harvard.edu/abs/2019ApJ...877...39M/abstract 
        2. half:
            Start/end is at center of valley block
        3. sharp:
            Neglect valley block
        4. flip:
            Extrapolate flare behavior
        -> See GitHub description and Jupyter Notebook for more information
    """
    def __init__(self, time, flux, flux_error, name=None, z=None):
        self.time = np.array(time)
        self.flux = np.array(flux)
        self.flux_error = np.array(flux_error)
        self.name = name
        self.z = z
        if len(time) != len(flux) or len(time) != len(flux_error):
            raise ValueError('Input arrays do not have same length')
        if len(flux[np.isnan(flux)]) > 0 or len(flux_error[np.isnan(flux_error)]) > 0:
            raise TypeError('flux or flux_error contain np.nan values')
        if len(time) != len(np.unique(time)):
            raise ValueError('time contains duplicate values')

    def plot_lc(self, data_color='k', **kwargs):
        plt.errorbar(x=self.time, y=self.flux, yerr=self.flux_error, ecolor=data_color, 
                     elinewidth=1, linewidth=0, marker='+', markersize=3, 
                     color=data_color, **kwargs)

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
        Note: get_bblocks as to be applied first
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
    def plot_bblocks(self, bb_color='steelblue', data_color='k', data_label='obs flux', size=1):
        try:
            plt.step(self.time, self.block_pbin, where='mid', linewidth=1*size, label='bblocks', 
            	     color=bb_color, zorder=1000)
            plt.errorbar(x=self.time, y=self.flux, yerr=self.flux_error, label=data_label, 
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
    def handle_hops(self, peaks, starts, ends, lc_edges):
        """
        Handle mismatches and issues with peak_time, start_time, and end_time combinations
        lc_edges:
            a) neglect:
                single start and end times are neglected
                peaks without start or end time are neglected
            b) add:
                single start and end times are neglected
                peaks without start/end: start/end is added in beginning/end of light curve
        """
        if len(peaks) < 1:
            logging.info('not variable enough, no peak found')
            return(None, None, None)
        if lc_edges == 'neglect':
            if len(starts) < 1 or len(ends) < 1:
                logging.info('not variable enough, missing start or end')
                return(None, None, None)
        if lc_edges == 'add':
            if len(starts) < 1:
                starts = np.insert(starts, 0, self.edges[0])
                logging.info('inserted single start in beginning of LC')
            if len(ends) < 1:
                ends = np.append(ends,self.edges[-1])
                logging.info('inserted single end in end of LC')
        if ends[0] < peaks[0]:
            ends = np.delete(ends, 0)
            logging.info('deleted single end in beginning of LC')
            if len(ends) < 1 and lc_edges == 'neglect':
                logging.info('this was the only end, not variable enough')
                return(None, None, None)
            if len(ends) < 1 and lc_edges == 'add':
                ends = np.append(ends, self.edges[-1])
                logging.info('inserted single end in end of LC and this is the only end')    
        if starts[-1] > peaks[-1]:
            starts = np.delete(starts, -1)
            logging.info('deleted single start in end of LC')
            if len(starts) < 1 and lc_edges == 'neglect':
                logging.info('this was the only start, not variable enough')
                return(None, None, None)
            if len(starts) < 1 and lc_edges == 'add':
                starts = np.insert(starts, 0, self.edges[0])
                logging.info('inserted single start in beginning of LC; this is the only start')
        if peaks[0] < starts[0]:
            if lc_edges == 'add':
                # artificially add start
                starts = np.insert(starts, 0, self.edges[0])
                logging.info('inserted single start in beginning of LC')
            if lc_edges == 'neglect':
                # conservatively dismiss first peak if there are multiple peaks
                while ends[0] > peaks[1]:
                    peaks = np.delete(peaks, 0)
                    logging.info('neglected first multiple peak in beginning of LC')
                #conservatively dismiss first peak and first end
                peaks = np.delete(peaks, 0)
                ends = np.delete(ends, 0)
                logging.info('start missing, neglected peak and end in beginning of LC')
                if len(peaks) < 1 or len(ends) < 1:
                    logging.info('this was the only peak or end, not variable enough')
                    return(None, None, None)
        if peaks[-1] > ends[-1]:
            if lc_edges == 'add':
                # artificially add end
                ends = np.append(ends, self.edges[-1])
                logging.info('inserted single end in end of LC') 
            if lc_edges == 'neglect':
                # conservatively dismiss last peak if there are multiple peaks
                if len(peaks) > 2:
                    while starts[-1] < peaks[-2]:
                        peaks = np.delete(peaks, -1)
                        logging.info('neglected last multiple peak in end of LC')
                    # conservatively dismiss last peak and last start
                peaks = np.delete(peaks, -1)
                starts = np.delete(starts, -1)
                logging.info('neglected peak and start in end of LC')
                if len(peaks) < 1 or len(starts) < 1:
                    logging.info('this was the only peak or start, not variable enough')
                    return(None, None, None)

        return(peaks, starts, ends)
    
    #----------------------------------------------------------------------------------------------
    def get_hop_baseline(self, baseline=None, lc_edges='neglect'):
        """
        BASELINE METHOD
        see Meyer et al. 2019 https://ui.adsabs.harvard.edu/abs/2019ApJ...877...39M/abstract
        Define flare as group of blocks (HOP group) with start, peak, and end time
        Determine peak_time of flare to be at center of colal maxima of the blocks
        Determine start_time/end_time to be where flux exceeds/goes under baseline

        baseline: 
            e.g. mean of flux (default), median of flux, quiescent background ...

        lc_edges:
            a) 'neglect'
                single start and end times are neglected
                incomplete flares (peaks without start or end time) are conservatively neglected
            b) 'add'
                single start and end times are neglected
                if peak has no start/end it is artificially added in beginning/end of light curve

        returns:
            HOP groups, e.g. [[start, peak, end],[start, peak, end]]
            (Note: all starts can be called with lc.start_times_bl, for example)
        """
        if baseline is None:
            baseline = np.mean(self.flux)
            self.baseline = np.mean(self.flux)
            logging.info('use default baseline: mean(flux)')
        else:
            self.baseline = baseline

        diff = np.diff(self.block_val)
        peak_times = [] #time of all local peaks over baseline (in units of edges = units of time)
        for i in range(1,len(diff)):
            # if previous rising; this falling
            if diff[i-1] > 0 and diff[i] < 0:
                if self.block_val[i] > baseline:
                    # peak_time = middle of peak block
                    peak_times.append(self.edges[i] + (self.edges[i+1] - self.edges[i]) /2)
        start_times = []  
        end_times = []    
        for i in range(len(self.block_val)-1):
            # if this smaller; next one higher
            if self.block_val[i] < baseline and self.block_val[i+1] > baseline:
                start_times.append(self.edges[i+1])
            # if this larger; next one lower
            if self.block_val[i] > baseline and self.block_val[i+1] < baseline:
                end_times.append(self.edges[i+1])
        peak_times, start_times, end_times = self.handle_hops(
            np.array(peak_times), np.array(start_times), np.array(end_times), lc_edges) 
        if peak_times is None:
            self.start_times_bl, self.end_times_bl = None, None
            logging.warning('light curve is not variable enough; no hop found.')
            return(None)
        # baseline method could result in multiple peaks within one HOP 
        # -> neglect smaller peak (not so senseful..)
        while len(end_times) < len(peak_times):
            for x,_ in enumerate(end_times):
                if end_times[x] > peak_times[x+1]:
                    if (self.block_val[self.bb_i(peak_times[x])] 
                        < self.block_val[self.bb_i(peak_times[x+1])]):
                            peak_times = np.delete(peak_times, x)
                    elif (self.block_val[self.bb_i(peak_times[x])] 
                    	  >= self.block_val[self.bb_i(peak_times[x+1])]):
                            peak_times = np.delete(peak_times, x+1)
                    logging.info('neglected double peak in HOP ' + str(x))
                    break
        self.peak_times_bl = peak_times
        self.start_times_bl = start_times
        self.end_times_bl = end_times
        return(np.array([start_times, peak_times, end_times]).transpose())

    #----------------------------------------------------------------------------------------------
    def hop_procedure(self, method, lc_edges):
        """
        OTHER METHODS
        Define flare as group of blocks (HOP group) with start, peak, and end time
        Determine peak_time of flare to be at center of colal maxima of the blocks
        Use .get_hop_method() analogous to .get_hop_baseline()
        
        method:
            a) 'half'
                Determine start/end of flare to be at center of valley block
            b) 'flip'
                Extrapolate behavior of flare by flipping adjacent block onto valley block
                Note: half method is used to avoid overlap (i.e. when flip > 1/2 valley block)
            c) 'sharp'
                Neglect valley block

        lc_edges:
            a) 'neglect'
                single start and end times are neglected
                incomplete flares (peaks without start or end time) are conservatively neglected
            b) 'add'
                single start and end times are neglected
                if peak has no start/end it is artificially added in beginning/end of light curve

        returns:
            HOP groups, e.g. [[start, peak, end],[start, peak, end]]
            (Note: all starts can be called with lc.start_times_bl, for example)
        """
        diff = np.diff(self.block_val)
        peak_times = [] # time of all local peaks (units of edges, i.e. units of time)
        start_times = []
        end_times = []
        for i in range(1,len(diff)):
            # peak = previous rising; this falling
            if diff[i-1] > 0 and diff[i] < 0:
                # peak_time = middle of peak block
                peak_times.append(self.edges[i] + (self.edges[i+1] - self.edges[i]) /2)
            # change = previous falling; this rising
            if diff[i-1] < 0 and diff[i] > 0:
                half_block_time = (self.edges[i+1] - self.edges[i]) / 2
                if method == 'half':
                    start_times.append(self.edges[i+1] - half_block_time)
                    end_times.append(self.edges[i] + half_block_time)
                if method == 'flip':
                	#clap previous block onto change block
                    clap_from_left = self.edges[i] - self.edges[i-1]
                    #clap following block onto change block
                    clap_from_right = self.edges[i+2] - self.edges[i+1]
                    end_times.append(self.edges[i] 
                    	             + np.minimum(half_block_time, clap_from_left))
                    start_times.append(self.edges[i+1] 
                    	               - np.minimum(half_block_time, clap_from_right))
                if method == 'sharp':
                    start_times.append(self.edges[i+1])
                    end_times.append(self.edges[i])
        return(self.handle_hops(np.array(peak_times), np.array(start_times),
        	   np.array(end_times), lc_edges))
                
    def get_hop_half(self, lc_edges='neglect'):
        self.peak_times, self.start_times_half, self.end_times_half = self.hop_procedure('half', lc_edges)
        if self.peak_times is None:
            logging.warning('light curve is not variable enough; no hop found.')
            return(None)
        else:
            return((np.array([self.start_times_half, self.peak_times, self.end_times_half])).transpose())
    
    def get_hop_flip(self, lc_edges='neglect'):
        self.peak_times, self.start_times_flip, self.end_times_flip = self.hop_procedure('flip', lc_edges)
        if self.peak_times is None:
            logging.warning('light curve is not variable enough; no hop found.')
            return(None)
        else:
            return((np.array([self.start_times_flip, self.peak_times, self.end_times_flip])).transpose())
    
    def get_hop_sharp(self, lc_edges='neglect'):
        self.peak_times, self.start_times_sharp, self.end_times_sharp = self.hop_procedure('sharp', lc_edges)
        if self.peak_times is None:
            logging.warning('light curve is not variable enough; no hop found.')
            return(None)
        else:
            return((np.array([self.start_times_sharp, self.peak_times, self.end_times_sharp])).transpose())

    #----------------------------------------------------------------------------------------------
    def hop_around(self, gamma_value=None, p0_value=0.05, lc_edges='neglect'):
        """
        Initialize Bayesian blocks and all HOP methods with default settings in one go
        """
        self.get_bblocks(gamma_value, p0_value)
        self.get_hop_baseline(lc_edges=lc_edges) # necessary because of baseline argument
        self.get_hop_half(lc_edges)
        self.get_hop_flip(lc_edges)
        self.get_hop_sharp(lc_edges)
        logging.debug('hoppped around')

    #----------------------------------------------------------------------------------------------
    def plot_hop_by_time(self, start_times, end_times): 
        """
        Plot shaded area for given start and end times
        for example: lc.plot_hop_by_time(lc.start_times_flip, lc.end_times_flip) 
        """
        if start_times is None:
            return() # no hop found
        for i,_ in enumerate(start_times):
            x = np.linspace(start_times[i], end_times[i])
            y = np.ones(len(x)) * np.max(self.flux)
            #y1 = np.zeros(len(x))
            y1 = np.min(self.flux)
            if i == 0:
                plt.fill_between(x, y, y1, step="mid", color='lightsalmon', alpha=0.2,
                	             label='hop', zorder=0)
            if i == 1:
                plt.fill_between(x, y, y1, step="mid", color='orchid', alpha=0.2, label='hop',
                	             zorder=0)
            elif i % 2:
                plt.fill_between(x, y, y1, step="mid", color='orchid', alpha=0.2, zorder=0)
            elif i != 0:
                plt.fill_between(x, y, y1, step="mid", color='lightsalmon', alpha=0.2, zorder=0)

    def plot_hop(self, method = 'flip'):
        """
        Plot shaded area for given HOP method
        """
        if method == 'half':
            self.plot_hop_by_time(self.start_times_half, self.end_times_half)
        if method == 'flip':
            self.plot_hop_by_time(self.start_times_flip, self.end_times_flip)
        if method == 'sharp':
            self.plot_hop_by_time(self.start_times_sharp, self.end_times_sharp)
        if method == 'baseline':
            self.plot_hop_by_time(self.start_times_bl, self.end_times_bl)
            plt.hlines(self.baseline, xmin=np.min(self.time), xmax=np.max(self.time),
            	       color='deeppink', label='baseline', linewidth=1, zorder=100)

    #----------------------------------------------------------------------------------------------
    def plot_all_hop(self, gamma_value=None, p0_value=0.05, lc_edges='neglect'):
        """
        Plot all HOP methods in one figure for comparison
        """
        self.hop_around(gamma_value, p0_value, lc_edges)
        fig = plt.figure(0,(15,7))
        plt.suptitle('All HOP methods', fontsize=16)

        ax0 = fig.add_subplot(511)
        self.plot_bblocks()
        self.plot_hop('baseline')
        plt.ylabel('baseline')
        ax1 = fig.add_subplot(512)
        self.plot_bblocks()
        self.plot_hop('half')
        plt.ylabel('half')
        ax2 = fig.add_subplot(513)
        self.plot_bblocks()
        self.plot_hop('flip')
        plt.ylabel('flip')
        ax3 = fig.add_subplot(514)
        self.plot_bblocks()
        self.plot_hop('sharp')
        plt.ylabel('sharp')
        fig.subplots_adjust(hspace=0)


    #-----------------------------------------------------------------------------------------------
    def get_ou(self, sigma_sigma=0.343, sigma_alpha=1.48):
        """
        Interpret light curve as exponentiated Ornstein-Uhlenbeck process:
         -> https://ui.adsabs.harvard.edu/abs/1930PhRv...36..823U/abstract
        Extract characteristic OU parameters mu, simga, theta according to:
         -> Burd et al. 2020, https://ui.adsabs.harvard.edu/abs/2021A%26A...645A..62B/abstract
         -> and Kohlepp 2021, https://www.physik.uni-wuerzburg.de/fileadmin/11030400/bachelor_thesis_kohlhepp.pdf
        Implementation adapted from:
         -> https://github.com/PRBurd/astro-wue
        
        arguments:
        sigma_sigma = size of epsilon environment to extract sigma and abs(theta)
        sigma_alpha = size of epsilon environment to extract sign of theta
         -> see Kohlepp 2021 for determination of the default parameters
        
        returns:
        mu = mean revision nlevel = expectation value
        sigma = sigma * sqrt(dt) = randomness/innovation
        theta = theta * dt = mean revision rate
         -> see references above to get more information
        """
        
        # Assumption: lc flux is exponentiated time series (we analyze the latter)
        if len(np.where(self.flux < 0)[0]) > 0 or len(self.flux)<4:
            self.ou_mu = None
            self.ou_sigma = None
            self.ou_theta = None
            #raise ValueError('Flux contains negative values, cannot take np.log10()')
            print('Flux contains negative values, cannot take np.log10()')
            return None
        data = np.log10(self.flux) # time series = OU
        # adding a random buffer to make flux positive results in different parameters - bäh!
        
        # mu = mean revision nlevel 
        # -> simply the expectation value = mean of time series
        # -> Note: this is meaningless if buffer != 0
        #self.ou_mu = np.mean(data)
        self.ou_mu = np.mean(data)
        
        # sigma = randomness/innovation 
        # -> consider data points close to mean (within standard deviation * sigma_sigma)
        # -> assumption: difference to next data point is due to randomness of OU process
        # -> variance of these differences resembles ou_sigma
        std = np.std(data)
        close_mask = np.array(data > (self.ou_mu - sigma_sigma * std), dtype=bool) *\
                     np.array(data < (self.ou_mu + sigma_sigma * std), dtype=bool)
        close_mask_diff = np.delete(close_mask, -1) 
        diff = np.diff(data)
        self.ou_sigma = np.std(diff[close_mask_diff]) #= sigma * sqrt(dt)
        
        # theta = mean revision rate = friction coefficient/tensor = 1 - alpha
        # -> value of alpha is computed according to equation 2.14 in Kohlepp 2021
        # -> sign of alpha is computed according to equation 2.15e in Kohlepp 2021
        alpha_value = np.sqrt(np.abs(1 - (self.ou_sigma**2 / np.var(data))))
        pos = np.array(data > (self.ou_mu - sigma_alpha * std), dtype=bool) *\
              np.array(data < (self.ou_mu + sigma_alpha * std), dtype=bool)
        # discards last element (no u_T+1 would exist)
        pos[-1] = False
        # positions of u_T+1 (shift positions by +1)
        pos1 = np.zeros(len(pos), dtype=bool)
        pos1[1:] = pos[:-1]    
        alphas = (data[pos1] - self.ou_mu)/(data[pos] - self.ou_mu)
        alpha_sign = np.sign(np.mean(alphas))
        self.ou_theta = 1 - (alpha_sign * alpha_value)  #= theta * dt
        return(self.ou_mu, self.ou_sigma, alpha_sign * alpha_value, self.ou_theta )



''' 
FUTURE WORK:
    #-----------------------------------------------------------------------------------------------
    def get_psd(self):
    	self.psd_slope = ...
    	return(psd_slope)

    def plot_psd(self):
    	plot se psd
    
    def get_Fvar():
        #TBD: Bernd
        ...
        return(F_var, F_var_error)
    
    def is_constant():
        #TBD: Bernd
        ...
        return(p_value)
'''
