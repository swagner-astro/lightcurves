"""
Light Curve Class - Sarah Wagner
================================

Takes data of a light curve (time, flux, flux error) 
and characterizes its first order varaibility properties

changes:
* added name and z in LC initialization

with code from:
* BB_LC_def.py
* LCs.py
* FF_def_2.py

INDEX:
* flux_puffer(flux, threshold, threshold_error) 
* asym(peak_time, start_time, end_time)
* LightCurve(time, flux, flux_error)
* get_bblocks(self, gamma_value=None, p0_value=0.05)
* get_bblocks_above(self, threshold, pass_gamma_value=None, pass_p0_value=None)
   -> set small block_val to threshold and neglect edges under threshold
* plot_bblocks(self)
* bb_i(self, time)
   -> block index at certain time
* handle_hops(self, peaks, starts, ends, lc_edges)
   -> handle mismatches of peak_time, start_time, and end_time combinations from get_hop_xy
* get_hop_bl(self, baseline=None, lc_edges='neglect')
* hop_procedure(self, method, lc_edges)
   -> used to find peaks and changes of all hop methods (peaks returned for all methods)
* get_hop_half(self, lc_edges='neglect')
   -> start and end are at the middle of change block
* get_hop_halfclap(self, lc_edges='neglect')
   -> min(middle and clap) 
* get_hop_sharp(self, lc_edges='neglect')
   -> drop/neglect change block
* hop_around(self, gamma_value=None, p0_value=0.05, lc_edges='neglect')
   -> initialize all hop properties for self
* plot_hop(self, start_times, end_times)
* plot_all_hop(self, gamma_value=None, p0_value=0.05, lc_edges='neglect')
* hopses(self, gamma_value=None, p0_value=0.05, lc_edges='neglect') --> zuckerl ;)

FUTURE:
* LC could be parent class and then LC->bblocks and LC->PSD and LC->fluxdisti.. meh
"""

import logging
logging.basicConfig(level=logging.ERROR)
"""
set logging to the desired level
logging options:
DEBUG:      whatever happens will be thrown at you
INFO:       confirmation that things are working as expected
WARNING:    sth unexpected happened
ERROR:      sth didn't work, abort mission

Luca: auch raise Exception(text) is option
""" 

import numpy as np 
from matplotlib import pyplot as plt
import astropy.stats.bayesian_blocks as bblocks
#https://docs.astropy.org/en/stable/api/astropy.stats.bayesian_blocks.html


def flux_puffer(flux, threshold, threshold_error):
    # ATTENTION! this returns artificial flux values!
    # set every flux bin under threshold to threshold before initializing light curve
    flux_new = np.where(flux > threshold, flux, threshold)
    flux_error_new = np.where(flux > threshold, flux_error, th_error)
    return(flux_new, flux_error_new)
    ## bblocks of flux_new represent significant variations wrt threshold -> flare definition?
    ## equivalent(?) approach could be to make sigma schlauch around threshold and see which blocks are above...


class LightCurve:

    def __init__(self, time, flux, flux_error, name=None, z=None):
        
    #raise error if there are nans (np.isnan(flux), test if array is Ture anywhere) and check if all 3 have same len
    #fragen: gscheide where mask mit mehr conditions
        self.time = time
        self.flux = flux
        self.flux_error = flux_error
        self.name = name
        self.z = z


    def plot_lc(self, data_color='k', data_label='obs flux'):
        plt.errorbar(x=self.time, y=self.flux, yerr=self.flux_error, label=data_label, ecolor=data_color, elinewidth=1, 
            linewidth=0, marker='+', markersize=3, color=data_color)

    #--------------------------------------------------------------------------------------------------------------------------------
    def get_bblocks(self, gamma_value=None, p0_value=0.05): 
        # get Bayesian Blocks for light curve
        self.edges = bblocks(t=self.time, x=self.flux, sigma=self.flux_error, fitness='measures', gamma=gamma_value, p0=p0_value)
        logging.debug('got edges for light curve')

        if len(self.edges) <= 2:
            logging.warning('light curve is constant; only one bayesian block found.')
            self.block_pbin = np.ones(len(self.flux)) * np.mean(self.flux)
            self.block_val = np.array([np.mean(self.flux)])
            self.block_val_error = np.array([np.std(self.flux)])
            self.edge_index = np.array([0,-1])
            self.edges = np.array([self.time[0],self.time[-1]])
            return(self.block_pbin, self.block_val, self.block_val_error, self.edge_index, self.edges)
        # get edge_index to access other arrays
        self.edge_index = np.array([np.where(self.time >= self.edges[i])[0][0] for i,_ in enumerate(self.edges)])
        #change last entry such that loop over [j:j+1] gives all BBs 
        self.edge_index[-1] += 1

        # determine flux value (mean) and error (Gaussian propagation) for each block
        self.block_val = np.zeros(len(self.edge_index)-1)  
        self.block_val_error = np.zeros(len(self.edge_index)-1) 
        for j in range(len(self.edge_index)-1):
            self.block_val[j] = np.mean(self.flux[self.edge_index[j]:self.edge_index[j+1]])
            self.block_val_error[j] = np.sqrt(np.sum(
                self.flux_error[self.edge_index[j]:self.edge_index[j+1]]**2))/(self.edge_index[j+1]-self.edge_index[j])

        # create BB array corresponding to data
        self.block_pbin = np.zeros(len(self.flux))
        for k,_ in enumerate(self.block_val):
            self.block_pbin[self.edge_index[k]:self.edge_index[k+1]] = self.block_val[k]
        logging.debug('got block parameters for light curve')

        return(self.block_pbin, self.block_val, self.block_val_error, self.edge_index, self.edges)

    #--------------------------------------------------------------------------------------------------------------------------------
    def get_bblocks_above(self, threshold, pass_gamma_value=None, pass_p0_value=None):
        # first get_bblocks
        # then only consider the ones over threshold (e.g. quiescent background)
        # i.e. set small block_val to threshold and neglect edges under threshold
        self.block_pbin = np.where(self.block_pbin > threshold, self.block_pbin, threshold)
        self.block_val = np.where(self.block_val > threshold, self.block_val, threshold)
        block_mask = np.ones(len(self.block_val), dtype = bool)
        edge_mask = np.ones(len(self.edges), dtype=bool)
        for i in range(len(self.block_val)-1):
            if self.block_val[i] == threshold and self.block_val[i+1] == threshold:
                block_mask[i+1]=False
                edge_mask[i+1] = False
        self.block_val = self.block_val[block_mask]
        self.block_val_error = self.block_val_error[block_mask]
        self.edge_index = self.edge_index[edge_mask]
        self.edges = self.edges[edge_mask]
        return(self.block_pbin, self.block_val, self.block_val_error, self.edge_index, self.edges)

    #--------------------------------------------------------------------------------------------------------------------------------
    def plot_bblocks(self, bb_color='steelblue', data_color='k', data_label='obs flux'):
        # first get_bblocks
        plt.step(self.time, self.block_pbin, where='mid', linewidth=1, label='bblocks', color=bb_color, zorder=1000)
        plt.errorbar(x=self.time, y=self.flux, yerr=self.flux_error, label=data_label, ecolor=data_color, elinewidth=1, 
            linewidth=0, marker='+', markersize=3, color=data_color)
        ##if get_bblocks is not run the following error is returned which could be rephrased to: execute get_bblocks first!
        ##AttributeError: 'LightCurve' object has no attribute 'block_pbin'

    #--------------------------------------------------------------------------------------------------------------------------------
    def bb_i(self, t):
        # translates time to index of corresponding bayesian block (eg block_value of peak_time)
        # only works for half and halfclap because baseline and sharp use edges as start and end time!!!!!
        if t == self.edges[0]:
            return(int(0))
        else:
            block_index = [e for e in range(len(self.edges)-1) if t > self.edges[e] and t <= self.edges[e+1] ]
            return(int(block_index[0]))
    #####To Be Still TESTEEEED ############
    def bb_i_start(self,t):
        #translates time to index of corresponding bayesian block, assuming that time is start time
        #i.e. if time=edge -> take block on the left
        block_index = [e for e in range(len(self.edges)-1) if t >= self.edges[e] and t < self.edges[e+1] ]
        return(int(block_index[0]))
    def bb_i_end(self,t):
        #translates time to index of corresponding bayesian block, assuming that time is end time
        #i.e. if time=edge -> take block on the right
        block_index = [e for e in range(len(self.edges)-1) if t > self.edges[e] and t <= self.edges[e+1] ]
        return(int(block_index[0]))
  
    #--------------------------------------------------------------------------------------------------------------------------------
    def handle_hops(self, peaks, starts, ends, lc_edges):
        # handeling mismatches of peak_time, start_time, and end_time combinations from get_hop_xy
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
                logging.info('inserted single start in beginning of LC and this is the only start')
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
    
        
    #--------------------------------------------------------------------------------------------------------------------------------
    def get_hop_baseline(self, baseline=None, lc_edges='neglect'):
        # LC_edges == 'add' -> start/ end of flare is artificially added at start/end of LC
        # LC_edges == 'neglect' -> incomplete flares (missing start or end) are conservatively neglected
        if baseline == None:
            baseline = np.mean(self.flux)
            logging.info('use default baseline: mean(flux)')

        diff = np.diff(self.block_val)
        peak_times = [] #time of all local peaks over baseline (units of edges, i.e. units of time)
        for i in range(1,len(diff)):
            # previous rising; this falling
            if diff[i-1] > 0 and diff[i] < 0:
                if self.block_val[i] > baseline:
                    # peak_time = middle of peak block
                    peak_times.append(self.edges[i] + (self.edges[i+1] - self.edges[i]) /2)
        start_times = []  
        end_times = []    
        for i in range(len(self.block_val)-1):
            if self.block_val[i] < baseline and self.block_val[i+1] > baseline:
                #old:
                ## start_time = last bin under baseline
                #start_times.append(self.time[self.edge_index[i+1]-1])
                ##old version: self.edges[i+1] for start and end was changed bc of bb_i
                #-> fixed with bb_end and bb_start
                start_times.append(self.edges[i+1])
            if self.block_val[i] > baseline and self.block_val[i+1] < baseline:
                #old:
                # end_time = first bin under baseline
                #end_times.append(self.time[self.edge_index[i+1]])  
                end_times.append(self.edges[i+1])
        peak_times, start_times, end_times = self.handle_hops(
            np.array(peak_times), np.array(start_times), np.array(end_times), lc_edges)
        # neglect multiple peaks within one HOP (not so senseful.. for all peaks use other HOPs or check out peak_mask in FF_def_2)
        if peak_times is None:
            self.start_times_bl, self.end_times_bl = None, None
            logging.warning('light curve is not variable enough; no hop found.')
            return(None)
        while len(end_times) < len(peak_times):
            for x,_ in enumerate(end_times):
                if end_times[x] > peak_times[x+1]:
                    if self.block_val[self.bb_i(peak_times[x])] < self.block_val[self.bb_i(peak_times[x+1])]:
                        peak_times = np.delete(peak_times, x)
                    elif self.block_val[self.bb_i(peak_times[x])] >= self.block_val[self.bb_i(peak_times[x+1])]:
                        peak_times = np.delete(peak_times, x+1)
                    logging.info('neglected double peak in HOP ' + str(x))
                    break
        self.peak_times_bl = peak_times
        self.start_times_bl = start_times
        self.end_times_bl = end_times
        return(np.array([start_times, peak_times, end_times]).transpose())

    #--------------------------------------------------------------------------------------------------------------------------------
    def hop_procedure(self, method, lc_edges):
        # find peak, start and end of each hop gorup for half, halfclap and sharp mehtod
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
                if method == 'halfclap':
                    clap_from_left = self.edges[i] - self.edges[i-1] #clap previous block onto change block
                    clap_from_right = self.edges[i+2] - self.edges[i+1] #clap following block onto change block
                    end_times.append(self.edges[i] + np.minimum(half_block_time, clap_from_left))
                    start_times.append(self.edges[i+1] - np.minimum(half_block_time, clap_from_right))
                if method == 'sharp':
                    ##A) use edge: makes problems in bbi 
                    start_times.append(self.edges[i+1])
                    end_times.append(self.edges[i])
                    ##B) use bin: take first/last data bin -> if large gap peak (middle of block) could be left/right of that! -> Problem!
                    #start_times.append(self.time[self.edge_index[i+1]])
                    #end_times.append(self.time[self.edge_index[i]-1])

        return(self.handle_hops(np.array(peak_times), np.array(start_times), np.array(end_times), lc_edges))
                
    def get_hop_half(self, lc_edges='neglect'):
        self.peak_times, self.start_times_half, self.end_times_half = self.hop_procedure('half', lc_edges)
        if self.peak_times is None:
            logging.warning('light curve is not variable enough; no hop found.')
            return(None)
        else:
            return((np.array([self.start_times_half, self.peak_times, self.end_times_half])).transpose())
    
    def get_hop_hc(self, lc_edges='neglect'):
        self.peak_times, self.start_times_hc, self.end_times_hc = self.hop_procedure('halfclap', lc_edges)
        if self.peak_times is None:
            logging.warning('light curve is not variable enough; no hop found.')
            return(None)
        else:
            return((np.array([self.start_times_hc, self.peak_times, self.end_times_hc])).transpose())
    
    def get_hop_sharp(self, lc_edges='neglect'):
        self.peak_times, self.start_times_sharp, self.end_times_sharp = self.hop_procedure('sharp', lc_edges)
        if self.peak_times is None:
            logging.warning('light curve is not variable enough; no hop found.')
            return(None)
        else:
            return((np.array([self.start_times_sharp, self.peak_times, self.end_times_sharp])).transpose())

    #--------------------------------------------------------------------------------------------------------------------------------
    def hop_around(self, gamma_value=None, p0_value=0.05, lc_edges='neglect'):
        # initialize bblocks and all hop methods in one go
        self.get_bblocks(gamma_value, p0_value)
        self.get_hop_baseline(lc_edges=lc_edges) # necessary cuz otherwise it would think this is baseline
        self.get_hop_half(lc_edges)
        self.get_hop_hc(lc_edges)
        self.get_hop_sharp(lc_edges)
        logging.debug('hoppped around')

    #--------------------------------------------------------------------------------------------------------------------------------
    def plot_hop(self, start_times, end_times): # lc.plot_hop(lc.start_times_hc, lc.end_times_hc) 
        if start_times is None:
            return() # nop hop found
        for i,_ in enumerate(start_times):
            x = np.linspace(start_times[i], end_times[i])
            y = np.ones(len(x)) * np.max(self.flux)
            y1 = np.zeros(len(x))
            if i == 0:
                plt.fill_between(x, y, y1, step="mid", color='lightsalmon', alpha=0.2, label='hop', zorder=0)
            if i == 1:
                plt.fill_between(x, y, y1, step="mid", color='orchid', alpha=0.2, label='hop', zorder=0)
            elif i % 2:
                plt.fill_between(x, y, y1, step="mid", color='orchid', alpha=0.2, zorder=0)
            elif i != 0:
                plt.fill_between(x, y, y1, step="mid", color='lightsalmon', alpha=0.2, zorder=0)

    def plot_hop_method(self, method = 'halfclap'):
        if method == 'half':
            self.plot_hop(self.start_times_half, self.end_times_half)
        if method == 'halfclap':
            self.plot_hop(self.start_times_hc, self.end_times_hc)
        if method == 'sharp':
            self.plot_hop(self.start_times_sharp, self.end_times_sharp)
        if method == 'baseline':
        	self.plot_hop(self.start_times_bl, self.end_times_bl)

    #--------------------------------------------------------------------------------------------------------------------------------
    def plot_all_hop(self, gamma_value=None, p0_value=0.05, lc_edges='neglect'):
        self.hop_around(gamma_value, p0_value, lc_edges)
        fig = plt.figure(0,(15,7))
        plt.suptitle('hop methods')

        ax0 = fig.add_subplot(511)
        self.plot_bblocks()
        plt.hlines(np.mean(self.flux), xmin=min(self.time), xmax=max(self.time), color='deeppink',
                   linewidth=1, label='baseline = mean', zorder=100)
        self.plot_hop(self.start_times_bl, self.end_times_bl)
        plt.ylabel('baseline')
        ax1 = fig.add_subplot(512)
        self.plot_bblocks()
        self.plot_hop(self.start_times_half, self.end_times_half)
        plt.ylabel('half')
        ax2 = fig.add_subplot(513)
        self.plot_bblocks()
        self.plot_hop(self.start_times_hc, self.end_times_hc)
        plt.ylabel('halfclap')
        ax3 = fig.add_subplot(514)
        self.plot_bblocks()
        self.plot_hop(self.start_times_sharp, self.end_times_sharp)
        plt.ylabel('sharp')
        fig.subplots_adjust(hspace=0)

    #--------------------------------------------------------------------------------------------------------------------------------
    def hopses(self, gamma_value=None, p0_value=0.05, lc_edges='neglect'):
        # HIPPETES HOPPETES DOMINAEEE
        self.get_bblocks(gamma_value, p0_value)
        baseline = np.mean(self.flux)
        self.get_hop_bl(lc_edges=lc_edges)
        self.get_hop_half(lc_edges)
        self.get_hop_hc(lc_edges)
        self.get_hop_sharp(lc_edges)
        self.plot_all_hop(gamma_value, p0_value, lc_edges)
        plt.show()
        from IPython.display import YouTubeVideo
        import time as pytime
        pytime.sleep(2)
        print('...hippetes hoppetes..')
        pytime.sleep(2)
        pope_snowball = YouTubeVideo('XBSAbcTzOjk')
        display(pope_snowball)

''' 
    #--------------------------------------------------------------------------------------------------------------------------------
    def get_ou_params(self):
    	#check for negative values
    	#check for flux units
    	return(mu, theta, sigma)

    #--------------------------------------------------------------------------------------------------------------------------------
    def psd(self):
    	#
    	return(psd_slope)

    def plot_psd(self):
    	plot se psd
'''




