'''
Light Curve Set - Sarah Wagner
==============================

Process and analyze all Hopjects (see HOP.py) in a set of Light Curves (see LC_new.py)

'''

## best way to create array of lcs:
#lcs = np.zeros(len(files), dtype = object)
#[lcs[i] = LightCurve(time, flux, flux_error) for i in files]


import numpy as np
from matplotlib import pyplot as plt
import astropy.visualization.hist as fancy_hist
from selfmade_py.LC_new import LightCurve
from selfmade_py.HOP import Hopject
import logging
logging.basicConfig(level=logging.ERROR) #see LC_new

class LC_Set:

    def __init__(self, lcs, hop_method='halfclap', lc_edges='neglect', baseline='mean', block_min=1):
        # lcs = list or np.array of Light Curve Objects and get_bblocks was done for each
        # lc.get_bblock needs to be run already for each lc (do that in initialization)!!
        # check weather get_bblockswas run for lc in lcs; return error if not
        self.lcs = lcs
        mom_lc = []
        names = []
        hopjects = []

        for i,lc in enumerate(lcs):
            logging.debug(str(i))
            #optional: multiprocessing for hoparound of each lc here
            if hop_method == 'baseline' and baseline == 'mean':
                hops = lc.get_hop_baseline(np.mean(lc.flux),lc_edges)
            elif hop_method == 'baseline':
                hops = lc.get_hop_bl(baseline,lc_edges)
            elif hop_method == 'half':
                hops = lc.get_hop_half(lc_edges)
            elif hop_method == 'halfclap':
                hops = lc.get_hop_hc(lc_edges)
            elif hop_method == 'sharp':
                hops = lc.get_hop_sharp(lc_edges)
            if hops is None:
                logging.info(str(i)+ ' no hop found; not variable enough')
                continue #skip this lc
            for hop in hops:
                hopjects.append(Hopject(hop, lc)) #hop = hop_params (start, peak, end)
                mom_lc.append(i)
                names.append(lc.name)
        self.n_blocks = np.array([h.n_blocks for h in hopjects])
        mask = np.where(self.n_blocks > block_min)
        # eg one-block hop: n_blocks = end_block - start_block = 5 - 3 = 2 !> 2 (minimum blocks of hop)

        self.mom_lc = np.array(mom_lc)[mask]
        self.names = np.array(names)[mask] #probably not necessary cuz there is mom_lc (index) anyway
        self.hopjects = np.array(hopjects, dtype = object)[mask]
        self.dur = np.array([h.dur for h in hopjects])[mask]
        self.rise_time = np.array([h.rise_time for h in hopjects])[mask]
        self.decay_time = np.array([h.decay_time for h in hopjects])[mask]
        self.asym = np.array([h.asym for h in hopjects])[mask]
        self.start_flux = np.array([h.start_flux for h in hopjects])[mask]
        self.peak_flux = np.array([h.peak_flux for h in hopjects])[mask]
        self.end_flux = np.array([h.end_flux for h in hopjects])[mask]
        self.rise_flux = np.array([h.rise_flux for h in hopjects])[mask]
        self.decay_flux = np.array([h.decay_flux for h in hopjects])[mask]
        self.z = np.array([h.z for h in hopjects])[mask]


    def zcor(self, times): #times = eg LC_Set.dur
        if len(np.where(np.isnan(self.z) == True)[0]) > 0:
            print('Error: not all LCs have a redshift')
        else:
            times_intr = times / (1 + self.z)
            return(times_intr)


    def plot_asym(self, N_bins=None, dens=True):
        histo, fancy_bins, p = fancy_hist(self.asym, bins='blocks', density=dens, histtype='step')
        if N_bins:
            plt.hist(self.asym, N_bins)
        else:
            histo, fancy_bins, p = fancy_hist(self.asym, bins='knuth', density=dens, edgecolor='k', color='hotpink')

    def plot_dur(self, N_bins=None, dens=True):
        histo, fancy_bins, p = fancy_hist(self.dur, bins='blocks', density=dens, histtype='step')
        if N_bins:
            plt.hist(self.asym, N_bins)
        else:
            histo, fancy_bins, p = fancy_hist(self.dur, bins='knuth', density=dens, edgecolor='k', color='limegreen')

    #TBD: find a nice way to scatter plot see scatter pro 
    def plot_dt(self):
        plt.scatter(self.rise_time, self.decay_time)

    def plot_dF(self):
        plt.scatter(self.rise_flux, self.decay_flux)








