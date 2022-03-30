
import numpy as np 
from matplotlib import pyplot as plt
from lightcurves.LC import LightCurve

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

def make_multi_lcs(lcs_per_telescope, lc_ids=None, lc_labels=None):
    """
    [] [telescope] [source] -> [] [source] [telescope]
    sort according to first telescope
    create list of MultiLC objects
    lc_ids = list of telescope identifier (strings) same order as lcs
    """
    multi_lcs = []
    prime_names = np.array([lc.name for lc in lcs_per_telescope[0]])
    prime_lcs = np.array([lc for lc in lcs_per_telescope[0]])
    for prime_name, prime_lc in zip(prime_names,prime_lcs):
        mlc_list = [prime_lc]
        for telescope in lcs_per_telescope[1:]:
            other_names = np.array([lc.name for lc in telescope])
            if len(np.where(other_names == prime_lc.name)[0]) > 0:
                mlc_list.append(telescope[np.where(other_names == prime_lc.name)[0][0]])
            else:
                time = np.array([np.min(prime_lc.time),np.max(prime_lc.time)])
                flux = np.array([0,0])
                flux_error = np.array([0,0])
                #mock_lc = LightCurve(prime_lc.time, prime_lc.flux, prime_lc.flux_error)
                mock_lc = LightCurve(time, flux, flux_error)
                mock_lc.get_bblocks()
                mock_lc.get_ou()
                mlc_list.append(mock_lc)
        mlc = MultiLC(mlc_list, lc_ids=lc_ids, lc_labels=lc_labels, name=prime_name)
        multi_lcs.append(mlc)
    return multi_lcs

#---------------------------------------------------------------------------------------------------
     
class MultiLC:
    def __init__(self, lc_list, lc_ids, lc_labels=None, name=None):
        """
        arguments:
        lc_list = list of LightCurve objects (lcs)
        lc_id = list of strings identifying each LC, eg ['fermi', 'xst', ...]
        lc_label = list of strings for telescope and flux unit to plot ['FACT \n[CU]',  ...]
        """
        self.lc_list = np.array(lc_list)
        self.lc_ids = np.array(lc_ids)
        if len(lc_list) != len(lc_ids):
            raise ValueError('LightCurves do not match identifiers')
        if lc_labels is None:
        	self.lc_labels = np.array(lc_ids)
        else:
        	self.lc_labels = np.array(lc_labels)
        self.name = str(name)
        self.n = len(lc_list)
            
    #-----------------------------------------------------------------------------------------------
       
    def plot_mlc(self, blocks=True, **kwargs):
        plt.rc('xtick', labelsize=15)
        ylen = 2 + self.n*2
        fig, self.axes = plt.subplots(self.n,1, figsize=(15,ylen), sharex=True)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=0)
        plt.suptitle(self.name, fontsize=20)
        for lc, lc_label, a in zip(self.lc_list, self.lc_labels, self.axes):
            plt.sca(a)
            if blocks:
                #bblocks have to be individually initialized for each LC first
                lc.plot_bblocks(**kwargs)
            else:
                lc.plot_lc(**kwargs)
            plt.ylabel(lc_label, fontsize=15)
        plt.xlabel('Time', fontsize=15)

    #-----------------------------------------------------------------------------------------------
    def insert_lc(self, position, lc, lc_id, lc_label=None):
        """
        add another light curve
        make sure to avoid mutable presto-chango with 
        >>> mlc = copy.deepcopy(mlc_array)
        >>> mlc.insert_lc()
        otherwise inserted in every execution!
        """
        if position == -1:
            self.lc_list = np.append(self.lc_list, lc)
            self.lc_id = np.append(self.lc_ids, lc_id)
            self.n = len(self.lc_list)
            if lc_label is None:
            	self.lc_labels = np.append(self.lc_labels, lc_id)
            else:
            	self.lc_labels = np.append(self.lc_labels, lc_label)
        else:
            self.lc_list = np.insert(self.lc_list, position, lc)
            self.lc_id = np.insert(self.lc_ids, position, lc_id)
            self.n = len(self.lc_list)
            if lc_label is None:
            	self.lc_labels = np.insert(self.lc_labels, position, lc_id)
            else:
            	self.lc_labels = np.insert(self.lc_labels, position, lc_label)
        return(self)

    #-----------------------------------------------------------------------------------------------
    def plot_a(self, **kwargs):
        """
        plot baseline in each light curve
        """
        for a in self.axes:
            plt.sca(a)
            plt.hlines(1e6, xmin=0, xmax=1e7, **kwargs)

    #-----------------------------------------------------------------------------------------------
    def plot_baselines(self, values, legend=None, **kwargs):
        """
        plot baseline in each light curve
        """
        """ -> this didn't work cuz everything was plotted on each axis
        -> try to create mean/median default values array and then go though plots as underneath
        if values == 'mean':
            for a, lc in zip(self.axes, self.lcs):
                plt.sca(a)
                plt.hlines(np.mean(lc.flux), xmin=np.min(lc.time), xmax=np.max(lc.time), **kwargs)
        if values == 'median':
            for a, lc in zip(self.axes, self.lcs):
                plt.sca(a)
                plt.hlines(np.median(lc.flux), xmin=np.min(lc.time), xmax=np.max(lc.time), 
                          label='median', **kwargs)
        #if values == 'qb':
        #    ...
        else:
        """
        if legend is None:
            for value, a, lc in zip(values, self.axes, self.lcs):
                plt.sca(a)
                if value:
                    plt.hlines(value, xmin=np.min(lc.time), xmax=np.max(lc.time), **kwargs)
                else:
                    continue
        else:
            for value, a, lc, label in zip(values, self.axes, self.lcs, legend):
                plt.sca(a)
                if value:
                    plt.hlines(value, xmin=np.min(lc.time), xmax=np.max(lc.time),
                               label=label, **kwargs)
                    plt.legend()
                else:
                    continue
            



''' 
FUTURE WORK:
    #-----------------------------------------------------------------------------------------------
    
'''
