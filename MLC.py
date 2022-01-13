
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

def make_multi_lcs(lcs_per_telescope, telescope_ids=None):
    """
    [] [telescope] [source] -> [] [source] [telescope]
    sort according to first telescope
    create list of MultiLC objects
    telescope_ids = list of telescope identifier (strings) same order as lcs
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
                mlc_list.append(mock_lc)
        mlc = MultiLC(mlc_list, telescope_ids, prime_name)
        multi_lcs.append(mlc)
    return multi_lcs

#---------------------------------------------------------------------------------------------------
     
class MultiLC:
    def __init__(self, lc_list, lc_id, name=None):
        """
        arguments:
        lc_list = list of LightCurve objects
        lc_id = list of strings identifying each LC, eg ['Fermi', 'XRT', ...]
        """
        self.lc_list = np.array(lc_list)
        self.lc_id = np.array(lc_id) 
        if len(lc_list) != len(lc_id):
            raise ValueError('LightCurves do not match identifiers')
        self.name = str(name)
        self.n = len(lc_list)
            
    #-----------------------------------------------------------------------------------------------
       
    def plot_mlc(self, blocks=True, **kwargs):
        plt.rc('xtick', labelsize=15)
        ylen = 2 + self.n*2
        fig, axes = plt.subplots(self.n,1, figsize=(15,ylen), sharex=True)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=0)
        plt.suptitle(self.name, fontsize=20)
        for lc, lc_id, a in zip(self.lc_list, self.lc_id, axes):
            plt.sca(a)
            if blocks:
                #bblocks have to be individually initialized for each LC first
                lc.plot_bblocks(**kwargs)
            else:
                lc.plot_lc(**kwargs)
            plt.ylabel(lc_id, fontsize=15)
        plt.xlabel('Time [MJD]', fontsize=15)

    #-----------------------------------------------------------------------------------------------
    def insert_lc(self, position, lc, lc_id):
        """
        add another light curve
        make sure to avoid mutable presto-chango with 
        >>> mlc = copy.deepcopy(mlc_array)
        >>> mlc.insert_lc()
        otherwise inserted in every execution!
        """
        if position == -1:
            self.lc_list = np.append(self.lc_list, lc)
            self.lc_id = np.append(self.lc_id, lc_id)
            self.n = len(self.lc_list)
        else:
            self.lc_list = np.insert(self.lc_list, position, lc)
            self.lc_id = np.insert(self.lc_id, position, lc_id)
            self.n = len(self.lc_list)
        return(self)




''' 
FUTURE WORK:
    #-----------------------------------------------------------------------------------------------
    
'''
