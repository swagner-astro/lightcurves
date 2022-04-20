import numpy as np 
#from lightcurves.LC import LightCurve
from lightcurves.HOP import Hopject

import logging
logging.basicConfig(level=logging.ERROR)

class HopFinder():
    def __init__(self, lc_edges='neglect'):
        self.lc_edges = lc_edges

    def find_start_end(self, lc):
        raise NotImplementedError

    def find_peaks(self, lc):
        raise NotImplementedError

    def find(self, lc): 
        starts, ends = self.find_start_end(lc)
        peaks = self.find_peaks(lc)
        peaks, starts, ends = self.clean(peaks, starts, ends, lc)
        if peaks is None:
            logging.info('no hop in this light curve')
            return None 
        peaks, starts, ends = self.clean_multi_peaks(peaks, starts, ends, lc)
        if peaks is None:
            logging.info('no hop in this light curve')
            return None 
        hops = []
        for p, s, e in zip(peaks, starts, ends):
            #TBD hier könnte man noch Kriterien für hopject einfügen (e.g. bins per block/hop)
            hops.append(Hopject((s,p,e), lc, method=type(self).__name__ ))
            ## type(self).__name__ = Name der Klasse
        return hops

    def clean(self, peaks, starts, ends, lc):
        if len(peaks) < 1:
            logging.info('not variable enough, no peak found')
            return(None, None, None) 
        if self.lc_edges == 'neglect':
            if len(starts) < 1 or len(ends) < 1:
                logging.info('not variable enough, missing start or end')
                return(None, None, None)
        if self.lc_edges == 'add':
            if len(starts) < 1:
                starts = np.insert(starts, 0, lc.edges[0])
                logging.info('inserted single start in beginning of LC')
            if len(ends) < 1:
                ends = np.append(ends,lc.edges[-1])
                logging.info('inserted single end in end of LC')
        if ends[0] < peaks[0]:
            ends = np.delete(ends, 0)
            logging.info('deleted single end in beginning of LC')
            if len(ends) < 1 and self.lc_edges == 'neglect':
                logging.info('this was the only end, not variable enough')
                return(None, None, None)
            if len(ends) < 1 and self.lc_edges == 'add':
                ends = np.append(ends, lc.edges[-1])
                logging.info('inserted single end in end of LC and this is the only end')    
        if starts[-1] > peaks[-1]:
            starts = np.delete(starts, -1)
            logging.info('deleted single start in end of LC')
            if len(starts) < 1 and self.lc_edges == 'neglect':
                logging.info('this was the only start, not variable enough')
                return(None, None, None)
            if len(starts) < 1 and self.lc_edges == 'add':
                starts = np.insert(starts, 0, lc.edges[0])
                logging.info('inserted single start in beginning of LC; this is the only start')
        if peaks[0] < starts[0]:
            if self.lc_edges == 'add':
                # artificially add start
                starts = np.insert(starts, 0, lc.edges[0])
                logging.info('inserted single start in beginning of LC')
            if self.lc_edges == 'neglect':
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
            if self.lc_edges == 'add':
                # artificially add end
                ends = np.append(ends, lc.edges[-1])
                logging.info('inserted single end in end of LC') 
            if self.lc_edges == 'neglect':
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

    def clean_multi_peaks(self, peaks, starts, ends, lc):
        # baseline method could result in multiple peaks within one HOP 
        # -> neglect smaller peak (not so senseful..)
        while len(ends) < len(peaks):
            for x,_ in enumerate(ends):
                if ends[x] > peaks[x+1]:
                    if (lc.block_val[lc.bb_i(peaks[x])] 
                        < lc.block_val[lc.bb_i(peaks[x+1])]):
                            peaks = np.delete(peaks, x)
                    elif (lc.block_val[lc.bb_i(peaks[x])] 
                          >= lc.block_val[lc.bb_i(peaks[x+1])]):
                            peaks = np.delete(peaks, x+1)
                    logging.info('neglected double peak in HOP ' + str(x))
                    break
        return peaks, starts, ends


#----------------------------------------------------------------------------------------------
class HopFinderBaseline(HopFinder):
    def find_peaks(self, lc):
        diff = np.diff(lc.block_val)
        peaks = [] #time of all local peaks over baseline (in units of edges = units of time)
        for i in range(1,len(diff)):
            # if previous rising; this falling
            if diff[i-1] > 0 and diff[i] < 0:
                if lc.block_val[i] > lc.baseline:
                    # peak_time = middle of peak block
                    peaks.append(lc.edges[i] + (lc.edges[i+1] - lc.edges[i]) /2)
        return peaks

    def find_start_end(self, lc):
        starts = []  
        ends = []    
        for i in range(len(lc.block_val)-1):
            # if this smaller; next one higher
            if lc.block_val[i] < lc.baseline and lc.block_val[i+1] > lc.baseline:
                starts.append(lc.edges[i+1])
            # if this larger; next one lower
            if lc.block_val[i] > lc.baseline and lc.block_val[i+1] < lc.baseline:
                ends.append(lc.edges[i+1])
        return starts, ends

#----------------------------------------------------------------------------------------------
class HopFinderProcedure(HopFinder):
    def find_peaks(self, lc):
        diff = np.diff(lc.block_val)
        peaks = [] # time of all local peaks (units of edges, i.e. units of time)
        for i in range(1,len(diff)):
            # peak = previous rising; this falling
            if diff[i-1] > 0 and diff[i] < 0:
                # peak_time = middle of peak block
                peaks.append(lc.edges[i] + (lc.edges[i+1] - lc.edges[i]) /2)
        return peaks

    def change_point(self, edges, i):
        raise NotImplementedError

    def find_start_end(self, lc):
        starts = []
        ends = []
        diff = np.diff(lc.block_val)
        for i in range(1,len(diff)):
            # change = previous falling; this rising
            if diff[i-1] < 0 and diff[i] > 0: 
                start, end = self.change_point(lc.edges, i)
                starts.append(start)
                ends.append(end)
        return starts, ends

#----------------------------------------------------------------------------------------------
class HopFinderHalf(HopFinderProcedure):
    def change_point(self, edges, i):
        half_block_time = (edges[i+1] - edges[i]) / 2
        return edges[i+1] - half_block_time, edges[i] + half_block_time

class HopFinderFlip(HopFinderProcedure):
    def change_point(self, edges, i):
        half_block_time = (edges[i+1] - edges[i]) / 2
        #clap previous block onto change block
        clap_from_left = edges[i] - edges[i-1]
        #clap following block onto change block
        clap_from_right = edges[i+2] - edges[i+1]
        e = edges[i] + min(half_block_time, clap_from_left)
        s = edges[i+1] - min(half_block_time, clap_from_right)
        return s,e

class HopFinderSharp(HopFinderProcedure):
    def change_point(self, edges, i):
        return edges[i+1], edges[i]


"""
FUTURE WORK:
    #-----------------------------------------------------------------------------------------------
     -> implement a more reasonable Baseline method (i.e. treat multiple peaks different)
"""

