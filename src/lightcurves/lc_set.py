from __future__ import annotations

import logging

import astropy.visualization.hist as fancy_hist
import numpy as np
from matplotlib import pyplot as plt

# https://docs.astropy.org/en/stable/api/astropy.visualization.hist.html

logging.basicConfig(level=logging.ERROR)  # see LC.py


class LC_Set:
    """
    Light Curve Set
    ===============
    Process and analyze all Hopjects (see HOP.py) in a set of Light Curves (see LC_new.py)
    Determine the distribution of all HOP parameters in the LC_Set
    Plot the resulting distributions of HOP properties in a fancy way

    lcs:
        list or np.array of Light Curve Objects
        Note: get_bblocks() needs to be applied to all lcs first!
        For example:
            lcs = np.zeros(len(files), dtype = object)
            for i in files:
                lc = LightCurve(time[i], flux[i], flux_error[i])
                lc.get_bblocks()
                lcs[i] = lc

    hop_method:
        a) 'baseline'
            Determine start_time/end_time to be where flux exceeds/goes under baseline
        b) 'half'
            Determine start/end of flare to be at center of valley block
        c) 'flip'
            Extrapolate behavior of flare by flipping adjacent block onto valley block
            Note: half method is used to avoid overlap (i.e. when flip > 1/2 valley block)
        d) 'sharp'
            Neglect valley block

    lc_edges:
        a) neglect:
            single start and end times are neglected
            peaks without start or end time are neglected
        b) add:
            single start and end times are neglected
            peaks without start/end: start/end is added in beginning/end of light curve

    baseline:
        e.g. mean of flux (default), median of flux, quiescent background ...

    block_min:
        Minimal number of blocks to be a flare, e.g. block_min = 2 -> no single-block flares
    """

    def __init__(
        self, lcs, hop_method="flip", lc_edges="neglect", baseline="mean", block_min=1
    ):
        # lc.get_bblock needs to be run already for each lc (do that in initialization)!!
        # check weather get_bblockswas run for lc in lcs; return error if not
        self.lcs = lcs
        mom_lc = []
        hopjects = []

        # make sure Bayesian blocks have been initialized before
        for i, lc in enumerate(lcs):
            if lc.block_pbin is None or lc.block_val is None:
                msg = "Initialize Bayesian blocks with lc.get_bblocks() first!"
                raise AttributeError(msg)

            logging.debug(str(i))
            # optional: multiprocessing for hoparound of each lc here
            if hop_method == "baseline" and baseline == "mean":
                hops = lc.find_hop("baseline", np.mean(lc.flux), lc_edges)
            elif hop_method == "baseline":
                hops = lc.find_hop("baseline", lc_edges)
            elif hop_method == "half":
                hops = lc.find_hop("half", lc_edges)
            elif hop_method == "flip":
                hops = lc.find_hop("flip", lc_edges)
            elif hop_method == "sharp":
                hops = lc.find_hop("sharp", lc_edges)
            if hops is None:
                logging.info("%d no hop found; not variable enough", i)
                continue  # skip this lc
            for hop in hops:
                if len(hop.flux) > 3:
                    hop.get_exp_fit()
                hopjects.append(hop)
                mom_lc.append(i)
        self.n_blocks = np.array([h.n_blocks for h in hopjects])
        mask = np.where(self.n_blocks > block_min)
        # eg one-block hop: n_blocks = end_block - start_block
        #                            = 5 - 3 = 2 !> 2 (minimum blocks of hop)

        # Patrick: lieber erst implementieren, wenns gebraucht wird bzw mit property decorator siehe unten
        self.mom_lc = np.array(mom_lc)[mask]
        self.hopjects = np.array(hopjects, dtype=object)[mask]
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

        ## these attributes only exist if HOP.get_exp_flare was run before
        # self.exp_tr = np.array([h.exp_tr for h in hopjects if not h.exp_tr is None])[mask]
        # self.exp_td = np.array([h.exp_td for h in hopjects])[mask]
        # self.exp_amp = np.array([h.exp_amp for h in hopjects])[mask]
        # self.exp_t0 = np.array([h.exp_t0 for h in hopjects])[mask]
        # self.exp_chisqr = np.array([h.exp_chisqr for h in hopjects])[mask]
        # self.exp_redchi = np.array([h.exp_redchi for h in hopjects])[mask]

    @property
    def exp_tr(self):
        return np.array([h.exp_tr for h in self.hopjects])

    # ----------------------------------------------------------------------------------------------
    def zcor(self, times):
        """Return times (e.g. LC_Set.dur) corrected by redshift: t_intr = t_obs / (1 + z)."""
        z = np.asarray(self.z, dtype=float)
        if np.isnan(z).any():
            msg = "Not all LCs have a redshift (NaN in self.z)."
            raise ValueError(msg)
        return np.asarray(times, dtype=float) / (1 + z)

    # ----------------------------------------------------------------------------------------------
    def plot_asym(self, N_bins=None, dens=True):
        histo, fancy_bins, p = fancy_hist(
            self.asym,
            bins="blocks",
            density=dens,
            histtype="step",
            label="Bayesian binning",
        )
        if N_bins:
            plt.hist(self.asym, N_bins)
        else:
            histo, fancy_bins, p = fancy_hist(
                self.asym,
                bins="knuth",
                density=dens,
                edgecolor="k",
                color="hotpink",
                label="Knuth bins",
            )

    def plot_dur(self, N_bins=None, dens=True):
        histo, fancy_bins, p = fancy_hist(
            self.dur, bins="blocks", density=dens, histtype="step"
        )
        if N_bins:
            plt.hist(self.asym, N_bins)
        else:
            histo, fancy_bins, p = fancy_hist(
                self.dur, bins="knuth", density=dens, edgecolor="k", color="limegreen"
            )
