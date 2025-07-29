from __future__ import annotations

import logging
import pickle

import astropy
import astropy.stats.bayesian_blocks as bblocks
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes  # for type hints only

# https://docs.astropy.org/en/stable/api/astropy.stats.bayesian_blocks.html
from lightcurves.HopFinder import *

logging.basicConfig(level=logging.ERROR)
"""
set logging to the desired level
logging options:
DEBUG:      whatever happens will be thrown at you
INFO:       confirmation that things are working as expected
WARNING:    sth unexpected happened
ERROR:      sth didn't work, abort mission
"""


def load_lc(path: str) -> LightCurve:
    """
    Load a pickled LightCurve instance from a file created with `save_lc()`.

    WARNING
    -------
    Uses pickle. Loaded instance reflects the class at save-time.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def load_lc_npy(path: str) -> LightCurve:
    """
    Load pickled LightCurve instance from numpy array created with `save_lc()`.

    WARNING
    -------
    Uses pickle. Loaded instance reflects the class at save-time.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def load_lc_csv(path: str) -> LightCurve:
    """
    Load a pickled LightCurve instance from a CSV file saved with `save_csv()`.
    """
    a = np.genfromtxt(path)
    return LightCurve(a[0], a[1], a[2])


def flux_puffer(
    flux: np.ndarray, flux_error: np.ndarray, threshold: float, threshold_error: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Set every flux value under a threshold to the threshold, with an error.

    Use Case
    --------
    Apply Bayesian blocks to puffered LC to detect significant variations
    relative to threshold (e.g. flares).

    WARNING
    -------
    Returns artificial flux values! Use with caution.

    Parameters
    ----------
    flux : array_like
        The input flux array.
    flux_error : array_like
        The uncertainties associated with each flux value.
    threshold : float
        Minimum flux value. Values below this will be replaced with this value.
    threshold_error : float
        Uncertainty to assign to new thrshold values.

    Returns
    -------
    flux_new : array_like
        Modified flux array with threshold applied.
    flux_error_new : array like
        Associated uncertainties, modified analogously.
    """
    flux_new = np.where(flux > threshold, flux, threshold)
    flux_error_new = np.where(flux > threshold, flux_error, th_error)
    return (flux_new, flux_error_new)


def clean_data(
    time: np.ndarray,
    flux: np.ndarray,
    flux_error: np.ndarray,
    ts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Remove NaNs from flux and flux_error and drop duplicate time stamps.

    WARNING
    -------
    This function deletes flux values! Use with caution.

    Parameters
    ----------
    time : np.ndarray
        Time values.
    flux : np.ndarray
        Flux values.
    flux_error : np.ndarray
        Associated flux uncertainties.
    ts : np.ndarray, optional
        Optional secondary array (e.g. TestStatistic) to clean in parallel.

    Returns
    -------
    time_unique : array_like
        Cleaned and unique time values.
    flux_clean : array_like
        Cleaned flux values.
    flux_error_clean : array_like
        Cleaned flux error values.
    ts_clean : array_like
        Cleaned TestStatistic or optional array, if provided.

    TBD
    ---
    this could be a LC method and return a new LC or alter the instance?
    """
    # Mask NaN values of flux and flux_error
    nan_mask = ~np.isnan(flux) & ~np.isnan(flux_error)
    flux_ = flux[nan_mask]
    flux_error_ = flux_error[nan_mask]
    time_ = time[nan_mask]
    logging.info(f"Deleted {len(flux) - len(flux_)} NaN entries.")

    # Remove duplicate times (keep first occurrence)
    time_unique, time_unique_id = np.unique(time_, return_index=True)
    flux_clean = flux_[time_unique_id]
    flux_error_clean = flux_error_[time_unique_id]
    logging.info(f"Deleted {len(time_) - len(time_unique)} time duplicates")
    if ts is not None:
        ts_ = ts[nan_mask]
        ts_clean = ts_[time_unique_id]
        return (time_unique, flux_clean, flux_error_clean, ts_clean)
    if ts is None:
        return (time_unique, flux_clean, flux_error_clean, None)


def get_gti_iis(
    time: np.ndarray, n_gaps: int, n_pick: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Determine Good Time Intervals (GTIs) of LC, based on largest time gaps.
    Useful for instruments with seasonal or periodic gaps, e.g., FACT.

    Parameters
    ----------
    time : np.ndarray
        Time values.
    n_gaps : int
        Number of gaps to divide the data into GTIs.
    n_pick : int, optional
        If provided, return only the n_pick longest GTIs.

    Returns
    -------
    GTI_start_ii : np.ndarray
        Start indices of the good time intervals.
    GTI_end_ii : np.ndarray
        End indices of the good time intervals.
    """
    diff = np.diff(time)
    diff1 = np.sort(diff)
    ii = [x for x in range(len(diff)) if diff[x] in diff1[-n_gaps:]]
    # index of the 10 longest gaps... risky business due to precision issues
    GTI_start_ii = np.array(ii) + 1
    GTI_start_ii = np.insert(GTI_start_ii, 0, 0)
    GTI_end_ii = np.array(ii)
    GTI_end_ii = np.append(GTI_end_ii, len(time) - 1)
    if n_pick:
        # only consider the n_pick longest GTIs
        # TBD: double check, might compute index differences, not time gaps..
        gap_len = np.array(
            [t - s for s, t in zip(GTI_start_ii, GTI_end_ii, strict=False)]
        )
        gap_len1 = np.sort(gap_len)
        ii = [x for x in range(len(gap_len)) if gap_len[x] in gap_len1[-n_pick:]]
        # n_gaps = considered gaps (longest not gaps)
        GTI_start_ii_ = GTI_start_ii[ii]
        GTI_end_ii_ = GTI_end_ii[ii]
        return GTI_start_ii_, GTI_end_ii_
    return GTI_start_ii, GTI_end_ii


def make_gti_lcs(lc: LightCurve, n_gaps: int, n_pick: int = None) -> np.ndarray:
    """
    Divide LC into several LCs (Good Time Intervals = GTIs) based on largest
    time gaps. Optionally only pick largest GTIs.

    Parameters
    ----------
    lc : "LightCurve"
        LightCurve instance to be divided.
    n_gaps : int
        Number of gaps to divide the data into GTIs.
    n_pick : int, optional
        If provided, return only the n_pick longest GTIs.

    Returns
    -------
    chunks : np.ndarray
        Array of LightCurve objects based on GTIs.
    """
    gti_starts, gti_ends = get_gti_iis(lc.time, n_gaps, n_pick)
    if n_pick is None:
        n_pick = n_gaps + 1  # select all
    chunks = []
    for g in range(n_pick):
        gti_lc = LightCurve(
            lc.time[gti_starts[g] : gti_ends[g] + 1],
            lc.flux[gti_starts[g] : gti_ends[g] + 1],
            lc.flux_error[gti_starts[g] : gti_ends[g] + 1],
            name=lc.name,
            z=lc.z,
        )
        chunks.append(gti_lc)
    return np.array(chunks, dtype=object)


# -----------------------------------------------------------------------------
class LightCurve:
    """
    A class to represent a light curve with time series data (time, flux, and
    flux uncertainty) and additional information.

    Parameters
    ----------
    time : array_like
        Time values of the light curve.
    flux : array_like
        Measured flux values.
    flux_error : array_like
        Uncertainties associated with the flux measurements.
    time_format : str, optional
        Format of `time` corresponding to astropy.time.Time (eg, 'MJD', 'JD').
    name : str, optional
        Source name or ID.
    z : float, optional
        Redshift associated with the observed source.
    telescope : str, optional
        Name of the telescope or instrument used.
    cadence : float, optional
        Time spacing between observations, in the same units as `time`.

    Attributes
    ----------
    time : np.ndarray
        Array of time values.
    flux : np.ndarray
        Array of flux values.
    flux_error : np.ndarray
        Array of flux uncertainties.
    time_format : str or None
        Format of `time` corresponding to astropy.time.Time (eg, 'MJD', 'JD').
    name : str or None
        Source name or ID.
    z : float or None
        Redshift of the source.
    telescope : str or None
        Telescope or instrument name.
    cadence : float or None
        Observational cadence.

    Examples
    --------
    >>> lc = LightCurve(
    ...     time=[1, 2, 3],
    ...     flux=[10.0, 11.5, 10.2],
    ...     flux_error=[0.2, 0.3, 0.2],
    ...     name="Star A"
    ... )
    >>> print(lc.name)
    Star A
    """

    def __init__(
        self,
        time: np.ndarray | list,
        flux: np.ndarray | list,
        flux_error: np.ndarray | list,
        time_format: str | None = None,
        name: str | None = None,
        z: float | None = None,
        telescope: str | None = None,
        cadence: float | None = None,
    ):
        self.time = np.array(time)
        self.flux = np.array(flux)
        self.flux_error = np.array(flux_error)
        self.time_format = time_format
        self.name = name
        self.z = z
        self.telescope = telescope
        self.cadence = cadence
        if len(time) != len(flux) or len(time) != len(flux_error):
            raise ValueError("Input arrays do not have same length")
        if len(flux[np.isnan(flux)]) > 0 or len(flux_error[np.isnan(flux_error)]) > 0:
            raise TypeError("flux or flux_error contain np.nan values")
        if len(time) != len(np.unique(time)):
            raise ValueError("time contains duplicate values")
        if time_format:
            """ format of the astropy.time.Time object """
            self.astropy_time = astropy.time.Time(time, format=time_format)

    def __repr__(self):
        """
        String representation of the LightCurve instance, eg for print
        TBD: could be extended (eg with bblocks)
        """
        return (
            f"LightCurve (bins = {len(self.flux)}, "
            f"name = {self.name}, "
            f"cadende = {self.cadence}, "
            f"telescope = {self.telescope}, "
            f"z = {self.z})"
        )

    def __len__(self):
        return len(self.time)

    def __getitem__(self, inbr: int | slice | list[int]) -> np.ndarray | LightCurve:
        """
        Access elements or subsets of the LightCurve using indexing or slicing.

        Supports:
        - Integer index: returns one bin (time, flux, flux_error), eg `lc[i]`
        - Slice: returns a new LightCurve, eg `lc[i:j]`
        - List of indices: returns selected bins (time, flux, flux_error),
          eg `lc[[i1, i2, i3]]`

        Parameters
        ----------
        inbr : int, slice, or list of int
        Index or indices specifying the data to retrieve.

        Returns
        -------
        np.ndarray or LightCurve
            - If `inbr` is an int or list of ints: returns an array of selected
              time, flux and flux_error values tuples.
            - If `inbr` is a slice: returns a new LightCurve instance.

        Raises
        ------
        TypeError
            If `inbr` is not int, slice, or list of int.

        """
        if isinstance(inbr, int):
            return np.array([self.time[inbr], self.flux[inbr], self.flux_error[inbr]])
        if isinstance(inbr, slice):
            return LightCurve(
                self.time[inbr],
                self.flux[inbr],
                self.flux_error[inbr],
                self.time_format,
                self.name,
                self.z,
                self.telescope,
                self.cadence,
            )
        if isinstance(inbr, list):
            # can't be implemented with 'int or list' -> confusion with slice
            return np.array([self.time[inbr], self.flux[inbr], self.flux_error[inbr]])
        raise TypeError("Index must be int, slice, or list of ints.")

    def select_by_time(self, t_min: float, t_max: float) -> LightCurve:
        """
        Select a portion of the light curve between two time bounds.

        Parameters
        ----------
        t_min : float
            Start time of the interval.
        t_max : float
            End time of the interval.

        Returns
        -------
        LightCurve
            A new LightCurve instance from `t_min` to `t_max`.
        """
        i_s = np.where(self.time >= t_min)[0][0]
        i_e = np.where(self.time >= t_max)[0][0]
        return self.__getitem__(slice(i_s, i_e, None))

    def save_npy(self, path: str) -> None:
        """
        Save light curve object as .npy file using pickle.

        Parameters
        ----------
        path : str
            Destination file path.

        Notes
        -----
        Use `load_lc_npy()` to read this file.
        This does not update `LC.py`, it saves current object state.
        """
        with open(path, "wb") as pickle_file:
            pickle.dump(self, pickle_file)

    def save_csv(self, path: str, bblocks: bool = False) -> None:
        """
        Save light curve data to CSV format.

        Parameters
        ----------
        path : str
            Output file path.
        bblocks : bool, optional
            If True, include `block_pbin` in the export.

        Notes
        -----
        Use `load_lc_csv()` to read this file and LC.py will be current state.
        """
        if bblocks is True:
            data = np.array([self.time, self.flux, self.flux_error, self.block_pbin])
            np.savetxt(path, data, comments="#time, flux, flux_error, block_pbin")
        else:
            data = np.array([self.time, self.flux, self.flux_error])
            np.savetxt(path, data, comments="#time, flux, flux_error")

    def plot_lc(
        self,
        data_color: str = "k",
        ax: Axes | None = None,
        new_time_format: str | None = None,
        size: float = 1,
        **kwargs,
    ) -> None:
        """
        Plot the light curve with error bars.

        Parameters
        ----------
        data_color : str, optional
            Color of the data points and error bars (default is 'k' for black).
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis to plot on. If None, uses the current axis.
        new_time_format : str, optional
            If set, adds a top x-axis with time converted to this format.
            Corresponding to `astropy.time.Time` (eg, 'isot', 'decimalyear')
        size : float, optional
            Scaling factor for marker size (default is 1).
        **kwargs
            Additional keyword arguments passed to `ax.errorbar`.

        Returns
        -------
        None

        Notes
        -----
        Initial `self.time_format` must be defined to use `new_time_format`.

        Examples
        --------
        >>> lc.plot_lc(data_color='hotpink', size=2)
        >>> lc.plot_lc(new_time_format='decimalyear')
        """
        if ax is None:
            ax = plt.gca()
        ax.errorbar(
            x=self.time,
            y=self.flux,
            yerr=self.flux_error,
            ecolor=data_color,
            elinewidth=1,
            linewidth=0,
            marker="+",
            markersize=3 * size,
            color=data_color,
            **kwargs,
        )
        if self.time_format and new_time_format is not None:
            axtop = ax.twiny()
            axtop.set_xticks(ax.get_xticks())
            axtop.set_xbound(ax.get_xbound())
            axtop.set_xlim(ax.get_xlim())
            format_labels = astropy.time.Time(
                [t for t in ax.get_xticks()], format=self.time_format
            )
            if new_time_format == "isot":
                new_labels = [
                    format_labels.to_value(format="isot")[i].split("T")[0]
                    for i in range(len(format_labels))
                ]
                axtop.set_xticklabels(new_labels)  # = yyyy-mm-dd
            elif new_time_format == "decimalyear":
                new_labels = format_labels.to_value(format="decimalyear")
                axtop.set_xticklabels(np.round(new_labels, 1))  # = yyyy.y
            else:
                new_labels = format_labels.to_value(format=new_time_format)
                axtop.set_xticklabels(new_labels)
            plt.sca(ax)  # go back to initial bottom axis

    def plot_hline(self, value: float, ax: Axes | None = None, **kwargs) -> None:
        """
        Plot a horizontal line across the light curve plot.

        Parameters
        ----------
        value : float
            Y-value where the horizontal line is drawn.
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. If None, uses current axis.
        **kwargs
            Additional keyword arguments passed to `ax.hlines`.

        Returns
        -------
        None
        """
        if ax is None:
            ax = plt.gca()
        ax.hlines(value, xmin=min(self.time), xmax=max(self.time), **kwargs)

    def plot_vline(self, value: float, ax: Axes | None = None, **kwargs) -> None:
        """
        Plot a vertical line across the light curve plot.

        Parameters
        ----------
        value : float
            X-value where the vertical line is drawn.
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. If None, uses current axis.
        **kwargs
            Additional keyword arguments passed to `ax.vlines`.

        Returns
        -------
        None
        """
        if ax is None:
            ax = plt.gca()
        ax.vlines(value, ymin=min(self.flux), ymax=max(self.flux), **kwargs)

    def plot_shade(
        self, start_time: float, end_time: float, ax: Axes | None = None, **kwargs
    ) -> None:
        """
        Plot a shaded region between two time points.

        Parameters
        ----------
        start_time : float
            Start of the shaded region.
        end_time : float
            End of the shaded region.
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. If None, uses current axis.
        **kwargs
            Additional keyword arguments passed to `ax.fill_between`.

        Returns
        -------
        None
        """
        if ax is None:
            ax = plt.gca()
        x = np.linspace(start_time, end_time)
        y = np.ones(len(x)) * np.max(self.flux)
        y1 = np.ones(len(x)) * np.min(self.flux)
        ax.fill_between(x, y, y1, step="mid", alpha=0.2, zorder=0, **kwargs)

    def plot_grid(self, spacing: float = 10, ax: Axes | None = None, **kwargs) -> None:
        """
        Add a minor grid to the time axis at specified spacing.

        Parameters
        ----------
        spacing : float, optional
            Separation between vertical grid lines in time units.
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. If None, uses current axis.
        **kwargs : dict
            Additional keyword arguments passed to `ax.grid`.

        Returns
        -------
        None
        """
        if ax is None:
            ax = plt.gca()
        rounded_start = np.round(np.min(self.time) / spacing) * spacing
        rounded_end = np.round(np.max(self.time) / spacing) * spacing
        ax.set_xticks(np.arange(rounded_start, rounded_end, spacing), minor=True)
        ax.grid(which="minor", **kwargs)

    # -------------------------------------------------------------------------
    def get_bblocks(
        self, gamma_value: float | None = None, p0_value: float | None = 0.05
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply the Bayesian Blocks algorithm to segment the light curve
        based on statistically significant change points.

        This method uses `astropy.stats.bayesian_blocks` with the "measures"
        fitness function, assuming Gaussian-distributed flux errors.
        Block values are computed as the mean flux in each segment.
        Corresponding uncertainties estimated with Gaussian error propagation.

        Parameters
        ----------
        gamma_value : float, optional
            Regularization parameter controlling the number of blocks.
            Higher values yield fewer blocks.
            `gamma_value` overwrites `p0_value`.
        p0_value : float, optional
            False alarm probability, used to control the sensitivity of change
            point detection. Default is 0.05. Calibrated with white noise.

        Returns
        -------
        block_pbin : np.ndarray
            Block-average flux value for each time bin (same shape as `flux`).
        block_val : np.ndarray
            Mean flux value for each identified block.
        block_val_error : np.ndarray
            Propagated uncertainty for each block's flux value.
        edge_index : np.ndarray
            Indices in the time array marking the block edges.
        edges : np.ndarray
            Time values of the detected block edges.
            #TBD: If I remember correctly, the edge time/index marks the start.
            #     This is relevant for computing bb_i and making Hopjects.

        Notes
        -----
        - See: Scargle et al. 2013, arXiv:1304.2818
        - See: Jupyter Notebook on GitHub for illustration.
        - For constant light curves, a single block is returned.
        """
        # get Bayesian block edges for light curve
        self.edges = bblocks(
            t=self.time,
            x=self.flux,
            sigma=self.flux_error,
            fitness="measures",
            gamma=gamma_value,
            p0=p0_value,
        )
        logging.debug("got edges for light curve")

        if len(self.edges) <= 2:
            logging.warning("light curve is constant; only one bayesian block found.")
            self.block_pbin = np.ones(len(self.flux)) * np.mean(self.flux)
            self.block_val = np.array([np.mean(self.flux)])
            self.block_val_error = np.array([np.std(self.flux)])
            self.edge_index = np.array([0, -1])
            self.edges = np.array([self.time[0], self.time[-1]])
            return (
                self.block_pbin,
                self.block_val,
                self.block_val_error,
                self.edge_index,
                self.edges,
            )

        # get edge_index
        self.edge_index = np.array(
            [
                np.where(self.time >= self.edges[i])[0][0]
                for i, _ in enumerate(self.edges)
            ]
        )
        # change last entry such that loop over [j:j+1] gives all blocks
        self.edge_index[-1] += 1

        # determine flux values (mean) and errors (Gaussian propagation)
        self.block_val = np.zeros(len(self.edge_index) - 1)
        self.block_val_error = np.zeros(len(self.edge_index) - 1)
        for j in range(len(self.edge_index) - 1):
            start = self.edge_index[j]
            end = self.edge_index[j + 1]
            self.block_val[j] = np.mean(self.flux[start:end])
            self.block_val_error[j] = np.sqrt(
                np.sum(self.flux_error[start:end] ** 2)
            ) / (end - start)

        # create block-per-bin array corresponding to flux
        self.block_pbin = np.zeros(len(self.flux))
        for k, _ in enumerate(self.block_val):
            self.block_pbin[self.edge_index[k] : self.edge_index[k + 1]] = (
                self.block_val[k]
            )
        logging.debug("got block parameters for light curve")

        return (
            self.block_pbin,
            self.block_val,
            self.block_val_error,
            self.edge_index,
            self.edges,
        )

    # -------------------------------------------------------------------------
    def get_bblocks_above(
        self, threshold: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Modify Bayesian blocks by replacing all block values below a threshold
        with the threshold, and merging consecutive blocks (neglecting edges)
        below this threshold.

        Use Case
        --------
        Highlight significant variations above the threshold (e.g., flares with
        respect to a certain baseline), or in other words discard significant
        variations below the threshold.

        WARNING
        -------
        Returns artificial block values and unreliable error estimates!
        Use with caution.

        Raises
        ------
        AttributeError
            If `block_pbin` is not initialized. Run `.get_bblocks()` first.

        Parameters
        ----------
        threshold : float
            Minimum block value to retain. All block values below this will be
            artificially raised to the threshold.

        Returns
        -------
        block_pbin : np.ndarray
            Modified block values per time bin (same shape as `flux`).
        block_val : np.ndarray
            Modified block values.
        block_val_error : np.ndarray
            Unchanged error estimates (inconsistent, TBD: thresholderror)
        edge_index : np.ndarray
            Indices in the time array marking the updated block edges.
        edges : np.ndarray
            Time values corresponding to the updated block edges.

        Notes
        -----
        - `lc.get_bblocks()` must be run before calling this function.
        - Errors are not recomputed after thresholding. TBD: thresholderror.

        """
        # Replace all values below threshold
        try:
            self.block_pbin = np.where(
                self.block_pbin > threshold, self.block_pbin, threshold
            )
            self.block_val = np.where(
                self.block_val > threshold, self.block_val, threshold
            )
        except AttributeError:
            raise AttributeError(
                "Initialize Bayesian blocks with lc.get_bblocks() first!"
            )

        # Merge neighbouring threshold blocks and delete edges
        block_mask = np.ones(len(self.block_val), dtype=bool)
        edge_mask = np.ones(len(self.edges), dtype=bool)
        for i in range(len(self.block_val) - 1):
            if self.block_val[i] == threshold and self.block_val[i + 1] == threshold:
                block_mask[i + 1] = False
                edge_mask[i + 1] = False

        self.block_val = self.block_val[block_mask]
        self.block_val_error = self.block_val_error[block_mask]
        # TBD -> thresholderror
        self.edge_index = self.edge_index[edge_mask]
        self.edges = self.edges[edge_mask]
        return (
            self.block_pbin,
            self.block_val,
            self.block_val_error,
            self.edge_index,
            self.edges,
        )

    # -------------------------------------------------------------------------
    def plot_bblocks(
        self,
        ax: Axes | None = None,
        color: str = "steelblue",
        linewidth: float = 1,
        **kwargs,
    ) -> None:
        """
        Plot the Bayesian block representation of the light curve.

        This creates a step plot of `block_pbin` over `time`, which represents
        the segmented light curve after applying `get_bblocks()`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object to plot on. If None, uses current axes.
        color : str, default "steelblue"
            Color of the step plot.
        linewidth : float, default 1
            Width of the step lines.
        **kwargs : dict
            Additional keyword arguments passed to `ax.step`.

        Raises
        ------
        AttributeError
            If `block_pbin` is not initialized. Run `.get_bblocks()` first.

        Notes
        -----
        - Uses `where="mid"` to visualize steps centered between time bins.
          But remember actual block edges correspond to time values (I think)
        - Make sure to call `.get_bblocks()` before using this function.
        """
        if ax is None:
            ax = plt.gca()
        try:
            ax.step(
                self.time,
                self.block_pbin,
                where="mid",
                linewidth=linewidth,
                color=color,
                **kwargs,
            )
        except AttributeError:
            raise AttributeError(
                "Initialize Bayesian blocks with .get_bblocks() first!"
            )

    # -------------------------------------------------------------------------
    def bb_i(self, t: float):
        """
        Get the index of the Bayesian block (in `block_val`) that contains the
        given time.

        If `t == edge`, return the block *left* of the edge (t = end of block).

        This correctly identifies times from HOP 'baseline'.
        Identify times from other HOP methods with `bb_i_start` and `bb_i_end`!

        Parameters
        ----------
        t : float
            Time value to locate within the Bayesian block segmentation.

        Returns
        -------
        int
            Index of `lc.block_val` containing `t`.

        Notes
        -----
        - Generally, Bayesian block edges mark the start of a new segment.
        - However, in this function, if: `t == edge`
          t is interpreted to belong to the block to the *left* of the edge
          i.e. as if t were the end of the block
          to correctly identify start and end in HOP 'baseline' .. I think..
        - Use `bb_i_start` and `bb_i_end` for other HOP methods
        """
        if t == self.edges[0]:
            return 0
        block_index = [
            e
            for e in range(len(self.edges) - 1)
            if t > self.edges[e] and t <= self.edges[e + 1]
        ]
        return int(block_index[0])

    def bb_i_start(self, t: float):
        """
        Get the index of the Bayesian block (in `block_val`) that starts with
        given time.

        If `t == edge`, return the block *right* of edge (t = start of block).

        This correctly identifies times from HOP 'flip', 'halfclap', 'sharp'
        To identify times from HOP 'baseline' use `bb_i`!

        Parameters
        ----------
        t : float
            Time value to locate within the Bayesian block segmentation.

        Returns
        -------
        int
            Index of `lc.block_val` starting with `t`.

        Notes
        -----
        - Works fine with HOP 'flip', 'halfclap', and 'sharp'
          but not sure for HOP 'baseline' -> use bb_i() instead
        """
        block_index = [
            e
            for e in range(len(self.edges) - 1)
            if t >= self.edges[e] and t < self.edges[e + 1]
        ]
        return int(block_index[0])

    def bb_i_end(self, t: float):
        """
        Get the index of the Bayesian block (in `block_val`) that ends with
        given time.

        If `t == edge`, return the block *left* of edge (t = end of block).

        This correctly identifies times from HOP 'flip', 'halfclap', 'sharp'
        To identify times from HOP 'baseline' use `bb_i`!

        Parameters
        ----------
        t : float
            Time value to locate within the Bayesian block segmentation.

        Returns
        -------
        int
            Index of `lc.block_val` starting with `t`.

        Notes
        -----
        - Works fine with HOP 'flip', 'halfclap', and 'sharp'
          but unsure for HOP 'baseline' -> use bb_i() instead
        """
        block_index = [
            e
            for e in range(len(self.edges) - 1)
            if t > self.edges[e] and t <= self.edges[e + 1]
        ]
        return int(block_index[0])

    # -------------------------------------------------------------------------
    def find_hop(
        self,
        method: str = "half",
        lc_edges: str = "neglect",
        baseline: float | None = None,
    ) -> list[Hopject]:  # this type is not known in LC.py yet
        """
        TBD
        currently this is kind of an implicit circular renference:
            * LC has List of HOPS
            * and HOP as LC as attribute
        fix such that:
            * LC imports Hopjects
            * find_hop determines HOP parameters and is called by..
            * dedicated get_hop function (here) which returns Hopjects with
                * time, flux, flux_error, iis, bblock things
                * and iis & bblock things are optional in creating a Hopject
        so spirit will be that HOP is not so standalone, just add-on to LC
        TBD
        """
        if method == "baseline":
            if baseline is None:
                self.baseline = np.mean(self.flux)
            else:
                self.baseline = baseline
            hopfinder = HopFinderBaseline(lc_edges)
        if method == "half":
            hopfinder = HopFinderHalf(lc_edges)
        if method == "sharp":
            hopfinder = HopFinderSharp(lc_edges)
        if method == "flip":
            hopfinder = HopFinderFlip(lc_edges)
        self.hops = hopfinder.find(self)
        return self.hops

    # -------------------------------------------------------------------------
    def plot_hop(
        self,
        ax: Axes | None = None,
        color: str | list[str] = ["lightsalmon", "orchid"],
        alpha: float = 0.2,
        label: str | None = None,
        zorder: float = 0,
        **kwargs,
    ) -> None:
        """
        Plot shaded rectangular area for each HOP group in the light curve.

        To visualize HOP segments (e.g. flares) by highlighting them as
        rectangular area in the light curve. Multiple colors can be used to
        alternate between consecutive segments for clarity.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object to plot on. If None, uses current axes.
        color : str or list of str, optional
            Color(s) used to shade the HOP areas. If a list is provided, colors
            will be cycled through to visually distinguish adjacent HOPs.
            Default is `["lightsalmon", "orchid"]`.
        alpha : float, optional
            Transparency level (0-1) for the shaded area. Default is 0.2.
        label : str, optional
            Label for the first HOP area, used in the plot legend. If None,
            defaults to `"hop <method>"`.
        zorder : float, optional
            Z-order for the plot layer. Default is 0 (in the background).
        **kwargs
            Additional keyword arguments passed to `ax.fill_between()`.

        Returns
        -------
        None
        """
        if self.hops is None:
            return  # no hop in this lc
        if ax is None:
            ax = plt.gca()
        if label is None:
            label = str(self.hops[0].method)

        # Make color iterable
        if isinstance(color, str):
            color = [color]

        for i, hop in enumerate(self.hops):
            x = np.linspace(hop.start_time, hop.end_time)
            y = np.ones(len(x)) * np.max(self.flux)
            y1 = np.min(self.flux)
            c = color[i % len(color)]

            ax.fill_between(
                x,
                y,
                y1,
                step="mid",
                color=c,
                alpha=alpha,
                label=label if i == 0 else None,
                zorder=zorder,
                **kwargs,
            )

    # -------------------------------------------------------------------------
    def plot_all_hop(self) -> None:
        """
        Execute and plot all HOP detection methods for visual comparison.

        Runs all HOP algorithm methods ('baseline', 'half', 'flip', 'sharp').
        Plots resulting segments on stacked subplots for sidebyside comparison.

        This is a lil old fashioned and could be improved but it does the job.
        """
        fig = plt.figure(0, (15, 9))
        plt.suptitle("All HOP methods", fontsize=16)

        ax0 = fig.add_subplot(511)
        self.find_hop("baseline")
        self.plot_bblocks()
        self.plot_lc()
        self.plot_hop()
        plt.ylabel("baseline")

        ax1 = fig.add_subplot(512)
        self.find_hop("half")
        self.plot_bblocks()
        self.plot_lc()
        self.plot_hop()
        plt.ylabel("half")

        ax2 = fig.add_subplot(513)
        self.find_hop("flip")
        self.plot_bblocks()
        self.plot_lc()
        self.plot_hop()
        plt.ylabel("flip")

        ax3 = fig.add_subplot(514)
        self.find_hop("sharp")
        self.plot_bblocks()
        self.plot_lc()
        self.plot_hop()
        plt.ylabel("sharp")
        fig.subplots_adjust(hspace=0)
