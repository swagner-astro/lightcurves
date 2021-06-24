# lightcurves

This is the lightcurve repository. 

## LC.py
Initialize a LightCurve object based on time, flux and flux_error. 
Study its Bayesian block representation based on Scargle et al. (2013)  https://ui.adsabs.harvard.edu/abs/2013arXiv1304.2818S/abstract
Characterize flares (start, peak, end time) with the HOP algorithm following Meyer et al. (2019) https://ui.adsabs.harvard.edu/abs/2019ApJ...877...39M/abstract 
There are four different methods to define flares (baseline, half, flip, sharp) as illustrated in the Jupyter Notebook with artificial light curves. 

## HOP.py
Initialize a Hopject to consider parameters of an individual flare.

## LC_Set
Initialize a (large) sample of light curves to study the distribution of flare parameters whithin that sample.

If you use this code please cite:
--> Proceedings of Science Article (ICRC; Wagner et al. 2021)
