# lightcurves

This is the lightcurve repository. 

## LC.py
Initialize a LightCurve object based on time, flux and flux_error. 
Study its Bayesian block representation (based on Scargle et al. 2013  https://ui.adsabs.harvard.edu/abs/2013arXiv1304.2818S/abstract ).<br>
Characterize flares (start, peak, end time) with the HOP algorithm (following Meyer et al. 2019 https://ui.adsabs.harvard.edu/abs/2019ApJ...877...39M/abstract ). There are four different methods to define flares (baseline, half, flip, sharp) as illustrated in the Jupyter Notebook. 

## HOP.py
Initialize a Hopject to consider parameters of an individual flare.

## LC_Set
Initialize a (large) sample of light curves to study the distribution of flare parameters whithin that sample.<br>


If you use this code please cite:
S.M. Wagner et al., accepted in Proceedings of Science, ICRC 2021
