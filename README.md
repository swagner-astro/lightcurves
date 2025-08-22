# lightcurves

This is the lightcurves repository. Check it out: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/swagner-astro/lightcurves/blob/main/illustration_lightcurve.ipynb) <br>

See here for scientific application of this code:
https://pos.sissa.it/395/868

## lc.py
Initialize a LightCurve object based on time, flux and flux_error.
Study its Bayesian block representation (based on Scargle et al. 2013  https://ui.adsabs.harvard.edu/abs/2013arXiv1304.2818S/abstract ).<br>
Characterize flares (start, peak, end time) with the HOP algorithm (following Meyer et al. 2019 https://ui.adsabs.harvard.edu/abs/2019ApJ...877...39M/abstract ). There are four different methods to define flares (baseline, half, flip, sharp) as illustrated in the Jupyter Notebook.

## hop.py
Initialize a Hopject to consider parameters of an individual flare.

## lc_set
Initialize a (large) sample of light curves to study the distribution of flare parameters whithin that sample.<br>


## Reference
If you use this code please cite: <br>
Wagner, S. M., Burd, P., Dorner, D., et al. 2021, PoS, ICRC2021, 868
<url>https://ui.adsabs.harvard.edu/abs/2022icrc.confE.868W/abstract</url>
