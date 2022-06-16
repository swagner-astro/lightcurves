# lightcurves

This is the lightcurve repository. Check it out: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OqafFK4FQA_tBwTTnYMG-1D5uhTQ5X0D#scrollTo=european-mechanism) <br>
See here for scientific application of this code:
https://pos.sissa.it/395/868 

## LC.py
Initialize a LightCurve object based on time, flux and flux_error. 
Study its Bayesian block representation (based on Scargle et al. 2013  https://ui.adsabs.harvard.edu/abs/2013arXiv1304.2818S/abstract ).<br>
Characterize flares (start, peak, end time) with the HOP algorithm (following Meyer et al. 2019 https://ui.adsabs.harvard.edu/abs/2019ApJ...877...39M/abstract ). There are four different methods to define flares (baseline, half, flip, sharp) as illustrated in the Jupyter Notebook. 

## HOP.py
Initialize a Hopject to consider parameters of an individual flare.

## LC_Set
Initialize a (large) sample of light curves to study the distribution of flare parameters whithin that sample.<br>






If you use this code please cite: <br>
@INPROCEEDINGS{2022icrc.confE.868W,
       author = {{Wagner}, S.~M. and {Burd}, P. and {Dorner}, D. and {Mannheim}, K. and {Buson}, S. and {Gokus}, A. and {Madejski}, G. and {Scargle}, J. and {Arbet-Engels}, A. and {Baack}, D. and {Balbo}, M. and {Biland}, A. and {Bretz}, T. and {Buss}, J. and {Elsaesser}, D. and {Eisenberger}, L. and {Hildebrand}, D. and {Iotov}, R. and {Kalenski}, A. and {Neise}, D. and {Noethe}, M. and {Paravac}, A. and {Rhode}, W. and {Schleicher}, B. and {Sliusar}, V. and {Walter}, R.},
        title = "{Statistical properties of flux variations in blazar light curves at GeV and TeV energies}",
     keywords = {Astrophysics - High Energy Astrophysical Phenomena},
    booktitle = {37th International Cosmic Ray Conference. 12-23 July 2021. Berlin},
         year = 2022,
        month = mar,
          eid = {868},
        pages = {868},
archivePrefix = {arXiv},
       eprint = {2110.14797},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022icrc.confE.868W},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
