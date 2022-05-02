import numpy as np
from lmfit import Model #https://lmfit.github.io/lmfit-py/
from matplotlib import pyplot as plt

class Hopject: 
    '''
    HOP Class
    ---------
    Segment in a light curve, i.e. a group of Bayesian blocks, that represent a flare ( HOP group)
    For definition of start, peak and end of the flare check out the LightCurve class
    '''
    def __init__(self, hop_params, lc, method=None): #e.g. Hopject(lc.get_hop_method[0], lc)
        self.start_time, self.peak_time, self.end_time = hop_params        
        self.lc = lc
        self.z = lc.z
        self.name = lc.name
        self.telescope = lc.telescope
        self.method = method #e.g. half or flip
        self.iis = np.where(np.logical_and(lc.time > self.start_time, 
        								   lc.time < self.end_time))
        self.time = lc.time[self.iis]
        self.flux = lc.flux[self.iis]
        self.flux_error = lc.flux_error[self.iis]
        self.n_bins = len(self.time)
        self.coverage = self.n_bins / (self.end_time - self.start_time)
        self.n_blocks = lc.bb_i_end(self.end_time) - lc.bb_i_start(self.start_time) 
        # e.g. one-block hop: 5 - 3 = 2 

        self.dur = self.end_time - self.start_time
        self.rise_time = self.peak_time - self.start_time
        self.decay_time = self. end_time - self. peak_time
        self.asym = (self.rise_time - self.decay_time)/(self.rise_time + self.decay_time)

        self.start_flux = lc.block_val[lc.bb_i_start(self.start_time)]
        self.peak_flux = lc.block_val[lc.bb_i(self.peak_time)]
        self.end_flux = lc.block_val[lc.bb_i_end(self.end_time)]
        self.rise_flux = self.peak_flux - self.start_flux
        self.decay_flux = self.peak_flux - self.end_flux

    #----------------------------------------------------------------------------------------------
    def plot_hop(self, ax=None):
        """
        Plot the snip of light curve with this flare
        """
        if ax is None:
            ax = plt.gca()
        self.lc.plot_bblocks(ax=ax)
        ax.set_xlim(self.start_time - self.dur/2, self.end_time + self.dur/2)
        x = np.linspace(self.start_time, self.end_time)
        y = np.ones(len(x)) * self.peak_flux
        y1 = np.zeros(len(x))
        ax.fill_between(x, y, y1, step="mid", color='lightsalmon', alpha=0.2, label='hop', zorder=0)

    #----------------------------------------------------------------------------------------------
    def exp_rd(self, t, amp, t_0, t_r, t_d):
        """
        exponential rise and exponential decay
        with flux = amp/2 at t=t_0 (peak could be shifted due to ratio of t_r and t_d)
        $F(x) = amp * ( exp(\frac{t-cen}{t_{decay}}) + exp(\frac{-t+cen}{t_{rise}}) )^{-1}$
        """
        return(amp * ( np.exp((t-t_0)/t_d) + np.exp((-t+t_0)/t_r))**-1)

    def get_exp_fit(self):
        """exponential fit on data points inside hop"""
        exp_model = Model(self.exp_rd)
        params = exp_model.make_params(amp=self.peak_flux, 
                                       t_0=self.peak_time, 
                                       t_r=self.rise_time,
                                       t_d=self.decay_time)
        result = exp_model.fit(self.flux, params, t=self.time, weights=1/self.flux_error)
        self.exp_tr = result.params['t_r'].value
        self.exp_td = result.params['t_d'].value
        self.exp_amp = result.params['amp'].value
        self.exp_t0 = result.params['t_0'].value
        self.exp_chisqr = result.chisqr # $\chi^2 = \sum_i \frac{(O-C)^2}{\sigma_i^2}$
        self.exp_redchi = result.redchi # $\chi^2_R = \sum_i \frac{(O-C)^2}{\sigma_i^2}$ 

        return result

    def plot_exp_fit(self, plotpoints=200, ax=None, **plot_kwargs):
        self.get_exp_fit()
        x_plot = np.linspace(self.start_time, self.end_time, plotpoints)
        y_plot = self.exp_rd(t=x_plot, amp=self.exp_amp, t_0=self.exp_t0, 
                             t_r=self.exp_tr, t_d=self.exp_td)
        if ax is None:
            ax = plt.gca()
        ax.plot(x_plot,y_plot, marker='', zorder=13148, **plot_kwargs)

    #----------------------------------------------------------------------------------------------
    def gaussian(self, x, amp, cen, wid):
        """1-d gaussian: gaussian(x=data, amp, cen, wid)"""
        return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

    def get_gauss_fit(self):
        """gaussian fit on data points inside hop"""
        x = self.time
        y = self.flux
        y_error = self.flux_error
        gmodel = Model(self.gaussian)
        params = gmodel.make_params(cen=np.mean(x), 
                                    amp=self.peak_flux, 
                                    wid=np.std(x))
        result = gmodel.fit(y, params, x=x, weights=y_error)#Poisson error: 1/np.where(y == 0, 1, y))
        self.gauss_amp = result.params['amp'].value
        self.gauss_mu = result.params['cen'].value  
        self.gauss_sigma = result.params['wid'].value
        self.gauss_chisqr = result.chisqr
        self.gauss_redchi = result.redchi
        return(result)

    def plot_gauss_fit(self, plotpoints=200, ax=None, **plot_kwargs):
        self.get_gauss_fit()
        x_plot = np.linspace(self.start_time, self.end_time, plotpoints)
        y_plot = self.gaussian(x_plot, self.gauss_amp, self.gauss_mu, self.gauss_sigma)
        if ax is None:
            ax = plt.gca()
        ax.plot(x_plot,y_plot, marker='', zorder=13148, **plot_kwargs)
        return()

