import lsqfit
import numpy as np
import gvar as gv
import sys
import os
# local modules 
# import non_analytic_functions as naf
from i_o import InputOutput

class FpiFit:

    def __init__(self, prior, data, phys_pt_data, model_info):
         
        self.prior = prior
        self.data = data
        self.model_info = model_info.copy()
        self._fit = None
        self._extrapolate = None
        self._simultaneous = False
        self._posterior = None
        self.phys_pt_data = phys_pt_data
        
        self.y = {'Fpi' : gv.gvar([gv.mean(g) for g in data['Fpi']], [gv.sdev(g) for g in data['Fpi']])}
        #self.y = {'Fpi': gv.dataset.avg_data(data['Fpi']) }
       
    def __str__(self):
        return str(self.fit)
    
    @property
    def fit(self):
        if self._fit is None:
            models = self._make_models()
            prior = self._make_prior()
            data = self.y
            fitter = lsqfit.MultiFitter(models=models)
            fit = fitter.lsqfit(data=data, prior=prior, fast=False, mopt=False)
            self._fit = fit

        return self._fit 

    def _make_models(self, model_info=None):
        if model_info is None:
            model_info = self.model_info.copy()

        models = np.array([])

        if 'Fpi' in model_info['particles']:
            models = np.append(models,Fpi(datatag='Fpi', model_info=model_info))
        return models

    def _make_prior(self, data=None):
        '''
        Only need priors for LECs/data needed in fit
        '''
        if data is None:
            data = self.data
        prior = self.prior
        new_prior = {}
        particles = []
        particles.extend(self.model_info['particles'])

        keys = []
        orders = []
        for p in particles:
            for l, value in [('light',self.model_info['order_light']), ('disc', self.model_info['order_disc']),
            ('strange', self.model_info['order_strange']), ('xpt', self.model_info['order_chiral'])]:
            # include all orders equal to and less than desired order in expansion #
                if value == 'lo':
                    orders = ['lo']
                elif value == 'n2lo':
                    orders = ['lo','n2lo']
                # elif value == 'n4lo':
                #     orders = ['llo', 'lo', 'nlo','n2lo', 'n4lo']
                else:
                    orders = []
                for o in orders:
                    keys.extend(self._get_prior_keys(particle=p, order=o, lec_type = l))
                
        for k in keys:
            new_prior[k] = prior[k]

        if self.model_info['order_strange'] is not None:
            new_prior['m_k'] = data['m_k']
            #new_prior['eps_pi'] = data['eps_pi']
        if self.model_info['order_light'] is not None:
            new_prior['eps_pi'] = data['eps_pi']
            #new_prior['eps2_a'] = data['eps2_a']
            #new_prior['m_pi'] = data['m_pi']
            #new_prior['lam_chi'] = data['lam_chi']
        if self.model_info['order_disc'] is not None:
            new_prior['eps2_a'] = data['eps2_a']
            #new_prior['m_pi'] = data['m_pi']
            #new_prior['lam_chi'] = data['lam_chi']

        for key in ['eps_pi']:
            new_prior[key] = data[key]
        # for key in ['eps2_a','lam_chi','m_pi','eps_pi']: 
        #     new_prior[key] = data[key]
        return new_prior

    def _get_prior_keys(self, particle = 'all', order='all',lec_type='all'):
        if particle == 'all':
            output = []
            for particle in ['Fpi']:
                keys = self._get_prior_keys(particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        elif order == 'all':
            output = []
            for order in ['llo', 'lo', 'nlo','n2lo']:
                keys = self._get_prior_keys(particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        elif lec_type == 'all':
            output = []
            for lec_type in ['disc','light','strange','xpt']:
                keys = self._get_prior_keys(particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        else: 
            # construct dict of lec names corresponding to particle, order, lec_type #
            output = {}
            for p in ['Fpi']:
                output[p] = {}
                for o in ['llo', 'lo', 'nlo','n2lo']:
                    output[p][o] = {}
            
            output['Fpi']['lo']['light']    = ['F0','l4_bar']
            output['Fpi']['lo']['disc']     = ['d_{fpi,a}']
            output['Fpi']['lo']['xpt']      = ['F0','l4_bar']
            output['Fpi']['n2lo']['light']  = ['F0','d_{fpi,ll}','c_2F']
            output['Fpi']['n2lo']['disc']   = ['d_{fpi,aa}','d_{fpi,al}']
            output['Fpi']['n2lo']['xpt']    = ['c_1F','l4_bar']

            if lec_type in output[particle][order]:
                return output[particle][order][lec_type]
            else:
                return []

class Fpi(lsqfit.MultiFitterModel):
    def __init__(self, datatag, model_info):
        super().__init__(datatag)
        self.model_info = model_info
        self.phys = InputOutput().get_data_phys_point()
    
    def fitfcn(self, p,xdata=None):
        xdata = {}
        #xdata['lam_chi'] = p['lam_chi']
        if self.model_info['fit_phys_units']:
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        elif self.model_info['fit_fpi_units']:
            xdata['eps_pi'] = p['eps_pi']
        if self.model_info['order_disc'] is not None:
            xdata['eps2_a'] = p['eps2_a']

        # USING BARYON POWER COUNTING FOR ORDERING #
        #output = p['F0']
        output = p['F0'] *(
            + 1
        + self.fitfcn_lo(p,xdata) 
        + self.fitfcn_lo_xpt(p,xdata)
        + self.fitfcn_n2lo(p,xdata)
        + self.fitfcn_n2lo_xpt(p,xdata)
        )
        
        return output
    # def fitfcn_llo(self, p,xdata):
    #     output = 0
    #     if self.model_info['order_light'] in ['llo','lo','n2lo']:
    #         output += p['F0']
    #     return output

    def fitfcn_lo(self,p,xdata):
        output = 0
        if self.model_info['order_light'] in ['lo','n2lo']:
            output += p['F0'] *xdata['eps_pi']**2 * (p['l4_bar'] - 4.3)
    
        if self.model_info['order_disc'] in ['lo','n2lo']:
            output += p['F0'] * (p['d_{fpi,a}'] * xdata['eps2_a'])
        return output

    def fitfcn_lo_xpt(self, p,xdata):
        output = 0
        if self.model_info['order_chiral'] in ['lo','n2lo']:
            #output +=  p['F0']*(-np.log(xdata['eps_pi']**2)
            output +=  -xdata['eps_pi']**2*np.log(xdata['eps_pi']**2)

        return output

    def fitfcn_n2lo_xpt(self,p,xdata):
        output = 0
        if self.model_info['order_chiral'] in ['n2lo']:
            output += p['F0']* (5/4*(xdata['eps_pi']**4*np.log(xdata['eps_pi']**2)**2 +(p['c_1F'] + 2)*np.log(xdata['eps_pi']**2) ))
            #output += p['F0']* (np.log(xdata['eps_pi']**2)**2 +(p['c_1F'] + 2)*np.log(xdata['eps_pi']**2) )
             
        return output

    def fitfcn_n2lo(self,p,xdata):
        output = 0
        if self.model_info['order_light'] in ['n2lo']:
            output += xdata['eps_pi']**4*(p['c_2F']-2*p['l4_bar'])

        if self.model_info['order_disc'] in ['n2lo']:
            output += p['F0']*( 
            p['d_{fpi,al}'] * xdata['eps2_a'] * xdata['eps_pi']**4 +p['d_{fpi,aa}'] * xdata['eps2_a']**2)
            

        return output

    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]
