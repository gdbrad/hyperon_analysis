import lsqfit
import gvar as gv 
import numpy as np
import functools
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import copy

# local modules
import non_analytic_functions as naf
import i_o as i_o
import get_models 

class FitCache(dict):
    def __missing__(self,key):
        self[key] = fit_model = self.load_fit(key)
        return fit_model
    
    def load_fit(self,key):
        if not hasattr(self,'_queue'):
            self._queue = []

class FitModel:
    """
    The `FitModel` class is designed to fit models to data using least squares fitting.
    It takes in the model information, and options for empirical Bayes analysis.
    Ideally, this module should be general enough to take information from a particular baryon's multifitter class as input. 

    Attributes:
       
        model_info (dict):  information about the model to be fit.
        empbayes (bool): A boolean indicating whether to perform empirical Bayes analysis.
        empbayes_grouping (list): A list of dictionaries containing information about how to group the data
                                  for the empirical Bayes analysis.
        _fit (tuple): A tuple containing the fit results. The first element is a dictionary of the fit
                      parameters, and the second element is a dictionary of the fit errors.
        _posterior (dict): information about the posterior distribution of the fit.
        _empbayes_fit (tuple):empirical Bayes fit results. The first element is the
                               empirical Bayes prior, and the second element is a dictionary of the fit
                               parameters.

    Methods:
        __init__(self, prior, data, model_info, empbayes, empbayes_grouping):
            Constructor method for the `FitRoutine` class.
        __str__(self):
            String representation of the fit.
        fit(self):
            Method to perform the fit. Returns the fit results as a tuple.
        _make_models(self):
            Method to create models.
        _make_prior(self):
            Method to create prior information.
    """

    def __init__(self,
                 data:dict,
                 prior:dict,
                 phys_pt_data:dict,
                 model_info:dict,
                 strange:str,
                 **kwargs
                ):
        self.data = data
        self.prior = prior
        if self.data is None:
            raise ValueError('you need to pass data to the fitter')
        self._phys_pt_data = phys_pt_data
        self.strange = strange
        # self._model_info = self.fetch_models()
        self.model_info = model_info
         
        self.options = kwargs
        # default values for optional params 
        self.svd_test = self.options.get('svd_test', False)
        self.svd_tol = self.options.get('svd_tol', None)
        self.mdl_key = self.options.get('mdl_key',None)

        if self.strange == '2':
            # self.observables = ['xi','xi_st']
            self.observables = ['xi_st']
        # xi_particles = ['xi','xi_st']
        self._posterior = None
        self.models, self.models_dict = self._make_models()
        if self.strange == '2':
            self.y = {part: self.data['m_'+part] for part in self.observables}

    def update_svd_tol(self,new_svd_tol):
        self.svd_tol = new_svd_tol
    
    def fetch_models(self,manual=False):
        models = {}
        if manual:
            with open('models_test.yaml', 'r') as f:
                _models = yaml.load(f, Loader=yaml.FullLoader)
            if self.strange == '2':
                models = _models['models']['xi']
            elif self.strange == '1':
                models = _models['models']['lam']    
            elif self.strange == '0':
                models = _models['models']['proton']
            return models
        else:
            models = get_models.GenerateModels(strange=self.strange)    

    # @functools.cached_property
    @property
    def fit(self):
        prior_final = self._make_prior()
        data = self.y
        fitter = lsqfit.MultiFitter(models=self.models)
        if self.svd_test:
            svd_cut = self.input_output.perform_svdcut()
            fit = fitter.lsqfit(data=data, prior=prior_final, fast=False, mopt=False,svdcut=svd_cut)
            # fig = plt.figure('svd_diagnosis', figsize=(7, 4))
            # for ens in self.ensembles:
            #     svd_test.plot_ratio(show=True)
        else:
            # if self.svd_tol is None:
            #     fit = fitter.lsqfit(data=data, prior=prior_final, fast=False, mopt=False)
            # else:
            
            fit = fitter.lsqfit(data=data, prior=prior_final, fast=False, mopt=False,svdcut=self.svd_tol)

        return fit
    
    @property
    def fit_info(self):
        fit_info = {}
        fit_info = {
            'name' : self.model_info['name'],
            'logGBF' : self.fit.logGBF,
            'chi2/df' : self.fit.chi2 / self.fit.dof,
            'Q' : self.fit.Q,
            'phys_point' : self.phys_point_data,
            # 'error_budget' : self.error_budget,
            'prior' : self.prior,
            'posterior' : self.posterior
        }
        return fit_info
    
    @property
    def get_posterior(self):
        return self._get_posterior()
    
    def _get_posterior(self,param=None):
        if param is not None:
            return self.fit.p[param]
        elif param == 'all':
            return self.fit.p
        output = {}
        for key in self.prior:
            if key in self.fit.p:
                output[key] = self.fit.p[key]
        return output

    
    def _make_models(self, model_info=None):
        if model_info is None:
            model_info = self.model_info.copy()

        model_array = np.array([])
        model_dict = {}

        if 'xi' in model_info['particles']:
            xi_model = Xi(datatag='xi', model_info=model_info)
            model_array = np.append(model_array, xi_model)
            model_dict['xi'] = xi_model

        if 'xi_st' in model_info['particles']:
            xi_st_model = Xi_st(datatag='xi_st', model_info=model_info)
            model_array = np.append(model_array, xi_st_model)
            model_dict['xi_st'] = xi_st_model

        return model_array, model_dict 
    
    def _make_prior(self, data=None,z=None,scale_data=None,verbose=False):
        """Only need priors for LECs/data needed in fit.
        Separates all parameters that appear in the hyperon extrapolation formulae 
        """
        if data is None:
            data = self.data
        prior = self.prior
        new_prior = {}
        particles = []
        particles.extend(self.model_info['particles'])

        keys = []
        orders = []
        for p in particles:
            for l, value in [('light', self.model_info['order_light']), 
                             ('disc', self.model_info['order_disc']),
                             ('strange', self.model_info['order_strange']), 
                             ('xpt', self.model_info['order_chiral'])]:
                # include all orders equal to and less than the desired order in the expansion #
                # llo: 0 , lo: 1, nlo: 2, n2lo:3 #
                # Note: the chiral order should not exceed that of the light order! #
                # If an order for a type is None, none of the corresponding keys will be included in fit #
                if value is not None:
                    if value == 0:
                        orders = [0]
                    elif value == 1:
                        orders = [0,1] 
                    elif value == 2:
                        orders = [0,1,2]
                    elif value == 3:
                        orders = [0,1,2,3]
                    else:
                        orders = []
                    for o in orders:
                        keys = self._get_prior_keys(particle=p, order=o, lec_type=l)
                        if verbose:
                            print(f"paricle={p}, order={o}, lec_type={l}, keys={keys}")
                        keys.extend(keys)
                        for key in keys:
                            new_prior[key] = prior[key]
        
        # this is "converting" the pseudoscalars into priors so that they do not count as data #
        if self.model_info['order_strange'] is not None:
            new_prior['m_k'] = data['m_k']
            # new_prior['a_fm'] = data['a_fm']

        if self.model_info['order_light'] is not None:
            new_prior['eps2_a'] = data['eps2_a']
            new_prior['m_pi'] = data['m_pi']
            new_prior['lam_chi'] = data['lam_chi']
            new_prior['eps_pi'] = data['eps_pi']
            new_prior['units'] = data['units']

        if self.model_info['order_disc'] is not None:
            new_prior['lam_chi'] = data['lam_chi']
            new_prior['m_pi'] = data['m_pi']
            # new_prior['a_fm'] = data['a_fm']


        # for empirical bayes fits
        if z is None:
            return new_prior
        zkeys = self._empbayes_grouping()

        for k in new_prior:
            for group in zkeys:
                if k in zkeys[group]:
                    new_prior[k] = gv.gvar(0, np.exp(z[group]))
        return new_prior

    def _get_prior_keys(self, particle='all', order='all', lec_type='all'):
        """construct the necessary prior keys which will be passed to the fit (xpt extrapolation formulae). 
        0: llo
        1: lo
        2: nlo
        3: n2lo
        """
        if particle == 'all':
            output = []
            for particle in self.observables:
                keys = self._get_prior_keys(
                    particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        elif order == 'all':
            output = []
            for order in [0,1,2,3]:
                keys = self._get_prior_keys(
                    particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        elif lec_type == 'all':
            output = []
            for lec_type in ['disc', 'light', 'strange', 'xpt']:
                keys = self._get_prior_keys(
                    particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        else:
        # construct dict of lec names corresponding to particle, order, lec_type #
            lec_key = f'{lec_type}'
            output = {}
            for p in ['proton', 'delta', 'lambda', 'sigma', 'sigma_st', 'xi', 'xi_st', 'omega']:
                output[p] = {}
                for o in [0,1,2,3]:
                    output[p][o] = {}

            output['xi'][0]['light'] = ['m_{xi,0}']
            output['xi'][1]['disc'] = ['d_{xi,a}']
            output['xi'][1]['light'] = ['s_{xi}']
            output['xi'][1]['strange'] = ['d_{xi,s}']
            output['xi'][1]['xpt'] = ['B_{xi,2}']

            output['xi'][2]['xpt'] = [
                'g_{xi,xi}', 'g_{xi_st,xi}', 'm_{xi_st,0}']
            
            output['xi'][3]['disc'] = ['d_{xi,aa}', 'd_{xi,al}', ]
            output['xi'][3]['strange'] = [
                'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}']
            output['xi'][3]['light'] = ['b_{xi,4}','B_{xi,4}']
            output['xi'][3]['xpt'] = ['a_{xi,4}', 's_{xi,bar}','S_{xi,bar}']
            output['xi'][3]['fpi'] = ['A_{xi,4}','c0']

            output['xi_st'][0]['light'] = ['m_{xi_st,0}']
            output['xi_st'][1]['disc'] = ['d_{xi_st,a}']
            output['xi_st'][1]['light'] = ['s_{xi,bar}']
            output['xi_st'][1]['strange'] = ['d_{xi_st,s}']
            output['xi_st'][2]['xpt'] = [
                'g_{xi_st,xi_st}', 'g_{xi_st,xi}', 'm_{xi,0}']
            output['xi_st'][3]['disc'] = ['d_{xi_st,aa}', 'd_{xi_st,al}']
            output['xi_st'][3]['strange'] = [
                'd_{xi_st,as}', 'd_{xi_st,ls}', 'd_{xi_st,ss}']
            output['xi_st'][3]['light'] = ['b_{xi_st,4}','B_{xi_st,4}']
            output['xi_st'][3]['xpt'] = ['a_{xi_st,4}', 's_{xi}']

            if lec_type in output[particle][order]:
                return output[particle][order][lec_type]
            else:
                return []
            
class Lambda(lsqfit.MultiFitterModel):
    """
    Constructs the mass extrapolation fit functions using SU(2) hbxpt. 
    fitfcn_{order}_xpt: comprised of terms that depend on the pseudo-Goldstone bosons
    fitfcn_{order}_ct: taylor(counterterm) expansion comprised of terms that arise from using a discetized lattice (a,L,m_pi)

    Note: the chiral order arising in the taylor expansion denotes inclusion of a chiral logarithm 
    """
    def __init__(self, datatag, model_info):
        super().__init__(datatag)
        self.model_info = model_info


    def fitfcn(self, p, data=None,xdata = None):
        """
        fitting in phys units:
            y = (aM)(hbarc/a)
            f = f(m_pi,lam_chi,eps_a,..)
            ---> c = a / hbarc
        Input into lsqfit for both types:
            y -> cy = aM
            f -> cf
        """
        hbar_c = 197.3269804

        if xdata is None:
            xdata ={}
        # if 'units' not in xdata:
        #     xdata['units'] = p['units']
        # if 'units_MeV' not in xdata:
        #     xdata['units_MeV'] = p['units_MeV']
        if 'm_pi' not in xdata:
            xdata['m_pi'] = p['m_pi']
        if 'lam_chi' not in xdata:
            xdata['lam_chi'] = p['lam_chi']
        if self.model_info['units'] == 'phys':
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        else:
            xdata['eps_pi'] = p['eps_pi'] # for F_pi unit fits
        if self.model_info['order_chiral'] is not None: # arg for non-analytic functions that arise in xpt terms
            xdata['eps_delta'] = (p['m_{xi_st,0}'] - p['m_{xi,0}']) / p['lam_chi']
        if 'eps2_a' not in xdata:
            xdata['eps2_a'] = p['eps2_a']

        #strange quark mass mistuning
        if self.model_info['order_strange'] is not None:
            xdata['d_eps2_s'] = ((2 * p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2) - 0.3513
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
        output = 0


        if self.model_info['units'] == 'phys':
            output += (
            self.fitfcn_llo_ct(p,xdata) +
            self.fitfcn_lo_ct(p, xdata) +
            self.fitfcn_nlo_xpt(p, xdata)+ 
            self.fitfcn_n2lo_ct(p, xdata) +
            self.fitfcn_n2lo_xpt(p, xdata) 
            )
        return output 
    

    def fitfcn_mass_deriv(self, p, data=None,xdata = None):
        xdata = self.prep_data(p, data, xdata)
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
    
        output = 0 #llo
        output += self.fitfcn_lo_deriv(p,xdata)  
        output += self.fitfcn_nlo_xpt_deriv(p,xdata) 
        output += self.fitfcn_n2lo_ct_deriv(p,xdata)
        output += self.fitfcn_n2lo_xpt_deriv(p,xdata)
        if self.model_info['units'] == 'fpi':
            output *= xdata['lam_chi']
        else:
            return output
        
    def fitfcn_llo_ct(self,p,xdata):
        output = 0
        if self.model_info['order_light'] in [0,1,2,3]:
            output+= p['m_{lambda,0}']
        return output 

    def fitfcn_lo_ct(self, p, xdata):
        ''''pure taylor extrapolation to O(m_pi^2)'''
        lo_orders = [1,2,3]

        output = 0
        if self.model_info['units'] == 'phys': # lam_chi dependence ON #
            if self.model_info['order_disc'] is not None and self.model_info['order_disc']in lo_orders:
                output+=p['m_{lambda,0}']*(p['d_{lambda,a}'] * xdata['eps2_a'])

            if self.model_info['order_strange'] is not None and self.model_info['order_light'] in lo_orders:
                output+= p['m_{lambda,0}']*(p['d_{lambda,s}'] * xdata['d_eps2_s'])

            if self.model_info['order_light'] is not None and self.model_info['order_strange']in lo_orders and 'd_{xi,s}' in p:
                output+= p['s_{lambda}'] * xdata['lam_chi'] * xdata['eps_pi']**2

        # if self.model_info['units'] == 'fpi': # lam_chi dependence OFF #
        #     if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
        #         output+=(p['d_{lambda,a}'] * xdata['eps2_a'])

        #     if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
        #         output+= (p['d_{lambda,s}'] * xdata['d_eps2_s'])

        #     if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
        #         output+= (p['S_{lambda}'] * xdata['eps_pi']**2)
        #         # if self.model_info['fpi_log']: #this extra log(eps_pi^2) term comes from fpi xpt expression
        #         #     output+= p['m_{lambda,0}'] * xdata['eps_pi']**2 * np.log(xdata['eps_pi']**2)
                    
        return output

    # def fitfcn_lo_deriv(self,p,xdata):
    #     '''derivative expansion to O(m_pi^2)'''
    #     output = 0
    #     if self.model_info['units'] == 'phys':
    #         if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
    #             output += p['m_{lambda,0}'] * (p['d_{lambda,a}'] * xdata['eps2_a'])
        
    #         if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
    #                 output+= p['s_{lambda}'] *xdata['eps_pi']* (
    #                         (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['eps_pi']**2)+
    #                         (2*xdata['lam_chi']*xdata['eps_pi'])
    #                 )
    #         if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
    #             output+= p['m_{lambda,0}']*(p['d_{lambda,s}'] *  xdata['d_eps2_s'])
            
    #     elif self.model_info['units'] == 'fpi':
    #         if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
    #             output += p['d_{lambda,a}'] * xdata['eps2_a']
        
    #         if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
    #                 output+= p['s_{lambda}'] *xdata['eps_pi']**2
                           
    #         if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
    #             output+= p['d_{lambda},s}'] *  xdata['d_eps2_s']

    #     return output
    def fitfcn_nlo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^3)'''

        def compute_phys_output():
            term1 = xdata['lam_chi'] * (-1/2) * p['g_{lambda,sigma}']**2 * naf.fcn_F(xdata['eps_pi'], xdata['eps_sigma'])
            term2 = (2 * p['g_{lambda,sigma_st}']**2 * xdata['lam_chi'] * naf.fcn_F(xdata['eps_pi'], xdata['eps_sigma_st']))
            return term1 - term2
        
        def compute_fpi_output():
            term1 = (-1/2) * p['g_{lambda,sigma}']**2 * naf.fcn_F(xdata['eps_pi'], xdata['eps_sigma'])
            term2 = (2 * p['g_{lambda,sigma_st}']**2 * naf.fcn_F(xdata['eps_pi'], xdata['eps_sigma_st']))
            return term1 - term2
        
        if self.model_info['order_chiral'] is not None:
            if self.model_info['order_chiral'] in [1,2,3]:
                if self.model_info['units'] == 'phys':
                    output += compute_phys_output()
                elif self.model_info['units'] == 'fpi':
                    output += compute_fpi_output()
            else:
                return output
        return output

    def fitfcn_nlo_xpt_deriv(self, p,xdata):
        """Derivative expansion XPT expression at O(m_pi^3)"""

        if not self.model_info['xpt']:
            return 0

        def compute_phys_output():
            term1 = (-1/2) * p['g_{lambda,sigma}']**2* xdata['eps_pi'] *((self.d_de_lam_chi_lam_chi(p, xdata) * xdata['lam_chi']) * xdata['eps_pi']**3 +(3 * xdata['lam_chi'] * xdata['eps_pi']**2))  * naf.fcn_F(xdata['eps_pi'], xdata['eps_sigma'])

            term2 = (2 * p['g_{lambda,sigma_st}']**2 ** xdata['eps_pi']*(
            xdata['lam_chi']* self.d_de_lam_chi_lam_chi(p, xdata)) * naf.fcn_F(xdata['eps_pi'], xdata['eps_sigma_st']) +
                xdata['lam_chi'] * naf.fcn_dF(xdata['eps_pi'], xdata['eps_sigma_st'])
            )
            return term1 - term2
        
        def compute_fpi_output():
            term1 = (-3/4) * p['g_{lambda,sigma}']**2  *xdata['eps_pi']**3 
            term2 = naf.fcn_F(xdata['eps_pi'], xdata['eps_sigma'])
            term2 = (2 * p['g_{lambda,sigma_st}']**2 * naf.fcn_dF(xdata['eps_pi'], xdata['eps_sigma_st']))
            return term1 - term2
        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output = compute_fpi_output()
        else:
            return 0

        return output


    def fitfcn_n2lo_ct(self, p, xdata):
        ''''pure taylor extrapolation to O(m_pi^4)'''
        def compute_order_strange():
            term1 = p['d_{lambda,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']
            term2 = p['d_{lambda,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2
            term3 = p['d_{lambda,ss}'] * xdata['d_eps2_s']**2

            return term1 + term2 + term3

        def compute_order_disc():
            term1 = p['d_{lambda,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2
            term2 = p['d_{lambda,aa}'] * xdata['eps2_a']**2

            return term1 + term2

        def compute_order_light():
            return xdata['eps_pi']**4 * p['b_{lambda,4}']

        def compute_order_chiral():
            return xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) * p['a_{lambda,4}']

        output = 0

        if self.model_info['units'] == 'phys':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += p['m_{lambda,0}'] * compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += p['m_{lambda,0}'] * compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += xdata['eps_pi']**4 * p['b_{lambda,4}'] * xdata['lam_chi']

            if self.model_info['order_chiral'] in ['n2lo']:
                output += xdata['lam_chi'] * compute_order_chiral()

        elif self.model_info['units'] == 'fpi':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light()

            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral()

        return output

    def fitfcn_n2lo_ct_deriv(self, p, xdata):
        ''''derivative expansion to O(m_pi^4) without terms coming from xpt expressions'''
        def compute_order_strange():
            term1 = p['d_{lambda,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']
            term2 = p['d_{lambda,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2
            term3 = p['d_{lambda,ss}'] * xdata['d_eps2_s']**2

            return term1 + term2 + term3

        def compute_order_disc():
            term1 = p['d_{lambda,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2
            term2 = p['d_{lambda,aa}'] * xdata['eps2_a']**2

            return term1 + term2

        def compute_order_light(fpi=None): 
            term1 =  p['b_{lambda,4}']* xdata['eps_pi']
            term2 =  (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi'])*xdata['eps_pi']**4 
            term3 =  4 * xdata['lam_chi'] * xdata['eps_pi']**3
            if fpi:

                termfpi = p['a_{lambda,4}']* xdata['eps_pi']**4 
                termfpi2 = 2 * p['b_{lambda,4}']* xdata['eps_pi']**4
                termfpi3 = p['s_{lambda}']*(1/4*xdata['eps_pi']**4 - 1/4* p['l3_bar']* xdata['eps_pi']**4)
                return termfpi + termfpi2 + termfpi3
            else:
                return term1*(term2+term3)

        def compute_order_chiral(fpi=None):
            term1 =  p['a_{lambda,4}']* xdata['eps_pi']
            term2 =  (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi'])*xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) 
            term3 = 4 * xdata['lam_chi'] * xdata['eps_pi']**3 * np.log(xdata['eps_pi']**2)
            term4 = 2 * xdata['lam_chi'] * xdata['eps_pi']**3 

            if fpi:
                return p['a_{lambda,4}']* (2*xdata['eps_pi']**4*np.log(xdata['eps_pi']**2))
            return term1*(term2+term3+term4)
        output = 0

        if self.model_info['units'] == 'phys':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += p['m_{lambda,0}'] * compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += p['m_{lambda,0}'] * compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light()

            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral()

        elif self.model_info['units'] == 'fpi':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light(fpi=True)

            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral(fpi=True)

        return output


    def fitfcn_n2lo_xpt(self, p, xdata):
        """XPT extrapolation to O(m_pi^4)"""
        if self.model_info['xpt'] is False:
            return 0

        term1 = 3/4 * p['g_{lambda,sigma}']** 2 * (p['s_{lambda}'] - p['s_{sigma}']) * xdata['eps_pi'] ** 2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_sigma'])
        term2 = 3* p['g_{lambda,sigma_st}']** 2 * (p['s_{lambda}'] - p['s_{sigma,bar}']) * xdata['eps_pi'] ** 2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_sigma_st'])

        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            return xdata['lam_chi'] * (term1+term2)

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            return (term1+term2)

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output = compute_fpi_output()
        else:
            return 0
        return output

    def fitfcn_n2lo_xpt_deriv(self, p, xdata):
        '''xpt expression for mass derivative expansion at O(m_pi^4)'''
        if self.model_info['xpt'] is False:
            return 0

        term1_base = 3/4 * p['g_{lambda,sigma}']** 2 * (p['s_{lambda}'] - p['s_{sigma}']) 

        term1 = (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi']) * xdata['eps_pi']** 2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_sigma'])
        term2 = 2* xdata['lam_chi'] *xdata['eps_pi'] *  naf.fcn_J(xdata['eps_pi'], xdata['eps_sigma'])
        term3 = xdata['lam_chi'] * xdata['eps_pi']**2 * naf.fcn_dJ(xdata['eps_pi'], xdata['eps_sigma'])

        term2_base = 3* p['g_{lambda,sigma_st}']** 2 * (p['s_{lambda}'] - p['s_{sigma,bar}']) * xdata['eps_pi'] ** 2 * xdata['lam_chi']
        term1_ = (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi']) * xdata['eps_pi']** 2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_sigma_st'])
        term2_ = 2* xdata['lam_chi'] *xdata['eps_pi'] *  naf.fcn_J(xdata['eps_pi'], xdata['eps_sigma_st'])
        term3_ = xdata['lam_chi'] * xdata['eps_pi']**2 * naf.fcn_dJ(xdata['eps_pi'], xdata['eps_sigma_st'])

        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            return term1_base*(term1+term2+term3) + term2_base*(term1_+term2_+term3_)

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            return (term1+term2)

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output = compute_fpi_output()
        else:
            return 0

        return output

    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]

class Sigma(BaseMultiFitterModel):
    '''
    SU(2) hbxpt extrapolation multifitter class for the Sigma baryon
    '''
    def __init__(self, datatag, model_info):
        super().__init__(datatag,model_info)
        self.model_info = model_info

    def fitfcn(self, p, data=None,xdata=None):
        '''extrapolation formulae'''
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
        xdata = self.prep_data(p,data,xdata)

        output = p['m_{sigma,0}'] #llo
        output += self.fitfcn_lo_ct(p, xdata)
        output += self.fitfcn_nlo_xpt(p, xdata)
        output += self.fitfcn_n2lo_ct(p, xdata)
        output += self.fitfcn_n2lo_xpt(p, xdata)

        return output

    def fitfcn_mass_deriv(self, p, data=None,xdata = None):
        xdata = self.prep_data(p, data, xdata)
        if data is not None:
            for key in data.keys():
                p[key] = data[key]

        output = 0 #llo
        output += self.fitfcn_lo_deriv(p,xdata)  
        output += self.fitfcn_nlo_xpt_deriv(p,xdata) 
        output += self.fitfcn_n2lo_ct_deriv(p,xdata)
        output += self.fitfcn_n2lo_xpt_deriv(p,xdata)
        # if self.model_info['units'] == 'fpi':
        #     output *= xdata['lam_chi']
        # else:
        return output

    def fitfcn_lo_ct(self, p, xdata):
        ''''taylor extrapolation to O(m_pi^2) without terms coming from xpt expressions'''
        output = 0 
        if self.model_info['units'] == 'phys':
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output+= p['m_{sigma,0}']*(p['d_{sigma,a}'] * xdata['eps2_a'])

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['m_{sigma,0}']*(p['d_{sigma,s}'] * xdata['d_eps2_s'])

            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                output+= (p['s_{sigma}'] * xdata['lam_chi'] * xdata['eps_pi']**2)

        if self.model_info['units'] == 'fpi': # lam_chi dependence ON #
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output+= (p['d_{sigma,a}'] * xdata['eps2_a'])

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= (p['d_{sigma,s}'] * xdata['d_eps2_s'])
            
            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                output+= (p['S_{sigma}'] * xdata['eps_pi']**2)
                # if self.model_info['fpi_log']:
                #     output+= p['m_{sigma,0}'] * xdata['eps_pi']**2 * np.log(xdata['eps_pi']**2)
                
        return output
    def fitfcn_lo_deriv(self,p,xdata):
        '''derivative expansion to O(m_pi^2)'''
        output = 0
        if self.model_info['units'] == 'phys':
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['m_{sigma,0}'] * (p['d_{sigma,a}'] * xdata['eps2_a'])
        
            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                    output+= p['s_{sigma}'] *xdata['eps_pi']* (
                            (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['eps_pi']**2)+
                            (2*xdata['lam_chi']*xdata['eps_pi'])
                    )
            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['m_{sigma,0}']*(p['d_{sigma,s}'] *  xdata['d_eps2_s'])
            
        elif self.model_info['units'] == 'fpi':
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['d_{sigma,a}'] * xdata['eps2_a']
        
            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                    output+= p['s_{sigma}'] *xdata['eps_pi']**2
                           
            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['d_{sigma},s}'] *  xdata['d_eps2_s']

        return output

    def fitfcn_nlo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^3)'''
        def compute_phys_output():
            term1 = xdata['lam_chi'] * (-np.pi) *p['g_{sigma,sigma}']**2 * xdata['eps_pi']**3
            term2 = 1/6 * p['g_{lambda,sigma}']**2 * xdata['lam_chi']* naf.fcn_F(xdata['eps_pi'], -xdata['eps_lambda'])
            term3 =  2/3 * p['g_{sigma_st,sigma}']**2 * xdata['lam_chi']* naf.fcn_F(xdata['eps_pi'], xdata['eps_sigma_st'])
            return term1 - term2 - term3
            
        def compute_fpi_output():
            term1 = (-np.pi) *p['g_{sigma,sigma}']**2 * xdata['eps_pi']**3
            term2 = 1/6 * p['g_{lambda,sigma}']**2 * naf.fcn_F(xdata['eps_pi'], -xdata['eps_lambda'])
            term3 =  2/3 * p['g_{sigma_st,sigma}']**2* naf.fcn_F(xdata['eps_pi'], xdata['eps_sigma_st'])
            return term1 - term2 - term3            
        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output = compute_fpi_output()
        else:
            return 0

        return output

    def fitfcn_nlo_xpt_deriv(self, p,xdata):
        """Derivative expansion XPT expression at O(m_pi^3)"""

        if not self.model_info['xpt']:
            return 0

        def compute_phys_output():
            term1 = -np.pi *p['g_{sigma,sigma}']**2 * xdata['eps_pi'] *((self.d_de_lam_chi_lam_chi(p, xdata) * xdata['lam_chi']) * xdata['eps_pi']**3 +(3 * xdata['lam_chi'] * xdata['eps_pi']**2))  * naf.fcn_F(xdata['eps_pi'], xdata['eps_lambda'])

            term2 = 1/6 * p['g_{lambda,sigma}']**2  * xdata['eps_pi']*(xdata['lam_chi']* self.d_de_lam_chi_lam_chi(p, xdata)) * naf.fcn_F(xdata['eps_pi'], -xdata['eps_lambda'])
            
            term3 =  2/3 * p['g_{sigma_st,sigma}']**2 * xdata['eps_pi']*(
            xdata['lam_chi']* naf.fcn_F(xdata['eps_pi'], xdata['eps_sigma_st']))
            return term1 - term2 - term3



        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output = compute_fpi_output()
        else:
            return 0

        return output
            


    def fitfcn_n2lo_ct(self, p, xdata):
        ''''taylor extrapolation to O(m_pi^4)'''
        def compute_order_strange():
            term1 = p['d_{sigma,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']
            term2 = p['d_{sigma,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2
            term3 = p['d_{sigma,ss}'] * xdata['d_eps2_s']**2

            return term1 + term2 + term3
        
        def compute_order_disc():
            term1 = p['d_{sigma,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2
            term2 = p['d_{sigma,aa}'] * xdata['eps2_a']**2

            return term1 + term2

        def compute_order_light():
            term1= xdata['eps_pi']**4 * p['b_{sigma,4}']
            return term1 
        
        def compute_order_chiral():
            return xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) * p['a_{sigma,4}']
        
        output = 0

        if self.model_info['units'] == 'phys':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += p['m_{sigma,0}'] * compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += p['m_{sigma,0}'] * compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light() * xdata['lam_chi']

            if self.model_info['order_chiral'] in ['n2lo']:
                output += xdata['lam_chi'] * compute_order_chiral()

        if self.model_info['units'] == 'fpi':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light()

            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral()

        return output
    
    def fitfcn_n2lo_ct_deriv(self, p, xdata):
        ''''derivative expansion to O(m_pi^4) without terms coming from xpt expressions'''
        def compute_order_strange():
            term1 = p['d_{sigma,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']
            term2 = p['d_{sigma,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2
            term3 = p['d_{sigma,ss}'] * xdata['d_eps2_s']**2

            return term1 + term2 + term3

        def compute_order_disc():
            term1 = p['d_{sigma,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2
            term2 = p['d_{sigma,aa}'] * xdata['eps2_a']**2

            return term1 + term2

        def compute_order_light(fpi=None): 
            term1 =  p['b_{sigma,4}']* xdata['eps_pi']
            term2 =  (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi'])*xdata['eps_pi']**4 
            term3 =  4 * xdata['lam_chi'] * xdata['eps_pi']**3
            if fpi:
                termfpi = p['a_{sigma,4}']* xdata['eps_pi']**4 
                termfpi2 = 2 * p['b_{sigma,4}']* xdata['eps_pi']**4
                termfpi3 = p['s_{sigma}']*(1/4*xdata['eps_pi']**4 - 1/4* p['l3_bar']* xdata['eps_pi']**4)
                return termfpi + termfpi2 + termfpi3
            return term1*(term2+term3)

        def compute_order_chiral(fpi=None):
            term1 =  p['a_{sigma,4}']* xdata['eps_pi']
            term2 =  (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi'])*xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) 
            term3 = 4 * xdata['lam_chi'] * xdata['eps_pi']**3 * np.log(xdata['eps_pi']**2)
            term4 = 2 * xdata['lam_chi'] * xdata['eps_pi']**3 

            if fpi:
                return p['a_{sigma,4}']* (2*xdata['eps_pi']**4*np.log(xdata['eps_pi']**2))
            return term1*(term2+term3+term4)
        output = 0

        if self.model_info['units'] == 'phys':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += p['m_{sigma,0}'] * compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += p['m_{sigma,0}'] * compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light()

            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral()

        elif self.model_info['units'] == 'fpi':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light(fpi=True)

            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral(fpi=True)

        return output


    def fitfcn_n2lo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^4)'''
        if self.model_info['xpt'] is False:
            return 0 
        
        term1 = p['g_{sigma_st,sigma}']**2 * (p['s_{sigma}']-p['s_{sigma,bar}'])*xdata['lam_chi']*xdata['eps_pi']**2 *naf.fcn_J(xdata['eps_pi'], xdata['eps_sigma_st'])
        term2 = (1/4)*p['g_{lambda,sigma}']**2 * (p['s_{sigma}'] -p['s_{lambda}']) * xdata['lam_chi'] * xdata['eps_pi']**2
        term3 = naf.fcn_J(xdata['eps_pi'], -xdata['eps_lambda'])

        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            return xdata['lam_chi'] * (term1+(term2*term3))

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            return (term1+(term2*term3))

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output = compute_fpi_output()
        else:
            return 0

        return output
    def fitfcn_n2lo_xpt_deriv(self, p, xdata):
        '''xpt expression for mass derivative expansion at O(m_pi^4)'''
        if self.model_info['xpt'] is False:
            return 0

        term1_base = 3/4 * p['g_{lambda,sigma}']** 2 * (p['s_{sigma}'] - p['s_{sigma}']) 

        term1 = (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi']) * xdata['eps_pi']** 2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_sigma'])
        term2 = 2* xdata['lam_chi'] *xdata['eps_pi'] *  naf.fcn_J(xdata['eps_pi'], xdata['eps_sigma'])
        term3 = xdata['lam_chi'] * xdata['eps_pi']**2 * naf.fcn_dJ(xdata['eps_pi'], xdata['eps_sigma'])

        term2_base = 3* p['g_{lambda,sigma_st}']** 2 * (p['s_{sigma}'] - p['s_{sigma,bar}']) * xdata['eps_pi'] ** 2 * xdata['lam_chi']
        term1_ = (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi']) * xdata['eps_pi']** 2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_sigma_st'])
        term2_ = 2* xdata['lam_chi'] *xdata['eps_pi'] *  naf.fcn_J(xdata['eps_pi'], xdata['eps_sigma_st'])
        term3_ = xdata['lam_chi'] * xdata['eps_pi']**2 * naf.fcn_dJ(xdata['eps_pi'], xdata['eps_sigma_st'])

        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            return term1_base*(term1+term2+term3) + term2_base*(term1_+term2_+term3_)

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            return (term1+term2)

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output = compute_fpi_output()
        else:
            return 0

        return output


    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]

class Sigma_st(BaseMultiFitterModel):
    '''
    SU(2) hbxpt extrapolation multifitter class for the sigma* baryon
    '''
    def __init__(self, datatag, model_info):
        super().__init__(datatag,model_info)
        self.model_info = model_info

    def fitfcn(self, p, data=None,xdata=None):
        '''extrapolation formulae'''
        xdata = self.prep_data(p,data,xdata)
        if data is not None:
            for key in data.keys():
                p[key] = data[key]

       # not-even leading order
        output = self.fitfcn_llo_ct(p,xdata)
        output += self.fitfcn_lo_ct(p, xdata)
        output += self.fitfcn_nlo_xpt(p, xdata)
        output += self.fitfcn_n2lo_ct(p, xdata)
        output += self.fitfcn_n2lo_xpt(p, xdata)

        return output
    
    def fitfcn_mass_deriv(self, p, data=None,xdata = None):
        xdata = self.prep_data(p, data, xdata)
        if data is not None:
            for key in data.keys():
                p[key] = data[key]

        output = 0 #llo
        output += self.fitfcn_lo_deriv(p,xdata)  
        output += self.fitfcn_nlo_xpt_deriv(p,xdata) 
        output += self.fitfcn_n2lo_ct_deriv(p,xdata)
        output += self.fitfcn_n2lo_xpt_deriv(p,xdata)
        if self.model_info['units'] == 'fpi':
            output *= xdata['lam_chi']
        else:
            return output
        
    def fitfcn_llo_ct(self,p,xdata):
        '''not-even leading order term proportional to the g.s mass of the hyperon'''
        output = 0
        output+= p['m_{sigma_st,0}'] # phys vs fpi flag already implemented in priors.py
        # if self.model_info['units'] == 'fpi': # lam_chi dependence ON #
        #     output+= p['c0'] # M_H^0 / lam_chi approximated as a cnst 
        return output 
    
    def fitfcn_lo_ct(self, p, xdata):
        ''''pure taylor extrapolation to O(m_pi^2)'''
        output = 0
        if self.model_info['units'] == 'phys': # lam_chi dependence ON #
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output+=p['m_{sigma_st,0}']*(p['d_{sigma_st,a}'] * xdata['eps2_a'])

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['m_{sigma_st,0}']*(p['d_{sigma_st,s}'] * xdata['d_eps2_s'])

            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                output+= (p['s_{sigma,bar}'] * xdata['lam_chi'] * xdata['eps_pi']**2)

        if self.model_info['units'] == 'fpi': # lam_chi dependence ON #
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output+=(p['d_{sigma_st,a}'] * xdata['eps2_a'])

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= (p['d_{sigma_st,s}'] * xdata['d_eps2_s'])

            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                output+= p['S_{sigma,bar}']  * xdata['eps_pi']**2 
                # if self.model_info['fpi_log']:
                #     output+= p['m_{sigma,0}'] * xdata['eps_pi']**2 * np.log(xdata['eps_pi']**2)

        return output
    
    def fitfcn_lo_deriv(self,p,xdata):
        '''derivative expansion to O(m_pi^2)'''
        output = 0
        if self.model_info['units'] == 'phys':
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['m_{sigma_st,0}'] * (p['d_{sigma_st,a}'] * xdata['eps2_a'])
        
            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                    output+= p['s_{sigma,bar}'] *xdata['eps_pi']* (
                            (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['eps_pi']**2)+
                            (2*xdata['lam_chi']*xdata['eps_pi'])
                    )
            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['m_{sigma_st,0}']*(p['d_{lambda,s}'] *  xdata['d_eps2_s'])
            
        elif self.model_info['units'] == 'fpi':
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['d_{sigma_st,a}'] * xdata['eps2_a']
        
            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                    output+= p['s_{sigma,bar}'] *xdata['eps_pi']**2
                           
            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['d_{sigma_st},s}'] *  xdata['d_eps2_s']

        return output

    def fitfcn_nlo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^3)'''

        def compute_phys_output():
            term1 =  ((-5/9)*np.pi) *p['g_{sigma_st,sigma_st}']**2 * xdata['eps_pi']**3
            term2 = (1/3) * p['g_{sigma_st,sigma}']**2  * naf.fcn_F(xdata['eps_pi'], -xdata['eps_sigma'])
            term3 = (1/3) * p['g_{lambda,sigma_st}']**2  * naf.fcn_F(xdata['eps_pi'], -xdata['eps_lambda'])
            if self.model_info['units'] == 'phys':
                return term1*xdata['lam_chi'] - term2* xdata['lam_chi']  - term3* xdata['lam_chi']
            
            return term1 - term2 - term3
        
        if self.model_info['xpt']:
            output = compute_phys_output()
        else:
            return 0
        return output

    def fitfcn_nlo_xpt_deriv(self, p,xdata):
        """Derivative expansion XPT expression at O(m_pi^3)"""

        if not self.model_info['xpt']:
            return 0

        def compute_phys_output():
            term1 = -np.pi *p['g_{sigma_st,sigma_st}']**2 * xdata['eps_pi'] *((self.d_de_lam_chi_lam_chi(p, xdata) * xdata['lam_chi']) * xdata['eps_pi']**3 +(3 * xdata['lam_chi'] * xdata['eps_pi']**2))  * naf.fcn_F(xdata['eps_pi'], xdata['eps_lambda'])

            term2 = 1/3 * p['g_{sigma_st,sigma}']**2  * xdata['eps_pi']*(xdata['lam_chi']* self.d_de_lam_chi_lam_chi(p, xdata)) * naf.fcn_F(xdata['eps_pi'], -xdata['eps_lambda'])
            
            term3 =  1/3 * p['g_{lambda,sigma_st}']**2 * xdata['eps_pi']*(
            xdata['lam_chi']* naf.fcn_F(xdata['eps_pi'], xdata['eps_sigma_st']))
            return term1 - term2 - term3

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output = compute_fpi_output()
        else:
            return 0

        return output

    def fitfcn_n2lo_ct(self, p, xdata):
        ''''pure taylor extrapolation to O(m_pi^4)'''
        def compute_order_strange():
            term1 = p['d_{sigma_st,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']
            term2 = p['d_{sigma_st,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2
            term3 = p['d_{sigma_st,ss}'] * xdata['d_eps2_s']**2

            return term1 + term2 + term3

        def compute_order_disc():
            term1 = p['d_{sigma_st,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2
            term2 = p['d_{sigma_st,aa}'] * xdata['eps2_a']**2

            return term1 + term2

        def compute_order_light():
            return xdata['eps_pi']**4 * p['b_{sigma_st,4}']

        def compute_order_chiral():
            return xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) * p['a_{sigma_st,4}']

        output = 0

        if self.model_info['units'] == 'phys':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += p['m_{sigma_st,0}'] * compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += p['m_{sigma_st,0}'] * compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += xdata['eps_pi']**4 * p['b_{sigma_st,4}'] * xdata['lam_chi']

            if self.model_info['order_chiral'] in ['n2lo']:
                output += xdata['lam_chi'] * compute_order_chiral()

        elif self.model_info['units'] == 'fpi':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light()

            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral()

        return output
    
    def fitfcn_n2lo_ct_deriv(self, p, xdata):
        ''''derivative expansion to O(m_pi^4) without terms coming from xpt expressions'''
        def compute_order_strange():
            term1 = p['d_{sigma_st,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']
            term2 = p['d_{sigma_st,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2
            term3 = p['d_{sigma_st,ss}'] * xdata['d_eps2_s']**2

            return term1 + term2 + term3

        def compute_order_disc():
            term1 = p['d_{sigma_st,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2
            term2 = p['d_{sigma_st,aa}'] * xdata['eps2_a']**2

            return term1 + term2

        def compute_order_light(fpi=None): 
            term1 =  p['b_{sigma_st,4}']* xdata['eps_pi']
            term2 =  (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi'])*xdata['eps_pi']**4 
            term3 =  4 * xdata['lam_chi'] * xdata['eps_pi']**3
            if fpi:

                termfpi = p['a_{sigma_st,4}']* xdata['eps_pi']**4 
                termfpi2 = 2 * p['b_{sigma_st,4}']* xdata['eps_pi']**4
                termfpi3 = p['s_{sigma,bar}']*(1/4*xdata['eps_pi']**4 - 1/4* p['l3_bar']* xdata['eps_pi']**4)
                return termfpi + termfpi2 + termfpi3
            else:
                return term1*(term2+term3)

        def compute_order_chiral(fpi=None):
            term1 =  p['a_{sigma_st,4}']* xdata['eps_pi']
            term2 =  (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi'])*xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) 
            term3 = 4 * xdata['lam_chi'] * xdata['eps_pi']**3 * np.log(xdata['eps_pi']**2)
            term4 = 2 * xdata['lam_chi'] * xdata['eps_pi']**3 

            if fpi:
                return p['a_{sigma_st,4}']* (2*xdata['eps_pi']**4*np.log(xdata['eps_pi']**2))
            else:
                return term1*(term2+term3+term4)
        output = 0

        if self.model_info['units'] == 'phys':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += p['m_{sigma_st,0}'] * compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += p['m_{sigma_st,0}'] * compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light()

            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral()

        elif self.model_info['units'] == 'fpi':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light(fpi=True)

            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral(fpi=True)

        return output

    def fitfcn_n2lo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^4)'''
        if self.model_info['xpt'] is False:
            return 0

        term1 = (1/2)*p['g_{sigma_st,sigma}']**2 * (p['s_{sigma,bar}']-p['s_{sigma}']) *xdata['lam_chi'] * xdata['eps_pi']**2 *(naf.fcn_J(xdata['eps_pi'], -xdata['eps_sigma']))
        term2 = (1/2)*p['g_{lambda,sigma_st}']**2 * (p['s_{sigma,bar}'] -p['s_{sigma}']) * xdata['lam_chi'] * xdata['eps_pi']**2 * naf.fcn_J(xdata['eps_pi'], -xdata['eps_lambda'])

        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            return xdata['lam_chi'] * (term1+term2)

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            term3 = 1/4*p['m_{sigma_st,0}']*xdata['eps_pi']**4*(np.log(xdata['eps_pi']**2)**2)

            return (term1+term2-term3)

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output = compute_fpi_output()
        else:
            return 0

        return output
        
    def fitfcn_n2lo_xpt_deriv(self, p, xdata):
        '''xpt expression for mass derivative expansion at O(m_pi^4)'''
        if self.model_info['xpt'] is False:
            return 0

        term1_base = 3/4 * p['g_{sigma_st,sigma}']** 2 * (p['s_{sigma,bar}'] - p['s_{sigma}']) 

        term1 = (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi']) * xdata['eps_pi']** 2 * naf.fcn_J(xdata['eps_pi'], -xdata['eps_sigma'])
        term2 = 2* xdata['lam_chi'] *xdata['eps_pi'] *  naf.fcn_J(xdata['eps_pi'], -xdata['eps_sigma'])
        term3 = xdata['lam_chi'] * xdata['eps_pi']**2 * naf.fcn_dJ(xdata['eps_pi'],- xdata['eps_sigma'])

        term2_base = 3* p['g_{lambda,sigma_st}']** 2 * (p['s_{sigma,bar}'] - p['s_{sigma}']) * xdata['eps_pi'] ** 2 * xdata['lam_chi']
        term1_ = (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi']) * xdata['eps_pi']** 2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_sigma_st'])
        term2_ = 2* xdata['lam_chi'] *xdata['eps_pi'] *  naf.fcn_J(xdata['eps_pi'], xdata['eps_lambda'])
        term3_ = xdata['lam_chi'] * xdata['eps_pi']**2 * naf.fcn_dJ(xdata['eps_pi'], xdata['eps_lambda'])

        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            return term1_base*(term1+term2+term3) + term2_base*(term1_+term2_+term3_)

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            return (term1+term2)

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output = compute_fpi_output()
        else:
            return 0

        return output

    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]


class Xi(lsqfit.MultiFitterModel):
    """
    Constructs the mass extrapolation fit functions using SU(2) hbxpt. 
    fitfcn_{order}_xpt: comprised of terms that depend on the pseudo-Goldstone bosons
    fitfcn_{order}_ct: taylor(counterterm) expansion comprised of terms that arise from using a discetized lattice (a,L,m_pi)

    Note: the chiral order arising in the taylor expansion denotes inclusion of a chiral logarithm 
    """
    def __init__(self, datatag, model_info):
        super().__init__(datatag)
        self.model_info = model_info

    def fitfcn(self, p, data=None,xdata = None):
        """
        fitting in phys units:
            y = (aM)(hbarc/a)
            f = f(m_pi,lam_chi,eps_a,..)
            ---> c = a / hbarc
        Input into lsqfit for both types:
            y -> cy = aM
            f -> cf
        """
        hbar_c = 197.3269804

        if xdata is None:
            xdata ={}
        # if 'units' not in xdata:
        #     xdata['units'] = p['units']
        # if 'units_MeV' not in xdata:
        #     xdata['units_MeV'] = p['units_MeV']
        if 'm_pi' not in xdata:
            xdata['m_pi'] = p['m_pi']
        if 'lam_chi' not in xdata:
            xdata['lam_chi'] = p['lam_chi']
        if self.model_info['units'] == 'phys':
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        else:
            xdata['eps_pi'] = p['eps_pi'] # for F_pi unit fits
        if self.model_info['order_chiral'] is not None: # arg for non-analytic functions that arise in xpt terms
            xdata['eps_delta'] = (p['m_{xi_st,0}'] - p['m_{xi,0}']) / p['lam_chi']
        if 'eps2_a' not in xdata:
            xdata['eps2_a'] = p['eps2_a']

        #strange quark mass mistuning
        if self.model_info['order_strange'] is not None:
            xdata['d_eps2_s'] = ((2 * p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2) - 0.3513
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
        output = 0


        if self.model_info['conversion_factor']:
            if self.model_info['units'] == 'phys':
                output += (
                self.fitfcn_llo_ct(p,xdata) +
                self.fitfcn_lo_ct(p, xdata) +
                self.fitfcn_nlo_xpt(p, xdata)+ 
                self.fitfcn_n2lo_ct(p, xdata) +
                self.fitfcn_n2lo_xpt(p, xdata) 
                )
        #     elif self.moif 'units' not in xdata:
        # xdata['units'] = p['units']del_info['units'] in ['fpi','latt']:
        #         output += xdata['a_fm']*xdata['lam_chi'] *(
        #         self.fitfcn_llo_ct(p,xdata) +
        #         self.fitfcn_lo_ct(p, xdata) +
        #         self.fitfcn_nlo_xpt(p, xdata)+ 
        #         self.fitfcn_n2lo_ct(p, xdata) +
        #         self.fitfcn_n2lo_xpt(p, xdata) 
        #         )

            

        else:
    
            output += self.fitfcn_llo_ct(p,xdata)
            output += self.fitfcn_lo_ct(p, xdata)
            output += self.fitfcn_nlo_xpt(p, xdata) 
            output += self.fitfcn_n2lo_ct(p, xdata) 
            output += self.fitfcn_n2lo_xpt(p, xdata) 

        return output 
    
    def fitfcn_llo_ct(self,p,xdata):
        output = 0
        if self.model_info['order_light'] in [0,1,2,3]:
            output+= p['m_{xi,0}']
        return output 

    def fitfcn_lo_ct(self, p, xdata):
        """pure taylor-type fit to O(m_pi^2)"""
        lo_orders = [1,2,3]

        output = 0
        if self.model_info['units'] == 'phys': # lam_chi dependence ON #
            if self.model_info['order_disc'] is not None and self.model_info['order_disc']in lo_orders:
                output += p['m_{xi,0}'] * (p['d_{xi,a}'] * xdata['eps2_a'])

            if self.model_info['order_light'] is not None and self.model_info['order_light'] in lo_orders:
                output += (p['s_{xi}'] * xdata['lam_chi'] * xdata['eps_pi']**2)

            if self.model_info['order_strange'] is not None and self.model_info['order_strange']in lo_orders and 'd_{xi,s}' in p:
                output += p['m_{xi,0}']*(p['d_{xi,s}'] * xdata['d_eps2_s'])

        if self.model_info['units'] == 'fpi': # lam_chi dependence OFF #
            if self.model_info['order_disc'] is not None and self.model_info['order_disc'] in lo_orders:
                output +=  (p['d_{xi,a}'] * xdata['eps2_a'])

            if self.model_info['order_light']  is not None and self.model_info['order_light'] in lo_orders:
                output += (p['s_{xi}'] * xdata['eps_pi']**2)

            if self.model_info['order_strange'] is not None and self.model_info['order_strange'] in lo_orders:
                output += (p['d_{xi,s}'] * xdata['d_eps2_s'])

            if self.model_info['order_chiral'] in lo_orders:
                    # if self.model_info['fv']:
                    #     output += p['c0'] * xdata['eps_pi']**2 * fv.fcn_I_m(xdata['eps_pi']**2,xdata['L'],xdata['lam_chi',10])
                    # else:
                    output += p['B_{xi,2}'] * xdata['eps_pi']**2
                    output += p['c0'] * xdata['eps_pi']**2 * np.log(xdata['eps_pi']**2)

        return output
    
    
    def fitfcn_nlo_xpt(self, p, xdata):
        """XPT extrapolation to O(m_pi^3)"""
        output= 0

        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            term1 = xdata['lam_chi'] * (-3/2) * np.pi * p['g_{xi,xi}']**2 * xdata['eps_pi']**3
            term2 = p['g_{xi_st,xi}']**2 * xdata['lam_chi'] * naf.fcn_F(xdata['eps_pi'], xdata['eps_delta'])

            return term1 - term2

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            term3 = (-3/2) * np.pi * p['g_{xi,xi}']** 2 * xdata['eps_pi'] ** 3
            term4 = p['g_{xi_st,xi}']** 2 * naf.fcn_F(xdata['eps_pi'], xdata['eps_delta'])
            return term3 - term4
        
        if self.model_info['order_chiral'] is not None:
            if self.model_info['order_chiral'] in [1,2,3]:
                if self.model_info['units'] == 'phys':
                    output += compute_phys_output()
                elif self.model_info['units'] == 'fpi':
                    output += compute_fpi_output()
            else:
                return output

        return output

    def fitfcn_n2lo_ct(self, p, xdata):
        """Taylor extrapolation to O(m_pi^4) without terms coming from xpt expressions"""

        def compute_order_strange():
            term1 = p['d_{xi,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']
            term2 = p['d_{xi,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2
            term3 = p['d_{xi,ss}'] * xdata['d_eps2_s']**2

            return term1 + term2 + term3

        def compute_order_disc():
            term1 = p['d_{xi,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2
            term2 = p['d_{xi,aa}'] * xdata['eps2_a']**2

            return term1 + term2

        def compute_order_light(fpi:bool):
            if fpi:
                return xdata['eps_pi']**4 * p['B_{xi,4}'] #term 1 in xpt expansion (no logs or non-analytic fcns)

            return xdata['eps_pi']**4 * p['b_{xi,4}']

        def compute_order_chiral(fpi:bool):
            # if self.model_info['fv']:
            #     return xdata['eps_pi']**4 * fv.fcn_I_m(xdata['eps_pi']**2,xdata['L'],xdata['lam_chi'],10)  * p['a_{xi,4}']
            # else:
            if self.model_info['order_chiral'] == 3:
                # return xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) * p['A_{xi,4}']

                return xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) * p['a_{xi,4}']

        output = 0

        if self.model_info['units'] == 'phys':  # lam_chi dependence ON 
            if self.model_info['order_strange'] == 3:
                output += p['m_{xi,0}'] * compute_order_strange()

            if self.model_info['order_disc'] == 3:
                output += p['m_{xi,0}'] * compute_order_disc()

            if self.model_info['order_light'] == 3:
                output += xdata['eps_pi']**4 * p['b_{xi,4}'] * xdata['lam_chi']

            if self.model_info['order_chiral'] == 3:
                output += xdata['lam_chi'] * compute_order_chiral(fpi=False)

        elif self.model_info['units'] == 'fpi':  # lam_chi dependence ON 
            if self.model_info['order_strange'] == 3:
                output += compute_order_strange()

            if self.model_info['order_disc'] == 3:
                output += compute_order_disc()

            if self.model_info['order_light'] == 3:
                output += compute_order_light(fpi=True)

            if self.model_info['order_chiral'] == 3:
                output += compute_order_chiral(fpi=True)

        return output

    def fitfcn_n2lo_xpt(self, p, xdata):
        """XPT extrapolation to O(m_pi^4)"""
        output = 0

        def base_term():
            """Computes the base term which is common for both 'phys' and 'fpi' units."""
            return (3/2) * p['g_{xi_st,xi}']** 2 * (p['s_{xi}'] - p['s_{xi,bar}']) * xdata['eps_pi'] ** 2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_delta'])

        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            return xdata['lam_chi'] * base_term()

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            return base_term()
             
        if self.model_info['order_chiral'] is not None:
            if self.model_info['order_chiral']== 3:
                if self.model_info['units'] == 'phys':
                    output = compute_phys_output()
                elif self.model_info['units'] == 'fpi':
                    output = compute_fpi_output()
            else:
                return 0

        return output
    
    def fpi_corrections_n2lo(self,p,xdata):
        output = 0 

        if self.model_info['units'] == 'fpi':
            return -1/4*p['c0']*xdata['eps_pi']**4*np.log(xdata['eps_pi']**2)**2


    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]

class Xi_st(lsqfit.MultiFitterModel):
    """
    SU(2) hbxpt extrapolation multifitter class for the Xi baryon
    """
    def __init__(self, datatag, model_info):
        super().__init__(datatag)
        self.model_info = model_info

    def fitfcn(self, p, data=None,xdata=None):
        """extraplation formulae
        When fitting in F_pi units, 
            y_data = M_H/lam_chi
            f = f(eps_pi,eps_a,..)
            ---> c = a*lam_chi
        fitting in phys units:
            y = (aM)(hbarc/a)
            f = f(m_pi,lam_chi,eps_a,..)
            ---> c = a / hbarc
        Input into lsqfit for both types:
            y -> cy = aM
            f -> cf
        """
        if xdata is None:
            xdata ={}

        # if 'units' not in xdata:
        #     xdata['units'] = p['units']
        if 'm_pi' not in xdata:
            xdata['m_pi'] = p['m_pi']
        # if 'units' not in xdata:
        #     xdata['units'] = p['units']
        # if 'units_MeV' not in xdata:
        #     xdata['units_MeV'] = p['units_MeV']
        if 'lam_chi' not in xdata:
            xdata['lam_chi'] = p['lam_chi']
        if self.model_info['units'] == 'phys':
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        else:
            xdata['eps_pi'] = p['eps_pi']
        if self.model_info['order_chiral'] >0:

            xdata['eps_delta'] = (p['m_{xi_st,0}'] - p['m_{xi,0}']) / p['lam_chi']
        if 'eps2_a' not in xdata:
            xdata['eps2_a'] = p['eps2_a']
        #strange quark mass mistuning
        if self.model_info['order_strange'] >0:
            xdata['d_eps2_s'] = ((2 * p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2) - 0.3513

        if data is not None:
            for key in data.keys():
                p[key] = data[key]
        output = 0
        hbar_c = 197.3269804


        if self.model_info['conversion_factor']:
            if self.model_info['units'] == 'phys':
                output += (
                self.fitfcn_llo_ct(p,xdata) +
                self.fitfcn_lo_ct(p, xdata) +
                self.fitfcn_nlo_xpt(p, xdata)+ 
                self.fitfcn_n2lo_ct(p, xdata) +
                self.fitfcn_n2lo_xpt(p, xdata) 
                )
            elif self.model_info['units'] in ['fpi','latt']:
                output += xdata['units'] *(
                self.fitfcn_llo_ct(p,xdata) +
                self.fitfcn_lo_ct(p, xdata) +
                self.fitfcn_nlo_xpt(p, xdata)+ 
                self.fitfcn_n2lo_ct(p, xdata) +
                self.fitfcn_n2lo_xpt(p, xdata) 
                )

        else:
    
            output += self.fitfcn_llo_ct(p,xdata)
            output += self.fitfcn_lo_ct(p, xdata)
            output += self.fitfcn_nlo_xpt(p, xdata) 
            output += self.fitfcn_n2lo_ct(p, xdata) 
            output += self.fitfcn_n2lo_xpt(p, xdata) 

        return output

        # if self.model_info['units'] == 'fpi':

        #     return output * xdata['lam_chi'] * xdata['a_fm']
        # elif self.model_info['units'] == 'phys':

    def fitfcn_llo_ct(self,p,xdata):
        output = 0
        if self.model_info['order_light'] in [0,1,2,3]:
            if self.model_info['units'] == 'phys':
                output+= p['m_{xi_st,0}']
            else:
                output+= p['c0']
        return output 
    
    def fitfcn_lo_ct(self, p, xdata):
        """'taylor extrapolation to O(m_pi^2)
        Note: chiral terms arise at this order when fitting in F_pi units due to expansion of lambda_chi
        """
        output = 0
        lo_orders = [1,2,3]
        if self.model_info['units'] == 'phys': # lam_chi dependence ON #
            if self.model_info['order_disc'] is not None and self.model_info['order_disc'] in lo_orders:
                output += (p['m_{xi_st,0}'] * (p['d_{xi_st,a}']*xdata['eps2_a']))

            if self.model_info['order_light'] is not None and self.model_info['order_light'] in lo_orders:
                output += (p['s_{xi,bar}'] * xdata['lam_chi'] * xdata['eps_pi']**2)

            if self.model_info['order_strange'] is not None and self.model_info['order_strange'] in lo_orders and 'd_{xi_st,s}' in p:
                    output += p['m_{xi_st,0}']*(p['d_{xi_st,s}'] * xdata['d_eps2_s'])
                    
        elif self.model_info['units'] == 'fpi': # lam_chi dependence OFF #
            if self.model_info['order_disc'] in lo_orders:
                output += (p['d_{xi_st,a}']*xdata['eps2_a'])

            if self.model_info['order_chiral'] in lo_orders:
                output += (p['c0'] * xdata['eps_pi']**2*np.log(xdata['eps_pi']))

            if self.model_info['order_strange'] in lo_orders:
                output += (p['d_{xi_st,s}'] * xdata['d_eps2_s'])

            if self.model_info['order_light'] in lo_orders:
                output+= (p['S_{xi,bar}'] * xdata['eps_pi']**2)

        return output
    
    def fitfcn_nlo_xpt(self, p, xdata):
        """xpt extrapolation to O(m_pi^3)"""
        output = 0
        def compute_phys_output():
             
            term1 = xdata['lam_chi']* (-5/6) * np.pi * p['g_{xi_st,xi_st}']**2 * xdata['eps_pi']**3
            term2 = xdata['lam_chi']* 1/2* p['g_{xi_st,xi}']**2 * naf.fcn_F(xdata['eps_pi'], -xdata['eps_delta'])

            return term1 - term2

               
        if self.model_info['order_chiral'] is not None:
                if self.model_info['order_chiral'] in [1,2,3]:
                    if self.model_info['units'] == 'phys':
                        output += compute_phys_output()
                else:
                    return output

        return output


    def fitfcn_n2lo_ct(self, p, xdata):
        """Taylor extrapolation to O(m_pi^4) without terms coming from xpt expressions"""

        def compute_order_strange():
            term1 = p['d_{xi_st,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']
            term2 = p['d_{xi_st,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2
            term3 = p['d_{xi_st,ss}'] * xdata['d_eps2_s']**2

            return term1 + term2 + term3

        def compute_order_disc():
            term1 = p['d_{xi_st,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2
            term2 = p['d_{xi_st,aa}'] * xdata['eps2_a']**2

            return term1 + term2

        def compute_order_light():
            return xdata['eps_pi']**4 * p['b_{xi_st,4}']

        # def compute_order_chiral():
        #     # if self.model_info['fv']:
        #     #     return p['a_{xi_st,4}']* (xdata['eps_pi']**4*fv.fcn_I_m(xdata['eps_pi']**2,xdata['L'],xdata['lam_chi'],10))
        #     # else:
        #     return xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) * p['a_{xi_st,4}']

        output = 0

        if self.model_info['units'] == 'phys':  
            if self.model_info['order_strange'] == 3:
                output += p['m_{xi_st,0}'] * compute_order_strange()

            if self.model_info['order_disc'] == 3:
                output += p['m_{xi_st,0}'] * compute_order_disc()

            if self.model_info['order_light'] == 3:
                output += xdata['eps_pi']**4 * p['b_{xi_st,4}'] * xdata['lam_chi']

            # if self.model_info['order_chiral'] == 3:
            #     output += xdata['lam_chi'] * compute_order_chiral()

        elif self.model_info['units'] == 'fpi':  
            if self.model_info['order_strange'] == 3:
                output += compute_order_strange()

            if self.model_info['order_disc'] == 3:
                output += compute_order_disc()

            if self.model_info['order_light'] == 3:
                output += compute_order_light()

            if self.model_info['order_chiral'] == 3:
                output += compute_order_chiral()

        return output
    
    def fitfcn_n2lo_xpt(self, p, xdata):
        """XPT extrapolation to O(m_pi^4)"""
        output = 0
        def base_term():
            """Computes the base term which is common for both 'phys' and 'fpi' units."""
            term1 =  (np.log(xdata['eps_pi']**2) * p['a_{xi_st,4}']*xdata['eps_pi']**2)
            term2 = (3/4) * p['g_{xi_st,xi}'] ** 2 * (p['s_{xi,bar}']-p['s_{xi}']) * xdata['eps_pi'] ** 2 * naf.fcn_J(xdata['eps_pi'], -xdata['eps_delta'])

            return xdata['eps_pi']**2*(term1+term2)
        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            return xdata['lam_chi'] * base_term()

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            return base_term()
        
        if self.model_info['order_chiral'] is not None:
            if self.model_info['order_chiral'] ==3:
                if self.model_info['units'] == 'phys':
                    output += compute_phys_output()
                elif self.model_info['units'] in ('fpi','lattice'):
                    output += compute_fpi_output()
        else:
            return 0

        return output
    
    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return data[self.datatag]
