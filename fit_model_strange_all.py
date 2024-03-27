import copy
import lsqfit
import numpy as np
import h5py as h5
import gvar as gv
import functools
from pathlib import Path
import matplotlib.pyplot as plt

# local modules
import xpt.non_analytic_functions as naf
import xpt.fv_corrections as fv
import xpt.i_o as i_o
# import xpt.priors as priors

class FitRoutine:
    """
    The `FitRoutine` class is designed to fit models to data using least squares fitting.
    It takes in the model information, and options for empirical Bayes analysis.

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
                 strange:str,
                 model_info:dict,
                 decorr_scale:str,
                ):

        self.prior = prior
        self.data = data
        if self.data is None:
            raise ValueError('you need to pass data to the fitter')
        self._phys_pt_data = phys_pt_data
        
        self.model_info = model_info.copy()
        self.svd_test = self.model_info['svd_test']
        self.svd_tol = self.model_info['svd_tol']
        self.strange = strange
        self.scheme = self.model_info['eps2a_defn']
        self.units = self.model_info['units']
        # self.convert_data = self.model_info['convert_data_before']
        self.decorr_scale = decorr_scale

        self.input_output = i_o.InputOutput(scheme=self.scheme,units=self.units,strange=self.strange,scale_correlation=self.decorr_scale)

        self._posterior = None
        self.emp_bayes = self.model_info['emp_bayes']
        self.emp_bayes_grouping = self.model_info['emp_bayes_grouping']

        self.empbayes_grouping = None #groups for empirical bayes prior study
        self._empbayes_fit = None
        # this is manually reconstructing the gvar to decorrelate x and y data
        lam_sigma_particles = ['lambda', 'sigma', 'sigma_st']
        xi_particles = ['xi','xi_st']
        y_particles = ['xi','xi_st','lambda', 'sigma', 'sigma_st']
        pseudoscalars = ['m_pi','m_k','eps_pi']
        hbarc =  197.3269804, # MeV-fm
        self.models, self.models_dict = self._make_models()
        

        # if self.model_info['units'] == 'phys':
            # self.scale_data = 
            # when one performs a simult. fit of s=1,2 hyperons, must ensure that the hyperons of other strangeness are excluded from the y data
        if 'lambda' in self.model_info['particles']:
            self.data_subset = {part : self.data['m_'+part] for part in lam_sigma_particles}
        if 'xi' in self.model_info['particles'] :
            self.data_subset = {part:self.data['m_'+part] for part in xi_particles}
            if self.model_info['units'] == 'fpi':
                self.y = {part: self.data['m_'+part]*data['lam_chi']*data['a_fm'] for part in xi_particles}
            if self.model_info['units'] == 'phys':
                self.y = {part: self.data['m_'+part]*data['a_fm']/hbarc for part in xi_particles}
        if 'proton' in self.model_info['particles'] :
            self.data_subset = {'proton':self.data['m_proton']}
            if self.model_info['units'] == 'fpi':
                self.y = {'proton': data['m_proton']*data['lam_chi']*data['a_fm']}
            if self.model_info['units'] == 'phys':
                self.y = {'proton':data['m_proton']*data['a_fm']/hbarc}

        #     # self.y = {k : gv.gvar(gv.mean(data['m_'+k]), gv.sdev(data['m_'+k])) for k in self.data_subset}
        # self.y = { k : gv.gvar(gv.mean(data['m_'+k]), gv.sdev(data['m_'+k])) for k in self.data_subset}
        # self.x = {scale : self.data[scale] for scale in pseudoscalars}
        
    def __str__(self):
        return str(self.fit)


    @functools.cached_property
    def fit(self):
        prior_final = self._make_prior()
        data = self.y
        fitter = lsqfit.MultiFitter(models=self.models)
        if self.svd_test:
            svd_cut = self.input_output.perform_svdcut()
            fit = fitter.lsqfit(data=data, prior=prior_final, fast=False, mopt=False,svdcut=svd_cut)
            # fig = plt.figure('svd_diagnosis', figsize=(7, 4))
            # for ens in self.ensembles:
           
            raise ValueError("Invalid input for strangeness. Should be either 0,1,2 as a string")
        model_names = []
        unique_names = set()

        # for order_c in orders_chiral:
        for order_chiral,order_strange, order_disc, order_light,unit in itertools.product(orders_chiral,orders_strange,orders_disc,orders_light,units):

            # Build the model name
            parts = []
            for particle in particles:
                parts.append(particle)

            unit_list = []
            unit_list.append(unit)

            # scale_corr_list = []
            # scale_corr_list.append(scale_corr)
            if order_chiral and order_light and order_chiral > order_light:
                continue
            if order_strange:
                parts.append(f"s_{order_strange}")
            if order_disc:
                parts.append(f"d_{order_disc}")
            if order_light:
                parts.append(f"l_{order_light}")

            name = ':'.join(parts)
            if name not in unique_names:
                unique_names.add(name)

            # Add model info
            model_info = {
                'name': name,
                'order_chiral': order_chiral,
                'order_strange': order_strange,
                'order_disc': order_disc, #     svd_test.plot_ratio(show=True)
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
    
    def extrapolate_observable(self, observable):
        results = {}
        for model_name, model in self.models_dict.items():
            if model_name in self.model_info['particles']:
                results[model_name] = model.extrapolate(p=self.posterior, data=self.phys_point_data, observable=observable)
        return results
    
    def extrapolation(self,observables, p=None, data=None, xdata=None):
        '''chiral extrapolations of baryon mass data using the Feynman-Hellmann theorem to quantify pion mass and strange quark mass dependence of baryon masses. Extrapolations are to the physical point using PDG data. 
        
        Returns(takes a given subset of observables as a list):
        - extrapolated mass (meV)
        - pion sigma term 
        - barred pion sigma term / M_B'''

        if p is None:
            p = self.posterior
        if data is None:
           
            raise ValueError("Invalid input for strangeness. Should be either 0,1,2 as a string")
        model_names = []
        unique_names = set()

        # for order_c in orders_chiral:
        for order_chiral,order_strange, order_disc, order_light,unit in itertools.product(orders_chiral,orders_strange,orders_disc,orders_light,units):

            # Build the model name
            parts = []
            for particle in particles:
                parts.append(particle)

            unit_list = []
            unit_list.append(unit)

            # scale_corr_list = []
            # scale_corr_list.append(scale_corr)
            if order_chiral and order_light and order_chiral > order_light:
                continue
            if order_strange:
                parts.append(f"s_{order_strange}")
            if order_disc:
                parts.append(f"d_{order_disc}")
            if order_light:
                parts.append(f"l_{order_light}")

            name = ':'.join(parts)
            if name not in unique_names:
                unique_names.add(name)

            # Add model info
            model_info = {
                'name': name,
                'order_chiral': order_chiral,
                'order_strange': order_strange,
                'order_disc': order_disc, data = self._phys_pt_data
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
        if xdata is None:
            xdata = {}
        if self.model_info['units'] == 'phys':
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        elif self.model_info['units'] == 'fpi':
            xdata['eps_pi'] = p['eps_pi']
        p['l3_bar'] = -1/4 * (
            gv.gvar('3.53(26)') + np.log(xdata['eps_pi']**2))
        p['l4_bar'] =  gv.gvar('4.73(10)')
        p['c2_F'] = gv.gvar(0,20)
        p['c1_F'] = gv.gvar(0,20)
         
        MULTIFIT_DICT = {
            'xi': Xi,
            'xi_st': Xi_st,
            'lambda': Lambda,
            'sigma': Sigma,
            'sigma_st': Sigma_st,
        }
        # a_fm conversion factors by ensemble 
        ENS_CONVERSION = {
            'a06': gv.gvar(0.05686,49),
            'a09': gv.gvar(0.08722,67),
            'a12': gv.gvar(0.12055,87),
            'a15': gv.gvar(0.15036,98)
        }
        # mapping fit.y results to individual ensembles to match conversion factor 
        ENS_MAP = {
            'a06': slice(0,None),
            'a09': slice(1, 5),
            'a12': slice(6, 10),
            'a15': slice(11, 16)
        }

        def convert_to_mev(y_values):
            converted_values = y_values.copy()

            for ensemble, s in ENS_MAP.items():
                conversion_factor = ENS_CONVERSION[ensemble]
                converted_values[s] = y_values[s] * conversion_factor

            return converted_values

        
        results = {}

        for particle in self.model_info['particles']:
            model_class = MULTIFIT_DICT.get(particle)
            if model_class is not None:
                model_instance = model_class(datatag=particle, model_info=self.model_info)
        
                results[particle] = {}
                for obs in observables:
                # results = {}

                    output = 0
                    mass = 0

                # compute the baryon sigma term
                    if obs == 'sigma_pi':
                        # if self.model_info['units'] == 'phys':
                            # output += xdata['eps_pi']*1/2 *(
                            #     1 + xdata['eps_pi']**2 *(
                            #     5/2 - 1/2*p['l3_bar'] - 2*p['l4_bar'])
                            #     ) * model_instance.fitfcn_mass_deriv(p=p,data=data,xdata=xdata)
                            output+= model_instance.fitfcn_mass_deriv(p=p,data=data,xdata=xdata)
                        # else:
                            # output += model_instance.fitfcn_mass_deriv(p=p,data=data,xdata=xdata)
                    elif obs == 'sigma_bar':
                        if self.model_info['units'] == 'phys':
                            output += xdata['eps_pi']*1/2 *(
                                1 + xdata['eps_pi']**2 *(
                                5/2 - 1/2*p['l3_bar'] - 2*p['l4_bar'])
                                ) * model_instance.fitfcn_mass_deriv(p=p,data=data,xdata=xdata)
                            output = output /  model_instance.fitfcn(p=p)
                        else:
                            output += model_instance.fitfcn_mass_deriv(p=p,data=data,xdata=xdata)
                            output = output / model_instance.fitfcn(p=p)
                    # extrapolate hyperon mass to the physical point 
                    elif obs == 'mass':
                        if self.model_info['units'] == 'lattice':
                            for particle in self.model_info['particles']:
                                output+= model_instance.fitfcn(p=p) * self.phys_point_data['hbarc']
                        if self.model_info['units'] == 'fpi':
                            output+= model_instance.fitfcn(p=p) * self.phys_point_data['lam_chi']
                        if self.model_info['units'] == 'phys':
                            output+= model_instance.fitfcn(p=p) 
                    results[particle][obs] = output
        return results
    

    def _empbayes_grouping(self):
        '''
        routine adapted from @millernb
        '''
        zkeys = {}
        if self.empbayes_grouping == 'all':
            for param in self.prior:
                zkeys[param] = [param]
        # include particle choice xi or xi_st to fill inside bracket
        elif self.empbayes_grouping == 'order':
            # vary all light quark terms together, strange terms together
            zkeys['chiral_llo'] = ['m_{xi,0}', 'm_{xi_st,0}']
            zkeys['chiral_lo'] = ['s_{xi}', 's_{xi,bar}']
            zkeys['chiral_nlo'] = ['g_{xi,xi}',
                                   'g_{xi_st,xi}', 'g_{xi_st,xi_st}']
            zkeys['chiral_n2lo'] = [
                'b_{xi,4}', 'b_{xi_st,4}', 'a_{xi,4}', 'a_{xi_st,4}']
            zkeys['disc_nlo'] = ['d_{xi,a}',
                                 'd_{xi_st,a}', 'd_{xi,s}', 'd_{xi_st,s}']
            zkeys['disc_n2lo'] = [
                'd_{xi,aa}', 'd_{xi,al}', 'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}']

        # discretization effects
        elif self.empbayes_grouping == 'disc':
            zkeys['chiral'] = ['m_{xi,0}', 'm_{xi_st,0}', 's_{xi}', 's_{xi,bar}', 'g_{xi,xi}', 'g_{xi_st,xi}', 'g_{xi_st,xi_st}',
                               'b_{xi,4}', 'b_{xi_st,4}', 'a_{xi,4}', 'a_{xi_st,4}']
            zkeys['disc'] = ['d_{xi,a}', 'd_{xi_st,a}', 'd_{xi,s}', 'd_{xi_st,s}',
                             'd_{xi,aa}', 'd_{xi,al}', 'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}']
            
        elif self.empbayes_grouping == 'disc_only':
            zkeys['disc'] = ['d_{xi,a}', 'd_{xi_st,a}', 'd_{xi,s}', 'd_{xi_st,s}',
                             'd_{xi,aa}', 'd_{xi,al}', 'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}',
                             'd_{xi_st,aa}','d_{xi_st,al}' ,'d_{xi_st,as}','d_{xi_st,ls}', 'd_{xi_st,ss}']
            
        all_keys = [k for g in zkeys for k in zkeys[g]]
        prior_keys = list(self._make_prior())
        ignored_keys = set(all_keys) - set(prior_keys)

        # Don't determine empirical priors in param not in model
        for group in zkeys:
            for key in ignored_keys:
                if key in ignored_keys and key in zkeys[group]:
                    zkeys[group].remove(key)

        return zkeys

    def _make_empbayes_fit(self, empbayes_grouping='disc_only',observable=None):
        if (self._empbayes_fit is None) or (empbayes_grouping != self.empbayes_grouping):
            self.empbayes_grouping = empbayes_grouping
            self._counter = {'iters' : 0, 'evals' : 0}

            z0 = gv.BufferDict()
            for group in self._empbayes_grouping():
                z0[group] = 1.0

            def analyzer(arg):
                self._counter['evals'] += 1
                print("\nEvals: ", self._counter['evals'], arg,"\n")
                print(type(arg[0]))
                return None
            models = self._make_models()
            fitter = lsqfit.MultiFitter(models=models)

            fit, z = fitter.empbayes_fit(z0, fitargs=self._make_fitargs, maxit=20, analyzer=analyzer,tol=0.1)
            self._empbayes_fit = fit

        return self._empbayes_fit

    def _make_fitargs(self, z):
        '''
        preparing fit args that will be passed to fitter.empbayes_fit
        '''
        data = self.data
        prior = self._make_prior(z=z)

        # Ideally:
        # Don't bother with more than the hundredth place
        # Don't let z=0 (=> null GBF)
        # Don't bother with negative values (meaningless)
        # But for some reason, these restrictions (other than the last) cause empbayes_fit not to converge
        # multiplicity = {}
        # for key in z:
        #     multiplicity[key] = 0
        #     z[key] = np.abs(z[key])

        # Helps with convergence (minimizer doesn't use extra digits -- bug in lsqfit?)
        def sig_fig(x): return np.around(
            x, int(np.floor(-np.log10(x))+3))  # Round to 3 sig figs

        def capped(x, x_min, x_max): return np.max([np.min([x, x_max]), x_min])

        zkeys = self._empbayes_grouping()
        zmin = 1e-2
        zmax = 1e3
        for group in z.keys():
            for param in prior.keys():
                if param in zkeys[group]:
                    z[group] = sig_fig(capped(z[group], zmin, zmax))
                    prior[param] = gv.gvar(0, 1) * z[group]

        return dict(data=data,prior=prior)

    @property
    def posterior(self):
        return self._get_posterior()

    def _get_posterior(self,param=None):
        if param == 'all':
            return self.fit.p
        if param is not None:
            return self.fit.p[param]
        
        output = {}
        for param in self.prior:
            if param in self.fit.p:
                output[param] = self.fit.p[param]
        return output  
        
    @property
    def phys_point_data(self):
        return self._get_phys_point_data()

    # need to convert to/from lattice units
    def _get_phys_point_data(self, parameter=None):
        if parameter is None:
            return self._phys_pt_data
        else:
            return self._phys_pt_data[parameter]

            
    def get_fitfcn(self,p=None,data=None,particle=None,xdata=None):
        output = {}
        if p is None:
            p = copy.deepcopy(self.posterior)
            # p.update(self.posterior)
        if data is None:
            data = copy.deepcopy(self.phys_point_data)
        p.update(data)
        model_array, model_dict = self._make_models(model_info=self.model_info)
        for mdl in model_array:
            part = mdl.datatag
            output[part] = mdl.fitfcn(p=p,data=data,xdata=xdata)
        if particle is None:
            return output
        return output[particle]

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

        if 'lambda' in model_info['particles']:
            lambda_model = Lambda(datatag='lambda', model_info=model_info)
            model_array = np.append(model_array, lambda_model)
            model_dict['lambda'] = lambda_model

        if 'sigma' in model_info['particles']:
            sigma_model = Sigma(datatag='sigma', model_info=model_info)
            model_array = np.append(model_array, sigma_model)
            model_dict['sigma'] = sigma_model

        if 'sigma_st' in model_info['particles']:
            sigma_st_model = Sigma_st(datatag='sigma_st', model_info=model_info)
            model_array = np.append(model_array, sigma_st_model)
            model_dict['sigma_st'] = sigma_st_model

        if 'proton' in model_info['particles']:
            proton_model = Proton(datatag='proton', model_info=model_info)
            model_array = np.append(model_array, proton_model)
            model_dict['proton'] = proton_model

        return model_array, model_dict

    
    def _make_prior(self, data=None,z=None,scale_data=None):
        '''
        Only need priors for LECs/data needed in fit.
        Separates all parameters that appear in the hyperon extrapolation formulae 
        '''
        if data is None:
            data = self.data
        # print(list(data))
        prior = self.prior
        new_prior = {}
        particles = []
        particles.extend(self.model_info['particles'])

        keys = []
        orders = []
        for p in particles:
            for l, value in [('light', self.model_info['order_light']), ('disc', self.model_info['order_disc']),
                             ('strange', self.model_info['order_strange']), ('xpt', self.model_info['order_chiral'])]:
                # include all orders equal to and less than desired order in expansion #
                if value == 'llo':
                    orders = ['llo']
                elif value == 'lo':
                    orders = ['llo', 'lo']
                elif value == 'nlo':
                    orders = ['llo', 'lo', 'nlo']
                elif value == 'n2lo':
                    orders = ['llo', 'lo', 'nlo', 'n2lo']
                elif value == 'n4lo':
                    orders = ['llo', 'lo', 'nlo', 'n2lo', 'n4lo']
                else:
                    orders = []

                for o in orders:
                    keys.extend(self._get_prior_keys(
                        particle=p, order=o, lec_type=l))
        for key in keys:
            new_prior[key] = prior[key]
        
        if self.model_info['order_strange'] is not None:
            new_prior['m_k'] = data['m_k']
        if self.model_info['order_light'] is not None:
            new_prior['eps2_a'] = data['eps2_a']
            new_prior['m_pi'] = data['m_pi']
            new_prior['lam_chi'] = data['lam_chi']
            new_prior['eps_pi'] = data['eps_pi']
            new_prior['a_fm'] = data['a_fm']
        if self.model_info['fv']:
            new_prior['L'] = data['L']

        if self.model_info['order_disc'] is not None:
            new_prior['lam_chi'] = data['lam_chi']

        if z is None:
            return new_prior
        zkeys = self._empbayes_grouping()

        for k in new_prior:
            for group in zkeys:
                if k in zkeys[group]:
                    new_prior[k] = gv.gvar(0, np.exp(z[group]))
        return new_prior

    def _get_prior_keys(self, particle='all', order='all', lec_type='all'):
        if particle == 'all':
            output = []
            for particle in ['lambda', 'sigma', 'sigma_st', 'xi', 'xi_st','proton','delta']:
                keys = self._get_prior_keys(
                    particle=particle, order=order, lec_type=lec_type)
                output.extend(keys)
            return np.unique(output)

        elif order == 'all':
            output = []
            for order in ['llo', 'lo', 'nlo', 'n2lo', 'n4lo']:
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
            output = {}
            for p in ['lambda', 'sigma', 'sigma_st', 'xi', 'xi_st','proton','delta']:
                output[p] = {}
                for o in ['llo', 'lo', 'nlo', 'n2lo', 'n4lo']:
                    output[p][o] = {}

            
            output['proton']['llo' ]['light'  ] = ['m_{proton,0}']
            output['proton']['lo'  ]['disc'   ] = ['d_{proton,a}']
            output['proton']['lo'  ]['light'  ] = ['b_{proton,2}','l4_bar','c0']
            output['proton']['lo'  ]['strange'] = ['d_{proton,s}']
            output['proton']['lo'  ]['xpt']     = ['l4_bar']
            output['proton']['nlo' ]['xpt'    ] = ['g_{proton,proton}', 'g_{proton,delta}','m_{delta,0}']
            output['proton']['n2lo']['disc'   ] = ['d_{proton,aa}', 'd_{proton,al}']
            output['proton']['n2lo']['strange'] = ['d_{proton,as}', 'd_{proton,ls}','d_{proton,ss}']
            output['proton']['n2lo']['light'  ] = ['b_{proton,4}']
            output['proton']['n2lo']['xpt'    ] = ['a_{proton,4}', 'g_{proton,4}']
            output['proton']['n4lo']['disc'   ] = ['d_{proton,all}', 'd_{proton,aal}']
            output['proton']['n4lo']['strange'] = []
            output['proton']['n4lo']['light'  ] = ['b_{proton,6}']
            output['proton']['n4lo']['xpt'    ] = []

            output['lambda']['llo']['light'] = ['m_{lambda,0}']
            output['lambda']['lo']['disc'] = ['d_{lambda,a}']
            output['lambda']['lo']['strange'] = ['d_{lambda,s}']
            output['lambda']['lo']['light'] = ['s_{lambda}','S_{lambda}']
            output['lambda']['nlo']['xpt'] = [
                'g_{lambda,sigma}', 'g_{lambda,sigma_st}', 'm_{sigma,0}', 'm_{sigma_st,0}']
            output['lambda']['n2lo']['disc'] = [
                'd_{lambda,aa}', 'd_{lambda,al}']
            output['lambda']['n2lo']['strange'] = [
                'd_{lambda,as}', 'd_{lambda,ls}', 'd_{lambda,ss}']
            output['lambda']['n2lo']['light'] = ['b_{lambda,4}','B_{lambda,4}']
            output['lambda']['n2lo']['xpt'] = ['a_{lambda,4}', 's_{sigma}', 's_{sigma,bar}']

            output['sigma']['llo']['light'] = ['m_{sigma,0}']
            output['sigma']['lo']['disc'] = ['d_{sigma,a}']
            output['sigma']['lo']['strange'] = ['d_{sigma,s}']
            output['sigma']['lo']['light'] = ['s_{sigma}','S_{sigma}']
            output['sigma']['nlo']['xpt'] = [
                'g_{sigma,sigma}', 'g_{lambda,sigma}', 'g_{sigma_st,sigma}', 'm_{lambda,0}', 'm_{sigma_st,0}']
            output['sigma']['n2lo']['disc'] = [
                'd_{sigma,aa}', 'd_{sigma,al}', ]
            output['sigma']['n2lo']['strange'] = [
                'd_{sigma,as}', 'd_{sigma,ls}', 'd_{sigma,ss}']
            output['sigma']['n2lo']['light'] = ['b_{sigma,4}','B_{sigma,4}']
            output['sigma']['n2lo']['xpt'] = [
                'a_{sigma,4}', 's_{lambda}', 's_{sigma,bar}']

            output['sigma_st']['llo']['light'] = ['m_{sigma_st,0}']
            output['sigma_st']['lo']['disc'] = ['d_{sigma_st,a}']
            output['sigma_st']['lo']['strange'] = ['d_{sigma_st,s}']
            output['sigma_st']['lo']['light'] = ['s_{sigma,bar}','S_{sigma,bar}']
            output['sigma_st']['nlo']['xpt'] = [
                'g_{sigma_st,sigma_st}', 'g_{lambda,sigma_st}', 'g_{sigma_st,sigma}', 'm_{lambda,0}', 'm_{sigma,0}']
            output['sigma_st']['n2lo']['disc'] = [
                'd_{sigma_st,aa}', 'd_{sigma_st,al}']
            output['sigma_st']['n2lo']['strange'] = [
                'd_{sigma_st,as}', 'd_{sigma_st,ls}', 'd_{sigma_st,ss}']
            output['sigma_st']['n2lo']['light'] = ['b_{sigma_st,4}']
            output['sigma_st']['n2lo']['xpt'] = ['a_{sigma_st,4}', 's_{sigma}','S_{sigma}']

            output['xi']['llo']['light'] = ['m_{xi,0}']
            output['xi']['lo']['disc'] = ['d_{xi,a}']
            output['xi']['lo']['light'] = ['s_{xi}','S_{xi}']
            output['xi']['lo']['strange'] = ['d_{xi,s}']
            output['xi']['lo']['xpt'] = ['c0','B_{xi,2}']

            output['xi']['nlo']['xpt'] = [
                'g_{xi,xi}', 'g_{xi_st,xi}', 'm_{xi_st,0}']
            output['xi']['n2lo']['disc'] = ['d_{xi,aa}', 'd_{xi,al}', ]
            output['xi']['n2lo']['strange'] = [
                'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}']
            output['xi']['n2lo']['light'] = ['b_{xi,4}','B_{xi,4}']
            output['xi']['n2lo']['xpt'] = ['a_{xi,4}', 's_{xi,bar}']

            output['xi_st']['llo']['light'] = ['m_{xi_st,0}']
            output['xi_st']['lo']['disc'] = ['d_{xi_st,a}']
            output['xi_st']['lo']['light'] = ['s_{xi,bar}','S_{xi,bar}']
            output['xi_st']['lo']['strange'] = ['d_{xi_st,s}']
            output['xi_st']['nlo']['xpt'] = [
                'g_{xi_st,xi_st}', 'g_{xi_st,xi}', 'm_{xi,0}']
            output['xi_st']['n2lo']['disc'] = ['d_{xi_st,aa}', 'd_{xi_st,al}']
            output['xi_st']['n2lo']['strange'] = [
                'd_{xi_st,as}', 'd_{xi_st,ls}', 'd_{xi_st,ss}']
            output['xi_st']['n2lo']['light'] = ['b_{xi_st,4}','B_{xi_st,4}']
            output['xi_st']['n2lo']['xpt'] = ['a_{xi_st,4}', 's_{xi}']

            if lec_type in output[particle][order]:
                return output[particle][order][lec_type]
            else:
                return []
            
class BaseMultiFitterModel(lsqfit.MultiFitterModel):
    """base class for all derived hyperon multifitter classes.
    provides the common `prep_data` routine"""
    def __init__(self, datatag, model_info):
        super().__init__(datatag)
        self.model_info = model_info

    def prep_data(self,p,data=None,xdata=None):
        if xdata is None:
            xdata = {}
        if 'a_fm' not in xdata:
            xdata['a_fm'] = p['a_fm']
        if 'm_pi' not in xdata:
            xdata['m_pi'] = p['m_pi']
        if 'lam_chi' not in xdata:
            xdata['lam_chi'] = p['lam_chi']
        if self.model_info['units'] == 'phys':
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        else:
            xdata['eps_pi'] = p['eps_pi']
        if self.datatag == 'proton':
            xdata['eps_delta_'] = (p['m_{delta,0}'] - p['m_{proton,0}']) / p['lam_chi']
        if self.datatag in ['xi', 'xi_st']:
            xdata['eps_delta'] = (p['m_{xi_st,0}'] - p['m_{xi,0}']) / p['lam_chi']
        if self.datatag == 'lambda':
            xdata['eps_sigma_st'] = (
                p['m_{sigma_st,0}'] - p['m_{lambda,0}']) / p['lam_chi']
            xdata['eps_sigma'] = (
                p['m_{sigma,0}'] - p['m_{lambda,0}']) / p['lam_chi']
        if self.datatag == 'sigma':
            xdata['eps_lambda'] = (
            p['m_{sigma,0}'] - p['m_{lambda,0}']) / p['lam_chi']
            xdata['eps_sigma_st'] = (
            p['m_{sigma_st,0}'] - p['m_{sigma,0}']) / p['lam_chi']
        if self.datatag == 'sigma_st':
            xdata['eps_lambda'] = (
            p['m_{sigma_st,0}'] - p['m_{lambda,0}']) / p['lam_chi']
            xdata['eps_sigma'] = (
            p['m_{sigma_st,0}'] - p['m_{sigma,0}']) / p['lam_chi']

        if 'eps2_a' not in xdata:
            xdata['eps2_a'] = p['eps2_a']
        # if 'L' not in xdata:
        #     xdata['L'] = p['L']
        
        #strange quark mass mistuning
        if self.model_info['order_strange'] is not None:
            xdata['d_eps2_s'] = ((2 * p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2) - 0.3513

        return xdata
    
    def d_de_lam_chi_lam_chi(self, p,xdata):
        '''
        see eq. 3.32 of Andre's notes. This is the derivative:
        .. math::
            \Lambda_{\Chi} \frac{\partial}{\partial \epsilon_{\pi}} \frac{M_B}{\Lambda_{\Chi}}
        '''
        output = 0
        if self.model_info['order_light'] in ['lo','n2lo']:
            output += 2 * xdata['eps_pi']* (p['l4_bar']  - np.log(p['eps_pi']**2) -1 )
        if self.model_info['order_light'] in ['n2lo']:
            output += xdata['eps_pi']**4 * (
                3/2 * np.log(xdata['eps_pi']**2)**2 + np.log(xdata['eps_pi']**2)*(2*p['c1_F'] + 2*p['l4_bar']+3/2)
                + 2*p['c2_F'] + p['c1_F'] - p['l4_bar']*(p['l4_bar']-1))

        return output 
    def builddata(self, data):
        return data[self.datatag]
    

class Proton(BaseMultiFitterModel):
    '''
    SU(2) hbxpt extrapolation multifitter class for the proton.
    Includes extrapolation function for the derivative nucleon mass in order to 
    extract the nucleon sigma term
    '''
    def __init__(self, datatag, model_info):
        super().__init__(datatag,model_info)
        self.model_info = model_info

    def fitfcn(self, p, data=None,xdata = None):
        xdata = self.prep_data(p, data, xdata)
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
    
        # output = p['m_{xi,0}'] #llo
        output = self.fitfcn_llo_ct(p,xdata)
        output += self.fitfcn_lo_ct(p, xdata)
        output += self.fitfcn_nlo_xpt(p, xdata) 
        output += self.fitfcn_n2lo_ct(p, xdata) 
        output += self.fitfcn_n2lo_xpt(p, xdata) 


    def fitfcn_llo_ct(self,p,xdata):
        output = 0
        output+= p['m_{proton,0}']
        return output 

    # def fitfcn_lo_xpt(self,p,xdata):
    #     if self.model_info['xpt']:
    #         output = (p['l4_bar'] + p['b_{proton,2}'])  * (xdata['eps_pi']**2 * p['m_{proton,0}'] * np.log(xdata['eps_pi']**2))
    #     else:
    #         return 0
    #     return output

    def fitfcn_lo_ct(self, p, xdata):
        output = 0

        if self.model_info['units'] == 'phys': # lam_chi dependence ON #
            if self.model_info['order_disc']    in  ['lo', 'nlo', 'n2lo']:
                output += p['m_{proton,0}'] * (p['d_{proton,a}'] * xdata['eps2_a'])
        
            if self.model_info['order_light'] in ['lo','nlo','n2lo']:
                output+= p['b_{proton,2}'] * xdata['lam_chi'] * xdata['eps_pi']**2 

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['m_{proton,0}']*   (p['d_{proton,s}'] * xdata['d_eps2_s'])


        elif self.model_info['units'] == 'fpi': # lam_chi dependence OFF #
            if self.model_info['order_disc'] in  ['lo', 'nlo', 'n2lo']:
                output += (p['d_{proton,a}'] * xdata['eps2_a'])
            
            if self.model_info['order_light'] in ['lo','nlo','n2lo','n4lo']:
                output+= (xdata['eps_pi']**2 * p['b_{proton,2}'])

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= (p['d_{proton,s}'] * xdata['d_eps2_s'])

            if self.model_info['xpt']:
                if self.model_info['order_chiral'] in ['lo', 'nlo', 'n2lo']: #include chiral log
                    output += p['c0']*xdata['eps_pi']**2 * np.log(xdata['eps_pi']**2)

        return output

    def fitfcn_nlo_xpt(self,p,xdata):
        """XPT extrapolation to O(m_pi^3)"""

        output = 0
        
        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            term1 = xdata['lam_chi'] * (-3/2) * np.pi * p['g_{proton,proton}']**2 * xdata['eps_pi']**3
            term2 = 4/3* p['g_{proton,delta}']**2 * xdata['lam_chi'] * naf.fcn_F(xdata['eps_pi'], xdata['eps_delta_'])

            return term1 - term2

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            term3 = (-3/2) * np.pi * p['g_{proton,proton}']** 2 * xdata['eps_pi'] ** 3
            term4 = 4/3* p['g_{proton,delta}']** 2 * naf.fcn_F(xdata['eps_pi'], xdata['eps_delta_'])
            return term3 - term4

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output += compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output += compute_fpi_output()
        else:
            return 0

        return output

    def fitfcn_n2lo_xpt(self, p, xdata):
        """XPT extrapolation to O(m_pi^4)"""
        output = 0

        def base_term():
            """Computes the base term which is common for both 'phys' and 'fpi' units."""
            return p['g_{proton,4}'] * xdata['eps_pi'] ** 2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_delta_'])

        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            return xdata['lam_chi'] * base_term()

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            return base_term()

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output = compute_fpi_output()
        else:
            return 0

        return output

    def fitfcn_n2lo_ct(self,p,xdata):
        """Taylor extrapolation to O(m_pi^4) without terms coming from xpt expressions"""

        output = 0
        def compute_order_strange():
            term1 = p['d_{proton,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']
            term2 = p['d_{proton,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2
            term3 = p['d_{proton,ss}'] * xdata['d_eps2_s']**2

            return term1 + term2 + term3

        def compute_order_disc():
            term1 = p['d_{proton,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2
            term2 = p['d_{proton,aa}'] * xdata['eps2_a']**2

            return term1 + term2

        def compute_order_light():
            return xdata['eps_pi']**4 * p['b_{proton,4}']

        def compute_order_chiral():
            # if self.model_info['fv']:
            #     return xdata['eps_pi']**4 * fv.fcn_I_m(xdata['eps_pi']**2,xdata['L'],xdata['lam_chi'],10)  * p['a_{xi,4}']
            # else:
            return xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) * p['a_{proton,4}']

        if self.model_info['units'] == 'phys':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += p['m_{proton,0}'] * compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += p['m_{proton,0}'] * compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += xdata['eps_pi']**4 * p['b_{proton,4}'] * xdata['lam_chi']

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


    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return super().builddata(data)
   


class Xi(BaseMultiFitterModel):
    '''
    SU(2) hbxpt extrapolation multifitter class for the Xi baryon
    Note: the chiral order arising in the taylor expansion denotes inclusion of a chiral logarithm 
    '''
    def __init__(self, datatag, model_info):
        super().__init__(datatag,model_info)
        self.model_info = model_info

    def fitfcn(self, p, data=None,xdata = None):
        xdata = self.prep_data(p, data, xdata)
        # print(data.keys())
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
    
        output = self.fitfcn_llo_ct(p,xdata)
        output += self.fitfcn_lo_ct(p, xdata)
        output += self.fitfcn_nlo_xpt(p, xdata) 
        output += self.fitfcn_n2lo_ct(p, xdata) 
        output += self.fitfcn_n2lo_xpt(p, xdata) 

        return output * xdata['lam_chi'] * xdata['a_fm']
    
    def fitfcn_llo_ct(self,p,xdata):
        output = 0
        output+= p['m_{xi,0}']
        return output 

    def fitfcn_lo_ct(self, p, xdata):
        ''''pure taylor-type fit to O(m_pi^2)'''
        output = 0
        if self.model_info['units'] == 'phys': # lam_chi dependence ON #
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['m_{xi,0}'] * (p['d_{xi,a}'] * xdata['eps2_a'])

            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                output += (p['s_{xi}'] * xdata['lam_chi'] * xdata['eps_pi']**2)

            if self.model_info['order_strange'] is not None:
                if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo'] and 'd_{xi,s}' in p:
                    output += p['m_{xi,0}']*(p['d_{xi,s}'] * xdata['d_eps2_s'])

        if self.model_info['units'] == 'fpi': # lam_chi dependence OFF #
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output +=  (p['d_{xi,a}'] * xdata['eps2_a'])

            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                output += (p['s_{xi}'] * xdata['eps_pi']**2)

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output += (p['d_{xi,s}'] * xdata['d_eps2_s'])

            if self.model_info['xpt']:
                if self.model_info['order_chiral'] in ['lo', 'nlo', 'n2lo']:
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

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output += compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output += compute_fpi_output()
        else:
            return 0

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
            if fpi:
                return xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) * p['A_{xi,4}']

            return xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) * p['a_{xi,4}']

        output = 0

        if self.model_info['units'] == 'phys':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += p['m_{xi,0}'] * compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += p['m_{xi,0}'] * compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += xdata['eps_pi']**4 * p['b_{xi,4}'] * xdata['lam_chi']

            if self.model_info['order_chiral'] in ['n2lo']:
                output += xdata['lam_chi'] * compute_order_chiral(fpi=False)

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
        output = 0

        def base_term():
            """Computes the base term which is common for both 'phys' and 'fpi' units."""
            return (3/2) * p['g_{xi_st,xi}']** 2 * (p['s_{xi}'] - p['s_{xi,bar}']) * xdata['eps_pi'] ** 2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_delta'])

        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            return xdata['lam_chi'] * base_term()

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            return -1/4*p['c0']*xdata['eps_pi']**4*np.log(xdata['eps_pi']**2)**2 + base_term()

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
        return super().builddata(data)

class Xi_st(BaseMultiFitterModel):
    '''
    SU(2) hbxpt extrapolation multifitter class for the Xi baryon
    '''
    def __init__(self, datatag, model_info):
        super().__init__(datatag,model_info)
        self.model_info = model_info

    def fitfcn(self, p, data=None,xdata=None):
        '''extraplation formulae'''
        xdata = self.prep_data(p,data,xdata)
        if data is not None:
            for key in data.keys():
                p[key] = data[key]

        output = p['m_{xi_st,0}'] #llo
        output += self.fitfcn_lo_ct(p, xdata)
        output += self.fitfcn_nlo_xpt(p, xdata)
        output += self.fitfcn_n2lo_ct(p, xdata)
        output += self.fitfcn_n2lo_xpt(p, xdata)

        if self.model_info['units'] == 'fpi':

            return output * xdata['lam_chi'] * xdata['a_fm']
        # elif self.model_info['units'] == 'phys':

        
    
    def fitfcn_lo_ct(self, p, xdata):
        ''''taylor extrapolation to O(m_pi^2) without terms coming from xpt expressions'''
        output = 0
        if self.model_info['units'] == 'phys': # lam_chi dependence ON #
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += (p['m_{xi_st,0}'] * (p['d_{xi_st,a}']*xdata['eps2_a']))

            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                output += (p['s_{xi,bar}'] * xdata['lam_chi'] * xdata['eps_pi']**2)

            if self.model_info['order_strange'] is not None:
                if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo'] and 'd_{xi_st,s}' in p:
                    output += p['m_{xi_st,0}']*(p['d_{xi_st,s}'] * xdata['d_eps2_s'])
                    
        elif self.model_info['units'] == 'fpi': # lam_chi dependence OFF #
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += (p['d_{xi_st,a}']*xdata['eps2_a'])

            if self.model_info['order_chiral'] in ['lo', 'nlo', 'n2lo']:
                output += (p['s_{xi,bar}'] * xdata['eps_pi']**2)

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output += (p['d_{xi_st,s}'] * xdata['d_eps2_s'])

        return output
    
    def fitfcn_nlo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^3)'''

        if not self.model_info['xpt']:
            return 0

        term1 = (-5/6) * np.pi * p['g_{xi_st,xi_st}']**2 * xdata['eps_pi']**3
        term2 = 1/2* p['g_{xi_st,xi}']**2 * naf.fcn_F(xdata['eps_pi'], -xdata['eps_delta'])

        if self.model_info['units'] == 'phys':
            return term1 * xdata['lam_chi'] - term2 * xdata['lam_chi']
        
        if self.model_info['units'] in ('fpi','lattice'):
            return term1 - term2

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

        def compute_order_chiral():
            # if self.model_info['fv']:
            #     return p['a_{xi_st,4}']* (xdata['eps_pi']**4*fv.fcn_I_m(xdata['eps_pi']**2,xdata['L'],xdata['lam_chi'],10))
            # else:
            return xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) * p['a_{xi_st,4}']

        output = 0

        if self.model_info['units'] == 'phys':  
            if self.model_info['order_strange'] in ['n2lo']:
                output += p['m_{xi_st,0}'] * compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += p['m_{xi_st,0}'] * compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += xdata['eps_pi']**4 * p['b_{xi_st,4}'] * xdata['lam_chi']

            if self.model_info['order_chiral'] in ['n2lo']:
                output += xdata['lam_chi'] * compute_order_chiral()

        elif self.model_info['units'] == 'fpi':  
            if self.model_info['order_strange'] in ['n2lo']:
                output += compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light()

            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral()

        return output
    
    def fitfcn_n2lo_xpt(self, p, xdata):
        """XPT extrapolation to O(m_pi^4)"""

        def base_term():
            """Computes the base term which is common for both 'phys' and 'fpi' units."""
            return (3/4) * p['g_{xi_st,xi}'] ** 2 * (p['s_{xi,bar}']-p['s_{xi}']) * xdata['eps_pi'] ** 2 * naf.fcn_J(xdata['eps_pi'], -xdata['eps_delta'])

        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            return xdata['lam_chi'] * base_term()

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            return base_term()

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] in ('fpi','lattice'):
                output = compute_fpi_output()
        else:
            return 0

        return output
    
    def buildprior(self, prior, mopt=False, extend=False):
        return prior

    def builddata(self, data):
        return super().builddata(data)

# Strangeness=1 Hyperons

class Lambda(BaseMultiFitterModel):
    '''
    SU(2) hbxpt extrapolation multifitter class for the Lambda_zero baryon.
    '''
    def __init__(self, datatag, model_info):
        super().__init__(datatag,model_info)
        self.model_info = model_info

    def fitfcn(self, p, data=None,xdata=None):
        '''extraplation formulae'''
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
        xdata = self.prep_data(p,data,xdata)
    
        output =  p['m_{lambda,0}'] #llo
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

    def fitfcn_lo_ct(self, p, xdata):
        ''''pure taylor extrapolation to O(m_pi^2)'''
        output = 0
        if self.model_info['units'] == 'phys': # lam_chi dependence ON #
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output+=p['m_{lambda,0}']*(p['d_{lambda,a}'] * xdata['eps2_a'])

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['m_{lambda,0}']*(p['d_{lambda,s}'] * xdata['d_eps2_s'])

            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                output+= p['s_{lambda}'] * xdata['lam_chi'] * xdata['eps_pi']**2

        if self.model_info['units'] == 'fpi': # lam_chi dependence OFF #
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output+=(p['d_{lambda,a}'] * xdata['eps2_a'])

            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= (p['d_{lambda,s}'] * xdata['d_eps2_s'])

            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                output+= (p['S_{lambda}'] * xdata['eps_pi']**2)
                # if self.model_info['fpi_log']: #this extra log(eps_pi^2) term comes from fpi xpt expression
                #     output+= p['m_{lambda,0}'] * xdata['eps_pi']**2 * np.log(xdata['eps_pi']**2)
                    
        return output

    def fitfcn_lo_deriv(self,p,xdata):
        '''derivative expansion to O(m_pi^2)'''
        output = 0
        if self.model_info['units'] == 'phys':
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['m_{lambda,0}'] * (p['d_{lambda,a}'] * xdata['eps2_a'])
        
            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                    output+= p['s_{lambda}'] *xdata['eps_pi']* (
                            (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['eps_pi']**2)+
                            (2*xdata['lam_chi']*xdata['eps_pi'])
                    )
            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['m_{lambda,0}']*(p['d_{lambda,s}'] *  xdata['d_eps2_s'])
            
        elif self.model_info['units'] == 'fpi':
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['d_{lambda,a}'] * xdata['eps2_a']
        
            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                    output+= p['s_{lambda}'] *xdata['eps_pi']**2
                           
            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['d_{lambda},s}'] *  xdata['d_eps2_s']

        return output
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
