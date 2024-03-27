import numpy as np
import gvar as gv
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.transforms import Bbox
import sys
from mpl_toolkits.mplot3d.axes3d import Axes3D
import sys
import copy
import textwrap

sys.setrecursionlimit(10000)

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['figure.figsize']  = [6.75, 6.75/1.618034333]
mpl.rcParams['font.size']  = 20
mpl.rcParams['legend.fontsize'] =  16
mpl.rcParams["lines.markersize"] = 5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.usetex'] = True

#internal xpt modules
from fit_model import FitModel
import i_o as i_o
import priors as priors

import lsqfitics
import yaml


class Fit_Analysis(FitModel):
    
    def __init__(self, 
                 data:dict,
                 prior:dict,
                 phys_pt_data:dict,
                 units:str,
                 model_info:dict,
                 strange:str,
                 **kwargs):
        
        self.data = data      
        self.prior = prior  
        self._phys_point_data = phys_pt_data
        self.units = unitsplot
        self.strange = strange
        self.model_info = model_info        # allows manual override of x,y data 
        self._input_prior = prior

        self.options = kwargs
        self.verbose = self.options.get('verbose',False) # print all data points
        self.svd_test = self.options.get('svd_test',False)
        self.svd_tol = self.options.get('svd_tol',None)
        self.fitter = fit_model.FitModel(data=self.data,prior=self.prior,phys_pt_data=phys_pt_data,strange=self.strange,kwargs= self.options)
        self._input_prior = prior
        self._fit = {}
        self.fit = self.fitter.fit
        self.model_collection = []
    # def _compare_models(self):
    #     compare_ = compare.ModelComparsion(models=self.model_data, compare_type='phys', units='phys')
    #     compare_.compare_models(particles=['xi', 'xi_st'])
    #     mdl_avg = compare_.model_average(particles=['xi', 'xi_st'])
    #     print(mdl_avg)
    #     compare_.model_plots(system='xi')


    def svd_analysis(self):
        # Specify the svd_tol values to loop over
        svd_tol_values = np.linspace(10e-6, 0.05, num=30)  #this for some reason is not working..
        # svd_tol_values = [1e-6,1e-5,1e-4,1e-3,1e-2]

        chi2 = []
        q = []

        results = {particle: {'mean': [], 'std': []} for particle in self.model_info['particles']}

        for svd_tol in svd_tol_values:
            # Update svd_tol value
            self.svd_tol = svd_tol

            self.fitter = fit.FitRoutine(data=self.data,prior=self.prior,phys_pt_data=self.phys_point_data,model_info=self.model_info, discard_cov=self.discard_cov, svd_test=self.svd_test, svd_tol=self.svd_tol)
            info = self.fitter.fit_info
            chi2.append(info['chi2/df'])
            q.append(info['Q'])

            # Run extrapolation
            fit_out = self.fitter.fit
            # mass_post = fit_out.p['m_{particle,0}']
            extrapolation = self.extrapolation(observables=['mass'])

            # Store results' means and standard deviations
            for particle in self.model_info['particles']:
                results[particle]['mean'].append(extrapolation[particle]['mass'].mean)
                results[particle]['std'].append(extrapolation[particle]['mass'].sdev)

        # Generate plots for each particle
        for particle, values in results.items():
            plt.figure(figsize=(8,6))
            plt.errorbar(svd_tol_values, values['mean'], yerr=values['std'], label=particle,fmt='o', alpha=0.6)
            plt.xlabel('svd_tol')
            plt.ylabel('Extrapolated Mass')
            plt.title(f"Results for {particle}")
            plt.legend()
            plt.grid(True, which="both", ls="--")
            plt.xscale('log')
            plt.tight_layout()
            plt.show()

        # Plot q-values
        plt.figure(figsize=(8,6))
        plt.plot(svd_tol_values, q, label='q-value')
        plt.xlabel('svd_tol')
        plt.ylabel('q-value')
        plt.xscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()

        # Plot chi2
        plt.figure(figsize=(8,6))
        plt.plot(svd_tol_values, chi2, label='chi2')
        plt.xlabel('svd_tol')
        plt.ylabel('chi2')
        plt.xscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.show()



    def __str__(self):
        # output = "Model: %s" %(self.model_info['name'])
        output = ""
        if self.extrapolate:
            output += '\n---\n'
            output+= 'Extrapolation:'
            output += '\n'
            output+= str(self.format_extrapolation(observables=['mass']))
        # output += '\n---\n'
        output += '\nError Budget:\n'
        for particle in self.model_info['particles']:
            max_len = np.max([len(key) for key in self.error_budget[particle].keys()])
        # else:
        #     max_len = np.max([len(key) for key in self.error_budget['sigma'].keys()])

        for particle in [p for p in self.model_info['particles']]:
            output += particle
            output += '\n'
            for key in dict(sorted(self.error_budget[particle].items(), key=lambda item: item[1], reverse=True)):
                output += '  '
                output += key.ljust(max_len+1)
                output += '{: .1%}\n'.format((self.error_budget[particle][key]/self.extrapolated_mass[particle].sdev)**2).rjust(7)
        if self.verbose:
            fit_str = self.fit.format(maxline=True,pstyle='vv')
        else:
            fit_str = self.fit.format(pstyle='m')
        output+= str(fit_str)
        return output 
    
    def extrapolation(self,observables=None,p=None,data=None):
        """returns extrapolated mass and/or sigma term"""
        if data is None:
            data = self._phys_point_data
        if p is None:
            p = self.posterior
        _extrapolation = self.extrapolation(observables,p,data)
        return _extrapolation
    
    def format_extrapolation(self, observables=None):
        """formats the extrapolation dictionary"""
        extrapolation_data = self.extrapolation(observables=observables)
        pdg_mass = {
            'xi': gv.gvar(1314.86,20),
            'xi_st': gv.gvar(1531.80,32),
            'lambda': gv.gvar(1115.683,6),
            'sigma': gv.gvar(1192.642,24),
            'sigma_st': gv.gvar(1383.7,1.0)
        }
        output = ""
        for particle, data in extrapolation_data.items():
            output += f"Particle: {particle}\n"
            for obs, val in data.items():
                measured = pdg_mass[particle]
                output += f"{obs}: {val} [PDG: {measured}]\n"
            output += "---\n"

        return output
    
    def extrapolation(self,observables, p=None, data=None, xdata=None):
        '''chiral extrapolations of baryon mass data using the Feynman-Hellmann theorem to quantify pion mass and strange quark mass dependence of baryon masses. Extrapolations are to the physical point using PDG data. 
        
        Returns(takes a given subset of observables as a list):
        - extrapolated mass (meV)
        - pion sigma term 
        - barred pion sigma term / M_B'''

        if p is None:
            p = self.posterior
        if data is None:
            data = self._phys_pt_data
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
        }


        
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
                    # extrapolate hyperon mass to the physical point 
                    if obs == 'mass':
                        if self.model_info['units'] == 'lattice':
                            for particle in self.model_info['particles']:
                                output+= model_instance.fitfcn(p=p) * self.phys_point_data['hbarc']
                        if self.model_info['units'] == 'fpi':
                            output+= model_instance.fitfcn(p=p) * self.phys_point_data['lam_chi']
                        if self.model_info['units'] == 'phys':
                            output+= model_instance.fitfcn(p=p) 
                    results[particle][obs] = output
        return results
    



    @property
    def error_budget(self):
        '''
        useful for analyzing the impact of the a priori uncertainties encoded in the prior
        '''
        return self._get_error_budget()

    def _get_error_budget(self, verbose=False,**kwargs):
        '''
        list of expansion parameters associated with each hyperon,
        calculates a parameter's relative contribution to the total error.
        Types:
        - statistics 
        - chiral model
        - lattice spacing (discretization)
        - physical point input
        '''
        output = None
        strange_keys = [
        'd_{lambda,s}','d_{sigma,s}', 'd_{sigma_st,s}', 'd_{xi,s}', 'd_{xi_st,s}',
        'd_{lambda,as}', 'd_{lambda,ls}', 'd_{lambda,ss}', 'd_{sigma,as}', 'd_{sigma,ls}', 'd_{sigma,ss}',
        'd_{sigma_st,as}', 'd_{sigma_st,ls}', 'd_{sigma_st,ss}', 'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}',
        'd_{xi_st,as}', 'd_{xi_st,ls}', 'd_{xi_st,ss}']
        
        chiral_keys = [
        's_{lambda}', 's_{sigma}', 's_{sigma,bar}', 's_{xi}', 's_{xi,bar}', 
        'g_{lambda,sigma}', 'g_{lambda,sigma_st}', 'g_{sigma,sigma}', 'g_{sigma_st,sigma}', 
        'g_{sigma_st,sigma_st}', 'g_{xi,xi}', 'g_{xi_st,xi}', 'g_{xi_st,xi_st}', 'b_{lambda,4}', 
        'b_{sigma,4}', 'b_{sigma_st,4}', 'b_{xi,4}', 'b_{xi_st,4}', 'a_{lambda,4}', 'a_{sigma,4}', 
        'a_{sigma_st,4}', 'a_{xi,4}', 'a_{xi_st,4}'] 
        
        disc_keys = [
        'd_{lambda,a}','d_{lambda,aa}','d_{lambda,al}', 
        'd_{sigma,a}', 'd_{sigma,aa}', 'd_{sigma,al}',
        'd_{sigma_st,a}', 'd_{sigma_st,aa}', 'd_{sigma_st,al}', 
        'd_{xi,a}', 'd_{xi,aa}', 'd_{xi,al}'
        'd_{xi_st,a}'  'd_{xi_st,aa}', 'd_{xi_st,al}']

        stat_keys_y = [
            'm_{lambda,0}', 'm_{sigma,0}', 'm_{sigma_st,0}', 'm_{xi,0}', 'm_{xi_st,0}'
        ]
        phys_keys = list(self.phys_point_data)
        stat_key = 'lam_chi'# Since the input data is correlated, only need a single variable as a proxy for all

        if verbose:
            if output is None:
                output = ''

            inputs = {}
            inputs.update({str(param)+' [disc]': self._input_prior[param] for param in disc_keys if param in self._input_prior})
            inputs.update({str(param)+' [xpt]': self._input_prior[param] for param in chiral_keys if param in self._input_prior})
            inputs.update({str(param)+ '[strange]': self._input_prior[param] for param in strange_keys if param in self._input_prior})
            inputs.update({str(param)+' [phys]': self.phys_point_data[param] for param in list(phys_keys)})
            inputs.update({'x [stat]' : self._input_prior[param] for param in stat_key if param in self._input_prior})
            inputs.update({'a [stat]' : self._input_prior['eps2_a'] })
            inputs.update({str(obs)+'[stat]' : self.fit.y[obs] for obs in self.fit.y})
            # inputs.update({'y [stat]' : self._input_prior[param] for param in stat_keys_y if param in self.fit.y})
            # , 'y [stat]' : self.fitter.fit.y})

            if kwargs is None:
                kwargs = {}
            kwargs.setdefault('percent', False)
            kwargs.setdefault('ndecimal', 10)
            kwargs.setdefault('verify', True)

            print(gv.fmt_errorbudget(outputs=self.extrapolated_mass, inputs=inputs, verify=True))
        else:
            if output is None:
                output = {}
            for particle in self.model_info['particles']:
                output[particle] = {}
                output[particle]['disc'] = self.extrapolated_mass[particle].partialsdev(
                            [self.prior[key] for key in disc_keys if key in self.prior])
                
                output[particle]['chiral'] = self.extrapolated_mass[particle].partialsdev(
                            [self.prior[key] for key in chiral_keys if key in self.prior])
                
                output[particle]['pp'] = self.extrapolated_mass[particle].partialsdev(
                            [self.phys_point_data[key] for key in phys_keys if key in phys_keys])
                
                output[particle]['stat'] = self.extrapolated_mass[particle].partialsdev(
                    [self.fit.prior[key] for key in ['eps2_a'] if key in self.fit.prior]+
                    [self._get_prior(stat_key)] 
                    + [self.fit.y[particle]]
                )
            

        return output

    
    def fitfcn(self, posterior=None, data=None, particle=None,xdata=None):
        '''returns resulting y_fit of hyperon extrapolation function'''
        output = {}
        if data is None:
            data = copy.deepcopy(self.phys_point_data)
        if posterior is None:
            posterior = copy.deepcopy(self.posterior)

            # p = {}
            # p.update(self.posterior)
        # if data is None:
        #     data = self.phys_point_data
        # p.update(data)
        model_array, model_dict = self.fitter._make_models(model_info=self.model_info)
        for mdl in model_array:
            part = mdl.datatag
            output[part] = mdl.fitfcn(p=posterior,data=data,xdata=xdata)
        if particle is None:
            return output
        return output[particle]

    @property
    def extrapolated_mass(self):
        '''returns mass of a hyperon extrapolated to the physical point'''
        output = {}
        mdls = fit.FitRoutine(data=self.data,prior=self.prior,model_info=self.model_info,phys_pt_data=self._phys_point_data,discard_cov=self.discard_cov,svd_tol=self.svd_tol,svd_test=self.svd_test,decorr_scale=self.decorr_scale)
                              
        output = mdls.get_fitfcn(p=self.posterior, data=self.phys_point_data)

        # if particle is None:
        return output
        # return output[particle]

    
    def _extrapolate_to_ens(self,ens=None, phys_params=None,observable=None):
        if phys_params is None:
            phys_params = []

        extrapolated_values = {}
        for j, ens_j in enumerate(self.ensembles):
            posterior = {}
            xdata = {}
            if ens is None or (ens is not None and ens_j == ens):
                for param in self.fit.p:
                    shape = self.fit.p[param].shape
                    if param in phys_params:
                        # posterior[param] = self.phys_point_data[param] / self.phys_point_data['hbarc']
                        posterior[param] = self.phys_point_data[param]       
                    elif shape == ():
                            posterior[param] = self.fit.p[param]
                    else:
                        posterior[param] = self.fit.p[param][j]
                if 'eps_pi' in phys_params:
                    xdata['eps_pi'] = self.phys_point_data['m_pi'] / self.phys_point_data['lam_chi']
                if 'mpi' in phys_params:
                    xdata['m_pi'] = self.phys_point_data['m_pi']
                if 'd_eps2_s' in phys_params:
                    xdata['d_eps2_s'] = (2 *self.phys_point_data['m_k']**2 - self.phys_point_data['m_pi']**2)/ self.phys_point_data['lam_chi']**2
                if 'eps2_a' in phys_params:
                    xdata['eps_a'] = 0
                if 'lam_chi' in phys_params:
                    xdata['lam_chi'] = self.phys_point_data['lam_chi']
                if ens is not None:
                    return self.fitfcn(posterior=posterior, data={},xdata=xdata,particle=observable)
                extrapolated_values[ens_j] = self.fitfcn(posterior=posterior, data={}, xdata=xdata,particle=observable)
        return extrapolated_values
    
    def shift_latt_to_phys(self, ens=None,eps=None, phys_params=None,observable=None,debug=None):
        '''
        shift fitted values of the hyperon on each lattice to a 
        new sector of parameter space in which all parameters are fixed except
        the physical parameter of interest,eg. eps2_a (lattice spacing), eps_pi (pion mass),
        etc.
        Since we have extrapolation to lam_chi as fcn of eps_pi, eps2_a, we use lattice value of lam_chi for analyis of masses. To then call this function when plotting extrapolation fit vs. one of these phys. parameters, can use fit to lam_chi(eps_pi,eps2_a)
        '''
        value_shifted = {}
        for j, ens_j in enumerate(self.ensembles):
            if ens is None or ens_j == ens:
                y_fit = self.fit.y[observable]
                value_latt =  y_fit[j]
                value_fit = self._extrapolate_to_ens(ens_j,observable=observable)
                value_fit_phys = self._extrapolate_to_ens(ens_j, phys_params,observable=observable)
               
                value_shifted[ens_j] = value_latt + value_fit_phys - value_fit
                if debug:
                    print(value_latt,"latt")
                    print(value_fit,"fit")
                    print(value_fit_phys,"phys")
                if ens is not None:
                    return value_shifted[ens_j]
        return value_shifted

    @property
    def fit_keys(self):
        output = {}
        keys1 = list(self._input_prior.keys())
        keys2 = list(self.fitter.fit.p.keys())
        output = np.intersect1d(keys1, keys2)
        return output

    @property
    def model(self):
        return self.model_info['name']

    @property
    def phys_point_data(self):
        '''returns dict of physial constants from the PDG'''
        return self._get_phys_point_data()

    # need to convert to/from lattice units
    def _get_phys_point_data(self, parameter=None):
        if parameter is None:
            return copy.deepcopy(self._phys_point_data)
        return self.phys_point_data.get(parameter,None)

    @property
    def posterior(self):
        '''Returns dictionary with keys fit parameters'''
        return self._get_posterior()

    def _get_posterior(self,param=None):
        if param == 'all':
            return self.fitter.fit.p
        if param is not None:
            return self.fitter.fit.p[param]
        output = {}
        for param_ in self._input_prior:
            if param_ in self.fitter.fit.p:
                output[param_] = self.fitter.fit.p[param_]
        return output

    @property
    def prior(self):
        return self._get_prior()

    def _get_prior(self, param=None):
        output = {}
        if param is None:
            output = {param : self.fitter.fit.prior[param] for param in self.fit_keys}
        elif param == 'all':
            output = self.fitter.fit.prior
        elif isinstance(param, list):  # New condition to handle lists of params
            output = {p: self.fitter.fit.prior[p] for p in param if p in self.fitter.fit.prior}
        else:
            output = self.fitter.fit.prior[param]

        return output
    
    def plot_params(self, observables, xparam=None, show_plot=None, eps=None,units=None):
        '''plot unshifted masses on each ensemble vs physical param. of interest eg. 
        eps2_a, eps_pi'''

        if isinstance(observables, str):
            observables = [observables]
        if xparam is None:
            xparam = 'eps2_a'
        colormap = {
            'a06' : 'purple',
            'a09' : 'blue',
            'a12' : 'green',
            'a15' : 'red',
        }
        x = {}
        y = {observable: {} for observable in observables}
        # y = {observable:{}}
        baryon_latex = {
            'sigma': '\Sigma',
            'sigma_st': '\Sigma^*',
            'xi': '\Xi',
            'xi_st': '\Xi^*',
            'lambda': '\Lambda'
        }
        fig, axs = plt.subplots(len(observables), 1, figsize=(12, 10))

        for idx, observable in enumerate(observables):
            ax = axs[idx] if len(observables) > 1 else axs
            for i in range(len(self.ensembles)):
                for j, param in enumerate([xparam, observable]):
                    if param in baryon_latex.keys():
                        if units == 'gev':
                            value = self.fit.y[param][i] / 1000
                        else:
                            value = self.fit.y[param][i]
                        latex_baryon = baryon_latex[param]
                        if units== 'gev':
                            label = f'$m_{{{latex_baryon}}}$(GeV)'
                        else:
                            if eps: 
                                label = f'$\\frac{{M_{latex_baryon}}}{{\\Lambda_{{\\chi}}}}$' 

                            else:
                                label = f'$m_{{{latex_baryon}}}$ (meV)'
                    if param == 'eps2_a':
                        value = self.data['eps2_a'][i] 
                        label = '$\epsilon_a^2$'
                    elif param == 'eps_pi':
                        value = self.fit.p['eps_pi'][i]
                        label = '$\epsilon_\pi$'

                    elif param == 'mpi_sq':
                        if units == 'gev':
                            value = (self.data['m_pi'][i])**2 /100000 #gev^2
                            label = '$m_\pi^2(GeV^2)$'
                        else:
                            value = (self.data['m_pi'][i])**2
                            label = '$m_\pi^2(MeV^2)$'
                    if j == 0:
                        x[i] = value
                        xlabel = label
                    else:
                        y[observable][i] = value
                        ylabel = label

            added_labels = set()

            for i in range(len(self.ensembles)):
                C = gv.evalcov([x[i], y[observable][i]])
                eVe, eVa = np.linalg.eig(C)
                color_key = self.ensembles[i][:3]
                color = colormap[color_key]
                label = f'{color_key.lower()}'

                for e, v in zip(eVe, eVa.T):
                    ax.plot([gv.mean(x[i])-1*np.sqrt(e)*v[0], 1*np.sqrt(e)*v[0] + gv.mean(x[i])],
                            [gv.mean(y[observable][i])-1*np.sqrt(e)*v[1], 1*np.sqrt(e)*v[1] + gv.mean(y[observable][i])],
                            alpha=1.0, lw=2, color=color)

                    if label not in added_labels:
                        ax.plot(gv.mean(x[i]), gv.mean(y[observable][i]), 
                                marker='o', mec='w', markersize=8, zorder=3, color=color, label=label)
                        added_labels.add(label)
                    else:
                        ax.plot(gv.mean(x[i]), gv.mean(y[observable][i]), 
                                marker='o', mec='w', markersize=8, zorder=3, color=color)

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=14, bbox_to_anchor=(1.05,1), loc='upper left')
            ax.grid(True)
            ax.set_xlabel(xlabel, fontsize=16)
            ax.set_ylabel(ylabel, fontsize=16)
            # This is inside the for loop where observable is the current baryon being plotted
            if eps:
                phys_point_observable = self._get_phys_point_data(parameter='eps_'+observable)
            else:
                phys_point_observable = self._get_phys_point_data(parameter='m_'+observable)
                        
            if observable == 'xi':
                marker_style = 'o'
                label = 'xi'
            elif observable == 'xi_st':
                marker_style = '^'
                label = 'xi_st'
            else:
                marker_style = 's'
                label = observable
                        
            ax.plot(0, gv.mean(phys_point_observable), marker=marker_style, color='black', markersize=10, zorder=4)
            ax.axvline(0, ls='--', color='black')


        plt.tight_layout()
        if show_plot:
            plt.show()
        plt.close()
        return fig

    
    def plot_params_fit(self,param,observable=None,shift=None,eps=None):
        fig, ax = plt.subplots(figsize=(10,8))
        colormap = {'a06' : 'purple', 'a09' : 'blue', 'a12' : 'green', 'a15' : 'red'}
        
        latt_spacing = {'06': gv.gvar('0.04009(18)'), '09': gv.gvar('0.09209(28)'), 
                        '12': gv.gvar('0.17016(37)'), '15': gv.gvar('0.25206(32)'), 
                        '00': gv.gvar('0(0)')}
        xi = {}
        
        for j, xx in enumerate(reversed(latt_spacing)): 
            min_max = lambda mydict : (gv.mean(np.nanmin([mydict[key] for key in mydict.keys()])), 
                                       gv.mean(np.nanmax([mydict[key] for key in mydict.keys()])))   
            phys_data = self.phys_point_data
            phys_data['eps2_a'] = latt_spacing[xx]
            if param == 'a':
                eps2_a_arr = [self.data['eps2_a']] 
                xi['eps2_a'] = np.linspace(0, gv.mean(np.max(eps2_a_arr)))
                x_fit = xi['eps2_a']
            if param == 'epi':
                eps_pi_arr = [self.data['eps_pi']] 
                xi['eps_pi'] = np.linspace(0, gv.mean(np.max(eps_pi_arr)))
                x_fit = xi['eps_pi']
            if param == 'mpi_sq':
                mpi_sq_arr = [self.data['m_pi']] 
                xi['mpi_sq'] = np.linspace(0.000,gv.mean(np.max(mpi_sq_arr)))
                x_fit = xi['mpi_sq']
        y_fit = {observable: self.fitter.get_fitfcn(data=phys_data, particle=observable, xdata=xi)}

        # y_fit = {'xi': self.fitter.get_fitfcn(data=phys_data, particle='xi', xdata=xi),
        #         'xi_st': self.fitter.get_fitfcn(data=phys_data, particle='xi_st', xdata=xi),
        #         'lambda': self.fitter.get_fitfcn(data=phys_data, particle='lambda', xdata=xi),
        #         'sigma': self.fitter.get_fitfcn(data=phys_data, particle='sigma', xdata=xi),
        #         'sigma_st': self.fitter.get_fitfcn(data=phys_data, particle='sigma_st', xdata=xi)}
        pm = lambda g, k : gv.mean(g) + k * gv.sdev(g)
        ax.fill_between(pm(x_fit, 0), pm(y_fit[observable], -1), pm(y_fit[observable], +1), 
                        facecolor='None', edgecolor='k', alpha=0.6, hatch='/')

        added_labels = set()
        x = {}
        y = {}
        baryon_latex = {'sigma': '\Sigma', 'sigma_st': '\Sigma^*', 'xi': '\Xi', 'xi_st': '\Xi^*', 'lambda': '\Lambda'}
        
        for ens in self.ensembles:
            if param == 'a':
                x = self.data['eps2_a']
                if shift is not None:
                    y[ens] = shift
                else:
                    if eps:
                        # y[ens] = 
                        y[ens] = self.shift_latt_to_phys(ens=ens,eps=True, phys_params=['eps_pi','d_eps2_s','m_pi','lam_chi'], observable=observable)
                    else:
                        y[ens] = self.shift_latt_to_phys(ens=ens, phys_params=['eps_pi','d_eps2_s','m_pi','lam_chi'], observable=observable)
                                                                               
                xlabel = r'$\epsilon_a^2$'
            elif param == 'epi':
                x = self.data['eps_pi']
                if shift is not None:
                    y[ens] = shift
                else:
                    if eps:
                        y[ens] = self.shift_latt_to_phys(ens=ens,eps=True, phys_params=['eps2_a','d_eps2_s','m_pi','lam_chi'], observable=observable)

                    else:
                        y[ens] = self.shift_latt_to_phys(ens=ens, phys_params=['eps2_a','d_eps2_s','m_pi','lam_chi'], observable=observable)
                xlabel = r'$\epsilon_\pi$'
            elif param == 'mpi_sq':
                x = self.data['m_pi']
                if shift is not None:
                    y[ens] = shift
                else:
                    y[ens] = self.shift_latt_to_phys(ens=ens, phys_params=['eps2_a','d_eps2_s'], observable=observable)
                xlabel = r'$m_\pi^2$'

        for i,ens in enumerate(self.ensembles):
            C = gv.evalcov([x[i], y[ens]])
            eVe, eVa = np.linalg.eig(C)
            color_key = self.ensembles[i][:3]
            color = colormap[color_key]
            label = f'{color_key.lower()}'

            for e, v in zip(eVe, eVa.T):
                ax.plot([gv.mean(x[i])-1*np.sqrt(e)*v[0], 1*np.sqrt(e)*v[0] + gv.mean(x[i])],
                        [gv.mean(y[ens])-1*np.sqrt(e)*v[1], 1*np.sqrt(e)*v[1] + gv.mean(y[ens])],
                        alpha=1.0, lw=2.5, color=color)  

                if label not in added_labels:
                    ax.plot(gv.mean(x[i]), gv.mean(y[ens]), marker='o', mec='w', markersize=10, 
                            zorder=3, color=color, label=label)  
                    added_labels.add(label)
                else:
                    ax.plot(gv.mean(x[i]), gv.mean(y[ens]), marker='o', mec='w', markersize=10, 
                            zorder=3, color=color)     

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        ax.legend(by_label.values(), by_label.keys(), fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
        ax.set_xlabel(xlabel, fontsize = 20)
        ax.set_title("Model: %s" %(self.model_info['name']), fontsize=20)  
        
        phys_point_observable = self._get_phys_point_data(parameter='m_'+observable)  
        phys_point_xparam = 0.0
        if observable in baryon_latex.keys():
            if eps:
                phys_point_observable = self._get_phys_point_data(parameter='eps_'+observable) 
            else:
                phys_point_observable = self._get_phys_point_data(parameter='m_'+observable)  
        phys_point_xparam = 0.0
        if observable in baryon_latex.keys():
            latex_baryon = baryon_latex[observable]
            if eps:
                label = f'$\\frac{{M_{latex_baryon}}}{{\\Lambda_{{\\chi}}}}$' 
            else:
                label = f'$M_{latex_baryon}$'
            # latex_baryon = baryon_latex[observable]
            # label = f'$M_{{{latex_baryon}}}$'
        ax.set_ylabel(label, fontsize = 20)

        # ax.plot(phys_point_xparam, gv.mean(phys_point_observable), marker='o', color='black', markersize=10, zorder=4)
        ax.axvline(phys_point_xparam, ls='--', color='black', label=label)
        
        return fig
    
    