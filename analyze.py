"""analyze resulting lsqfit object. 
SVD analysis
Error budget reporting 
Extrapolation to the physical point 
Plots of extrpolated fit
"""

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
import lsqfitics
import pandas as pd
from typing import List
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
mpl.rcParams['text.usetex'] = False

import fit_model
from fit_model import FitModel
import i_o
pdg_values = i_o._get_data_phys_point

def to_pandas(values,weights):
    df = pd.DataFrame(values).T

    # Drop the 'mass' key
    # df = df.applymap(lambda x: x.get('mass'))
    # Add a row with PDG values
    df.loc['PDG'] = [pdg_values.get('xi'), pdg_values.get('xi_st')]
    df['weight'] = [weights.get(model,0) for model in df.index]
    df = df.sort_values(by='weight', ascending=False)
    print(df.to_markdown())
    return df 

class SVD(FitModel):
    def __init__(self,fit_instance,**kwargs):
        data = fit_instance.data
        prior = fit_instance.prior
        phys_pt_data = fit_instance._phys_pt_data
        model_info = fit_instance.model_info
        strange = fit_instance.strange
        input_output = i_o.InputOutput(model_info['units'],strange=strange) 
        self.ensembles = input_output._get_ens

        # Call the constructor of the parent class using super()
        super().__init__(data, prior, phys_pt_data, model_info, strange, **kwargs)

    def svd_analysis(self,svd_tol_values:list):
        # Specify the svd_tol values to loop over
        if svd_tol_values is None:
            svd_tol_values = np.linspace(10e-6, 0.05, num=30)  #this for some reason is not working..
        else:
            svd_tol_values = np.linspace(svd_tol_values)

        chi2 = []
        q = []

        results = {particle: {'mean': [], 'std': []} for particle in self.model_info['particles']}

        for svd_tol in svd_tol_values:
            # Update svd_tol value
            self.svd_tol = svd_tol

            info = self.fit_instance.fit_info
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




class Analyze(FitModel):
    def __init__(self,fit_instance,**kwargs):
        self.fit_instance = fit_instance
        data = fit_instance.data
        prior = fit_instance.prior
        phys_pt_data = fit_instance._phys_pt_data
        model_info = fit_instance.model_info
        strange = fit_instance.strange
        input_output = i_o.InputOutput(model_info['units'],strange=strange) 
        self.ensembles = input_output._get_ens
        
        # Call the constructor of the parent class using super()
        super().__init__(data, prior, phys_pt_data, model_info, strange, **kwargs)

    # def __str__(self):
    #     output = ''
    #     output += '\n'
    #     output += print('Extrapolated mass: %s' % (self.extrapolation_out))
    #     output += '\n'
    #     output += self.fit_instance.fit

    #     return output 

    def _get_prior(self,param=None):
        output = {}
        if param is None:
            output = {param : self.prior[param] for param in self.fit_keys}
        elif param == 'all':
            output = self.prior
        elif isinstance(param, list):  # New condition to handle lists of params
            output = {p: self.prior[p] for p in param if p in self.prior}
        else:
            output = self.prior[param]

        return output

    def extrapolation_out(self,observables=None,p=None,data=None):
        """returns extrapolated mass and/or sigma term"""
        if data is None:
            data = self._phys_point_data
        if p is None:
            p = self.get
        _extrapolation = self.extrapolation(observables,p,data)
        return _extrapolation
        
    
    def format_extrapolation(self):
        """formats the extrapolation dictionary"""
        extrapolation_data = self.extrapolation(observables=['xi','xi_st'])
        print(extrapolation_data)
        pdg_mass = pdg_values
        output = ""
        for particle, data in extrapolation_data.items():
            output += f"Particle: {particle}\n"
            for obs, val in data.items():
                measured = pdg_mass[particle]
                output += f"{obs}: {val} [PDG: {measured}]\n"
            output += "---\n"

        return output
    
    def extrapolation(self,p=None, data=None, xdata=None):
        """perform extrapolations to the physical point using PDG data; this is the continuum, infinite volume limit where observables take their PDG values. 
        
        Returns(takes a given subset of observables as a list):
        - extrapolated mass (meV)
        """        
        if p is None:
            p = self.get_posterior
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
            'xi': fit_model.Xi,
            'xi_st': fit_model.Xi_st,
        }

        results = {}

        for particle in self.model_info['particles']:
            model_class = MULTIFIT_DICT.get(particle)
            if model_class is not None:
                model_instance = model_class(datatag=particle, model_info=self.model_info)
        
                results[particle] = {}
                # results = {}

                output = 0
                mass = 0
                # extrapolate hyperon mass to the physical point 
                if self.model_info['units'] == 'lattice':
                    for particle in self.model_info['particles']:
                        output+= model_instance.fitfcn(p=p) * self.phys_point_data['hbarc']
                if self.model_info['units'] == 'fpi':
                    output+= model_instance.fitfcn(p=p) * self.phys_point_data['lam_chi']
                if self.model_info['units'] == 'phys':
                    output+= model_instance.fitfcn(p=p) 
                results[particle] = output
        return results
    
    @property
    def error_budget(self):
        '''
        useful for analyzing the impact of the a priori uncertainties encoded in the prior on the resulting fit parameter for masses
        '''
        return self._get_error_budget()

    def _get_error_budget(self, verbose=False,**kwargs):
        '''
        list of independent parameters associated with each hyperon's mass expansion.
        calculates a parameter's relative contribution to the total error.
        Types:
        - statistics 
        - chiral model
        - lattice spacing (discretization)
        - physical point input
        - error contribution from each ensemble 
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
        phys_keys = list(self._phys_pt_data)
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

            print(gv.fmt_errorbudget(outputs=self.extrapolation, inputs=inputs, verify=True))
        else:
            # for ens in 
            extrapolated = self.extrapolation()
            print('e',extrapolated)
            if output is None:
                output = {}
            for particle in self.model_info['particles']:
                y_out = self.fit.p['m_{'+particle+',0}']
                output[particle] = {}
                output[particle]['disc'] = extrapolated[particle].partialsdev(
                            [self.prior[key] for key in disc_keys if key in self.prior])
                
                output[particle]['chiral'] = extrapolated[particle].partialsdev(
                            [self.prior[key] for key in chiral_keys if key in self.prior])
                
                output[particle]['pp'] = extrapolated[particle].partialsdev(
                            [self._phys_pt_data[key] for key in phys_keys if key in phys_keys])
                
                output[particle]['stat'] = extrapolated[particle].partialsdev(
                    [self.fit.prior[key] for key in ['eps2_a'] if key in self.fit.prior]+
                    [self._get_prior(stat_key)] 
                    + [self.fit.y[particle]]
                )
            

        return output
    
    def _get_multifit_inst(self):
    
        MULTIFIT_DICT = {
                'xi': fit_model.Xi,
                'xi_st': fit_model.Xi_st,
            }

        results = {}

        for particle in self.model_info['particles']:
            model_class = MULTIFIT_DICT.get(particle)
            if model_class is not None:
                model_instance = model_class(datatag=particle, model_info=self.model_info)

        return model_instance
    
    def fitfcn(self,posterior=None,data=None,particle=None,xdata=None):
        '''fetch fit function from `fit_model` to perform extrapolation to the physical pt. 
        Pass in physical point parameters 
        '''
        output = {}
        if data is None:
            data = copy.deepcopy(self.phys_point_data)
        if posterior is None:
            posterior = copy.deepcopy(self.posterior)

        model_array, model_dict = self._make_models(model_info=self.model_info)
        for mdl in model_array:
            part = mdl.datatag
            output[part] = mdl.fitfcn(p=posterior,data=data,xdata=xdata)
        if particle is None:
            return output
        return output[particle]
    
    def _extrapolate_to_ens(self,ens=None, phys_params=None,observable=None):
        """"""
        model_instance = self._get_multifit_inst()
        print(model_instance.fitfcn,'mdl_inst')
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
        shift resulting fit params (y data) of a hyperon on each lattice to a 
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
    


    
    

class ModelAverage(Analyze):
    """Consider multiple trunations of chiral models and average over the different models according to their calculated weights using the `lsqfitics` module"""
    def __init__(self,analysis,**kwargs):
        super().__init__(**kwargs)
        self.analysis = analysis 


    def model_average(self,particles):
        avg = {}
        avg_out = {}
        fit_collection = {}
        for mdl_key in self.models:
            xfa_instance= self.models[mdl_key]
            fit_out = xfa_instance.fit
            fit_collection[mdl_key] = fit_out
            for part in particles:
                if self.units == 'phys':
                    avg[part] = fit_out.p[f'm_{{{part},0}}']          
                if self.units =='fpi':
                    avg[part] = fit_out.p[f'm_{{{part},0}}'] * 4 *np.pi *gv.gvar('92.07(57)')          

        weights = lsqfitics.calculate_weights(fit_collection,'logGBF')
        # Sort weights dictionary by values in descending order
        weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)}
        for part in particles:
            avg_out[part] = lsqfitics.calculate_average(values=avg[part],weights=weights)

        return avg_out,weights



    


    