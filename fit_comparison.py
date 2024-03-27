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
import yaml

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
# from xpt.fit_analysis import Xpt_Fit_Analysis
import xi_fit as xi_fit
import i_o as i_o
import priors as priors
import lsqfitics

def get_data_and_prior_for_unit(unit,strange):
    prior = priors.get_prior(units=unit)
    input_output = i_o.InputOutput(units=unit, scheme='w0_imp', strange=strange, convert_data=False)
    
    data = input_output.perform_gvar_processing()
    new_prior = input_output.make_prior(data=data, prior=prior)
    
    if unit == 'fpi':
        phys_point_data = input_output.get_data_phys_point(fpi_units=True)
    else:
        phys_point_data = input_output.get_data_phys_point(fpi_units=False)
    
    return data, new_prior, phys_point_data

def plot_extrapolated_masses(extrapolated_values,decorr_scales):
    decorr_scales = list(extrapolated_values.keys())
    xi_masses = [extrapolated_values[scale]['xi']['mass'].mean for scale in decorr_scales]
    xi_errors = [extrapolated_values[scale]['xi']['mass'].sdev for scale in decorr_scales]

    xi_st_masses = [extrapolated_values[scale]['xi_st']['mass'].mean for scale in decorr_scales]
    xi_st_errors = [extrapolated_values[scale]['xi_st']['mass'].sdev for scale in decorr_scales]

    x = range(len(decorr_scales))

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot xi on the first subplot
    axs[0].errorbar(x, xi_masses, yerr=xi_errors, fmt='o', label='xi', capsize=5)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(de17corr_scales)
    axs[0].set_ylabel('Extrapolated Mass(MeV)')
    axs[0].set_title('Scale data decorrelation: xi')
    axs[0].legend()
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot xi_st on the second subplot
    axs[1].errorbar(x, xi_st_masses, yerr=xi_st_errors, fmt='^', label='xi_st', capsize=5)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(decorr_scales)
    axs[1].set_ylabel('Extrapolated Mass(MeV)')
    axs[1].set_title('Scale data decorrelation: xi_st')
    axs[1].legend()
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def plot_extrapolated_masses_lam(extrapolated_values,decorr_scales):
    decorr_scales = list(extrapolated_values.keys())
    xi_masses = [extrapolated_values[scale]['lambda']['mass'].mean for scale in decorr_scales]
    xi_errors = [extrapolated_values[scale]['lambda']['mass'].sdev for scale in decorr_scales]

    xi_st_masses = [extrapolated_values[scale]['sigma_st']['mass'].mean for scale in decorr_scales]
    xi_st_errors = [extrapolated_values[scale]['sigma_st']['mass'].sdev for scale in decorr_scales]

    sigma_masses = [extrapolated_values[scale]['sigma']['mass'].mean for scale in decorr_scales]
    sigma_errors = [extrapolated_values[scale]['sigma']['mass'].sdev for scale in decorr_scales]


    x = range(len(decorr_scales))

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # Plot xi on the first subplot
    axs[0].errorbar(x, xi_masses, yerr=xi_errors, fmt='o', label=None, capsize=5)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(decorr_scales)
    axs[0].set_ylabel('Extrapolated Mass(MeV)')
    axs[0].set_title('Scale data decorrelation: lambda')
    axs[0].legend()
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot xi_st on the second subplot
    axs[1].errorbar(x, xi_st_masses, yerr=xi_st_errors, fmt='^', label=None, capsize=5)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(decorr_scales)
    axs[1].set_ylabel('Extrapolated Mass(MeV)')
    axs[1].set_title('Scale data decorrelation: sigma_st')
    axs[1].legend()
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot xi_st on the second subplot
    axs[2].errorbar(x, sigma_masses, yerr=sigma_errors, fmt='^', label=None, capsize=5)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(decorr_scales)
    axs[2].set_ylabel('Extrapolated Mass(MeV)')
    axs[2].set_title('Scale data decorrelation: sigma')
    axs[2].legend()
    axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


def run_analysis(units,strange,test_mdl_key=None,compare_models=None,compare_scale=None,verbose=None):
    """run fits of all models or a single model"""

    # with open('../xpt/models.yaml', 'r') as f:
    with open('../xpt/models_test.yaml', 'r') as f:

        models = yaml.load(f, Loader=yaml.FullLoader)
    if strange == '2':
        system = 'xi'
        _models = models['models']['xi']
    if strange=='1':
        system = 'lam'
        _models = models['models']['lam']
    if strange=='0':
        system = 'proton'
        _models = models['models']['proton']
    discard_cov_opt = [True, False]
    model_data = {}

        
    # for unit in units:
    if compare_scale:
        extrapolated_vals = {}
        for decorr in ['full','partial','no']:

            data, new_prior, phys_point_data = i_o.get_data_and_prior_for_unit(unit=units, strange=strange, scheme='w0_imp', scale_correlation=decorr)
            print(data['units_MeV'])
    # if model_type == 'all':
            if test_mdl_key is not None:
                _model_info = _models[test_mdl_key].copy()
                # print(_model_info)
                _model_info['units'] = units
                updated_mdl_key = i_o.update_model_key_name(test_mdl_key, units)
                # for dc in discard_cov_opt:
                xfa_instance = Xpt_Fit_Analysis(data=data,
                                                    prior=new_prior,
                                                    model_info=_model_info,
                                                    phys_pt_data=phys_point_data,
                                                    strange=strange,
                                                    units=units,
                                                    extrapolate=True,
                                                    discard_cov=False,
                                                    decorr_scale=decorr,
                                                    verbose=False,
                                                    svd_test=False,
                                                    svd_tol=None)
                config_key = f"{updated_mdl_key}_decorr_scale-{decorr}"
                model_data[config_key] = xfa_instance  
                extrapolated_vals[decorr] = xfa_instance.extrapolation(observables=['mass'])

        for decorr,value in extrapolated_vals.items():
            if system == 'xi':
                xi_mass = value['xi']['mass']
                xi_st_mass = value['xi_st']['mass']
                print(f"Extrapolated mass for decorr_scale={decorr}: xi: {xi_mass}, xi_st: {xi_st_mass}")
                plot_extrapolated_masses(extrapolated_vals,decorr_scales=decorr)

            if system == 'lam':
                lambda_mass = value['lambda']['mass']
                sigma_mass = value['sigma']['mass']
                sigma_st_mass = value['sigma_st']['mass']

                print(f"Extrapolated mass for decorr_scale={decorr}: lam: {lambda_mass}, sigma_st: {sigma_st_mass}, sigma: {sigma_mass}")
                plot_extrapolated_masses_lam(extrapolated_vals,decorr_scales=decorr)

            

                

# Calling the function

    for key, value in model_data.items():
        # Extract the unit from the key assuming the unit was appended to the original key.
        unit_from_info = value.model_info['units']
        print(f"Results for {i_o.get_unit_description(unit_from_info)}:")
        print(key)
        print(value)
    if system == 'xi':
        fig0= xfa_instance.plot_params(xparam='eps_pi',observables=['xi','xi_st'],eps=False,show_plot=True)
        fig1= xfa_instance.plot_params(xparam='eps2_a',observables=['xi','xi_st'],eps=False,show_plot=True)
    if system == 'lam':
        fig0= xfa_instance.plot_params(xparam='eps_pi',observables=['lambda','sigma_st','sigma'],eps=False,show_plot=True)
        fig1= xfa_instance.plot_params(xparam='eps2_a',observables=['lambda','sigma_st','sigma'],eps=False,show_plot=True)



                # fig1= xfa_instance.plot_params_fit(param='a',observable='xi',eps=True)
                # fig2=xfa_instance.plot_params_fit(param='a',observable='xi_st',eps=True)
                # fig3=xfa_instance.plot_params_fit(param='epi',observable='xi_st',eps=True)
                # fig4=xfa_instance.plot_params_fit(param='epi',observable='xi',eps=True)


                # timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
                # pdf_pages = PdfPages(f'model_plots_single_{system}_{timestamp}.pdf')
                # pdf_pages.savefig(fig1)
                # pdf_pages.savefig(fig2)
                # pdf_pages.savefig(fig3)
                # pdf_pages.savefig(fig4)

                # plt.close(fig1)
                # plt.close(fig2)
                # plt.close(fig3)
                # plt.close(fig4)

    # else:
    #     for mdl_key in _models:
    #         _model_info = _models[mdl_key].copy()
    #         _model_info['units'] = units
    #         updated_mdl_key = i_o.update_model_key_name(mdl_key, units)
    #         for dc in discard_cov_opt:
    #             xfa_instance = Xpt_Fit_Analysis(data=data,
    #                                             prior=new_prior,
    #                                             model_info=_model_info,
    #                                             phys_pt_data=phys_point_data,
    #                                             units=units,
    #                                             extrapolate=True,
    #                                             discard_cov=dc,
    #                                             verbose=False,
    #                                             svd_test=False,
    #                                             svd_tol=None)
    #             config_key = f"{updated_mdl_key}_discardcov-{dc}"
    #             model_data[config_key] = xfa_instance  

    # Print the results
    if verbose:
        for key, value in model_data.items():
            # Extract the unit from the key assuming the unit was appended to the original key.
            unit_from_info = value.model_info['units']
            print(f"Results for {i_o.get_unit_description(unit_from_info)}:")
            print(key)
            print(value)

    if compare_models:
        compare_ = ModelComparsion(models=model_data,units=units)
        if strange == '2':
            particles = ['xi','xi_st']
        else:
            particles=['lambda','sigma_st','sigma']
        compare_.compare_models(particles=particles)
        mdl_avg = compare_.model_average(particles=particles)
        weights_dict = mdl_avg[1]
        top_model_key = max(weights_dict, key=weights_dict.get)
        truncated_top_model_key = top_model_key.split('_fpi')[0]
        print(f"Top weighted model key: {top_model_key}")
        print(f"Weight: {weights_dict[top_model_key]}")
        compare_.model_plots(system=system)

        # Extract the top model instance and its fit result
        top_model_instance = model_data[top_model_key]
        fit_result = top_model_instance.fit 

        # Print the fit result of the top model
        print(f"Fit Result for {top_model_key}:")
        print(fit_result)
        compare_.model_plots(system=system)


class ModelComparsion:
    '''Final analysis class. Generates a pdf with the model average and plots for each model'''
    def __init__(self,
                models:dict,
                units:str,
                **kwargs):
        self.models = models
        self.units = units
    
    def physical_points(self):
        physical_points = {
            'lambda' : gv.gvar(1115.683, 0.006),
            'sigma' : np.mean([gv.gvar(g) for g in ['1189.37(07)', '1192.642(24)', '1197.449(30)']]),
            'sigma_st' : np.mean([gv.gvar(g) for g in ['1382.80(35)', '1383.7(1.0)', '1387.2(0.5)']]),
            'xi' : np.mean([gv.gvar(g) for g in ['1314.86(20)', '1321.71(07)']]),
            'xi_st' : np.mean([gv.gvar(g) for g in ['1531.80(32)', '1535.0(0.6)']]),
            'proton' : gv.gvar(938.272,.0000058),
            'lam_chi' : 4 *np.pi *gv.gvar('92.07(57)')
        }
        return physical_points

    def add_fit_results_figure(self, fit_results, title):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('off')
        results_text = str(fit_results) #
        wrapper = textwrap.TextWrapper(width=80)
        results_text = wrapper.fill(results_text)
        textbox = plt.text(0.5, 0.5, results_text, fontsize=8, ha='center', va='center', wrap=True)
        textbox.set_bbox(dict(facecolor='white', alpha=1, edgecolor='black', boxstyle="round,pad=1"))
        # plt.title(title)
        plt.text(0.1, 0.5, title, rotation='vertical', fontsize=16, ha='center', va='center')
        return fig

    def model_plots(self,system=None,single_mdl=None):

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        pdf_pages = PdfPages(f'model_plots_{system}_{timestamp}.pdf')
        # Add model_average output.
        if system == 'xi':
            particles = ['xi','xi_st']
        else:
            particles=['lambda','sigma_st','sigma']
        avg_out, weights = self.model_average(particles=particles)

        avg_out_str = '\n'.join([f"{k}: {v}" for k, v in avg_out.items()])
        weights_str = '\n'.join([f"{k}: {v}" for k, v in weights.items()])
        avg_and_weights_str = f"Model averages:\n{avg_out_str}\n\nWeights:\n{weights_str}"
        fig_avg = plt.figure(figsize=(10, 10))
        plt.text(0.5, 0.5, avg_and_weights_str, fontsize=16, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5, boxstyle="round,pad=1"))
        plt.axis('off')
        pdf_pages.savefig(fig_avg, bbox_inches='tight')
        plt.close(fig_avg)

        fig_comp = self.compare_models(particles)
        pdf_pages.savefig(fig_comp, bbox_inches='tight')
        plt.close(fig_comp)
        # print(self.models)
        if single_mdl is not None:
            xfa_instance = self.models[single_mdl]
        else:
            for mdl_key in self.models:
                xfa_instance = self.models[mdl_key]

                # fit_out = xfa_instance.fit
                
                # fit_results_fig = self.add_fit_results_figure(fit_out, title='Model:'+mdl_key)
                # pdf_pages.savefig(fit_results_fig)
                # plt.close(fit_results_fig)
                plt.figure(figsize=(10, 10))  # create a new figure with specific size
                plt.text(0, 0, str(xfa_instance), fontsize=8)  # add text to figure
                plt.axis('off')  # hide axes
                pdf_pages.savefig()  # save current figure to pdf
                plt.close()  # close current figure
                if system == 'xi':
                    fig1= xfa_instance.plot_params_fit(param='a',observable='xi',eps=False)
                    fig2=xfa_instance.plot_params_fit(param='a',observable='xi_st',eps=False)
                    pdf_pages.savefig(fig1)
                    pdf_pages.savefig(fig2)
                    plt.close(fig1)
                    plt.close(fig2)
                else:
                    fig1= xfa_instance.plot_params_fit(param='a',observable='lambda',eps=False)
                    fig2=xfa_instance.plot_params_fit(param='a',observable='sigma',eps=False)
                    fig3=xfa_instance.plot_params_fit(param='a',observable='sigma_st',eps=False)
                    pdf_pages.savefig(fig1)
                    pdf_pages.savefig(fig2)
                    pdf_pages.savefig(fig3)
                    plt.close(fig1)
                    plt.close(fig2)
                    plt.close(fig3)
            if system == 'xi':
                fig_mpi = xfa_instance.plot_params(xparam='mpi_sq',observables=['xi','xi_st'],show_plot=True)
                fig_eps = xfa_instance.plot_params(xparam='eps2_a',observables=['xi','xi_st'],show_plot=True,eps=False)
                pdf_pages.savefig(fig_mpi)
                pdf_pages.savefig(fig_eps)
                plt.close(fig_mpi)
                plt.close(fig_eps)

            pdf_pages.close()

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

    def compare_models(self,particles=None):
        models = []
        extrapolated_masses = {particle: [] for particle in particles}
        q = []
        chi2_values_ = []
        weights = []
        _, weight_dict = self.model_average(particles)
        for mdl_key in self.models:
            weight_value = weight_dict[mdl_key]
            weights.append(weight_value)  # assuming th
            xfa_instance = self.models[mdl_key]
            mass = xfa_instance.extrapolation(observables=['mass'])
            # print(mass)
            models.append(mdl_key)
            info = xfa_instance.fit_info
            chi2_ = info['chi2/df']
            q_ =  info['Q']
            
            for particle in particles:
                extrapolated_mass = mass[particle]['mass']
                extrapolated_masses[particle].append(extrapolated_mass)
            chi2_values_.append(chi2_)
            q.append(q_)
        chi2_values = np.array(chi2_values_)
        q_values = np.array(q)

        fig, axs = plt.subplots(nrows=len(particles), ncols=3, figsize=(12, 6*len(particles)))
        # Create y values for scatter plot (this is just a range of integers)
        y_values = range(len(models))

        for idx, particle in enumerate(particles):
            physical_point_ = self.physical_points()
            physical_point = physical_point_[particle]
            means  = [gv.mean(val) for val in extrapolated_masses[particle]]
            stddevs = [gv.sdev(val) for val in extrapolated_masses[particle]]

            # Plot the extrapolated masses for each particle with error bars
            scatter = axs[idx, 0].errorbar(means, y_values, xerr=stddevs, fmt='o', alpha=0.6)
            axs[idx, 0].axvline(x=gv.mean(physical_point), color='r', linestyle='--')
            axs[idx, 0].set_xlabel('Extrapolated Mass')
            axs[idx, 0].set_ylabel('Models')
            axs[idx, 0].set_yticks(y_values)
            axs[idx, 0].set_yticklabels(models)
            axs[idx, 0].set_xlim([min(means)-max(stddevs)*1.5, max(means)+max(stddevs)*1.5])  # adjust this for desired zoom
            axs[idx, 0].set_title(particle)
            # axs[idx, 0].legend()

            axs[idx, 1].scatter(chi2_values, y_values, alpha=0.6)
            axs[idx, 1].set_xlabel('Chi2/df')
            axs[idx, 1].set_yticks([])  

            # Weights subplot
            axs[idx, 2].barh(y_values, weights, alpha=0.6)  # Assuming weights is a simple list; adjust as needed
            axs[idx, 2].set_xlabel('Weights')
            # axs[idx, 2].set_xlim([min(weights)-0.2, max(weights)+0.2])  # Adjust the x limits if required
            axs[idx, 2].set_yticks([])  # Remove y-ticks for consistency
        plt.tight_layout()
        return fig