import numpy as np
import gvar as gv
from pathlib import Path
import os
#import yaml
import h5py
import priors

class InputOutput:
    '''Bootstrapped data ingestion and output to gvar average datasets'''
    def __init__(self,
                 units:str,
                 strange:str,
                 **kwargs
                ):
        
        self.units = units # physical or fpi units
        self.strange = strange # strangeness S=0,1,2
        
        for key,value in kwargs.items():
            if key == 'scale_corr':
                self.scale_corr = value
            elif key == 'conversion_factor':
                self.conversion_factor = value


        cwd = Path(os.getcwd())
        project_root = cwd.parent
        self.data_dir = os.path.join(project_root, "data")
        data_path_hyperon = os.path.join(self.data_dir, "hyperon_data.h5")
        data_path_input = os.path.join(self.data_dir, "input_data.h5")

        # bootstrapped hyperon correlator data
        with h5py.File(data_path_hyperon, 'r') as f:
            ens_hyp = sorted(list(f.keys()))
            ens_hyp = sorted([e.replace('_hp', '') for e in  ens_hyp])

        # bootstrapped scale setting data 
        with h5py.File(data_path_input, 'r') as f: 
            ens_in = sorted(list(f.keys()))

        ensembles = sorted(list(set(ens_hyp) & set(ens_in)))
        # hopefully with new data these can be safely re-added
        ensembles.remove('a12m220')
        ensembles.remove('a12m220ms')
        ensembles.remove('a12m310XL')
        ensembles.remove('a12m220S')
        ensembles.remove('a12m180L')
        self.ensembles = ensembles

        if self.strange == '0':
            self.dim1_obs = ['m_proton','m_delta' ,'eps_pi','m_pi', 'm_k', 'lam_chi']
            self.masses = ['m_proton','m_delta']
        elif self.strange == '1':
            self.dim1_obs = ['m_lambda', 'm_sigma', 'm_sigma_st', 'm_pi', 'm_k', 'lam_chi', 'eps_pi']
        elif self.strange == '2':
            self.dim1_obs = ['m_pi', 'm_k', 'lam_chi', 'eps_pi']
            self.y_obs = ['m_xi', 'm_xi_st']
        
        else:
            raise ValueError(f"Unknown strange value: {self.strange}")
    
    @property
    def _get_ens(self):
        return self.ensembles
        
    def get_data_and_prior_for_unit(self):
        prior = priors.get_prior(units=self.units,conversion_factor=self.conversion_factor)
    
        if self.units == 'fpi':
            phys_point_data = self.get_data_phys_point(fpi_units=True)
        else:
            phys_point_data = self.get_data_phys_point(fpi_units=False)

        data = self.perform_gvar_processing()
        new_prior = self.make_prior(data=data, prior=prior)
        
        return data, new_prior, phys_point_data,self.ensembles
            
    def _get_bs_data(self):
        """Ingest bootstrapped correlator, pseudoscalar, and lattice params data. 
        Resulting raw dataset Will be passed to `perform_gvar_processing` for gvar averaging.
        Params:
            - scale_corr: choice of how lattice scale data is correlated
                1. full correlation
                2. partial correlation (correlate b/w groups of latt. spacing)
                3. no correlation (decorr a b/w all individual ens)
        """
        to_gvar = lambda arr : gv.gvar(arr[0], arr[1]) #just extracting mean,sdev from gvar
        to_gvar_afm = lambda g: gv.gvar(gv.mean(g),gv.sdev(g)) # creating a new gvar
        hbar_c = self.get_data_phys_point(param='hbarc',fpi_units=None) # MeV-fm (PDG 2019 conversion constant)
        scale_factors = gv.load(self.data_dir +'/scale_setting.p') # on-disk scale data
        a_fm =  gv.load(self.data_dir +'/a_fm_results.p')
        tmp_afm = {}
        # if self.scale_corr == 'partial':
        #     for lattice_space in ['a06','a09','a12','a15']:
        #         tmp_afm[lattice_space] = to_gvar_afm(a_fm[lattice_space])
        #     a_fm = tmp_afm

        scheme = 'w0_imp'
        data = {}
        with h5py.File(self.data_dir+'/input_data.h5','r') as f: 
            for ens in self.ensembles:
                data[ens] = {}
                data[ens]['a_fm'] = to_gvar(f[ens]['a_fm'][scheme][:])

                if scheme in ['w0_org','w0_imp'] and self.units=='phys':
                    data[ens]['units'] = hbar_c *scale_factors[scheme+':'+ens[:3]] /scale_factors[scheme+':w0']
                    data[ens]['a_fm'] = 1/ (scale_factors[scheme+':'+ens[:3]] /scale_factors[scheme+':w0']) # in fm
                elif scheme in ['w0_org', 'w0_imp'] and self.units in ['fpi','latt']:
                    data[ens]['units'] = scale_factors[scheme+':'+ens[:3]]
                elif scheme in ['w0_org', 'w0_imp'] and self.units=='w0':
                    data[ens]['units'] = hbar_c *scale_factors[scheme+':'+ens[:3]]
                # scale settting correlation options (3)
                # if self.scale_corr == 'partial':
                #     a_xx = [ens[:3] for ens in self.ensembles]
                #     temp = {}
                #     for a in a_xx:
                #         temp.update(gv.rebuild({ ens : data[ens]['units'] for ens in self.ensembles if ens.startswith(a)}))
                #         data[ens]['units_MeV'] = temp  # can we remove hbarc correlation
                elif self.scale_corr == 'full':
                    data[ens]['units_MeV'] = data[ens]['units']
                elif self.scale_corr == 'no':
                    data[ens]['units_MeV'] = gv.gvar(gv.mean(data[ens]['units']), gv.sdev(data[ens]['units']))

                # if self.scale_corr == 'partial':
                #     data[ens]['units_MeV'] = hbar_c / a_fm[ens[:3]]  # can we remove hbarc correlation
                #     data[ens]['units_MeV'] = hbar_c / a_fm[ens[:3]]  # can we remove hbarc correlation
                # elif self.scale_corr == 'full':
                #     data[ens]['units_MeV'] = hbar_c / to_gvar_afm(a_fm[ens[:3]])
                # elif self.scale_corr == 'no':
                #     data[ens]['units_MeV'] = hbar_c /a_fm[ens[:3]]

                data[ens]['hbar_c'] = hbar_c
                data[ens]['alpha_s'] = f[ens]['alpha_s']
                data[ens]['L'] = f[ens]['L'][()]
                data[ens]['m_pi'] = f[ens]['mpi'][1:]
                data[ens]['m_k'] = f[ens]['mk'][1:]
                data[ens]['lam_chi'] = 4 *np.pi *f[ens]['Fpi'][1:]
                data[ens]['Fpi'] = f[ens]['Fpi'][:] 
                data[ens]['eps_pi'] = data[ens]['m_pi'] / data[ens]['lam_chi']
                data[ens]['units_Fpi'] = 1/data[ens]['lam_chi'][:]
                if self.units=='Fpi':
                #     #for data[ens]['units'] not in data[ens]['lam_chi']:
                    data[ens]['units'] =  1/data[ens]['lam_chi'] #for removing lam_chi dependence of fits    
                if scheme == 'w0_imp':
                    data[ens]['eps2_a'] = 1 / (2 *to_gvar(f[ens]['w0a_callat_imp']))**2
            
        with h5py.File(self.data_dir+'/hyperon_data.h5', 'r') as f:
            for ens in self.ensembles:
                for obs in list(f[ens]):
                    data[ens].update({obs: f[ens][obs][:]})
                if ens+'_hp' in list(f):
                    for obs in list(f[ens+'_hp']):
                        data[ens].update({obs : f[ens+'_hp'][obs][:]})
                    for obs in ['lambda', 'sigma', 'sigma_st', 'xi_st', 'xi','proton','delta']:
                        if self.units == 'fpi':
                            data[ens].update({'m_'+obs: data[ens]['m_'+obs] / data[ens]['lam_chi'][:]})
        return data
    
    def perform_svdcut(self):
        svd_data = {}
        bs_data = self._get_bs_data()
        svd_data  = {(ens,o): bs_data[ens][o] for ens in list(bs_data) for o in [p for p in self.dim1_obs]}
        s= gv.dataset.svd_diagnosis(svd_data)
        # avgdata = gv.svd(s.avgdata,svdcut=s.svdcut)
        s.plot_ratio(show=True)
        # svd_cut = s.svdcut

        return s.svdcut
    
    def perform_gvar_processing(self):
        """convert raw baryon and pseudoscalar data to gvar datasets"""
        bs_data = self._get_bs_data()
        gv_data = {}
        for ens in self.ensembles:
            gv_data[ens] = gv.BufferDict()
            for obs in self.dim1_obs:
                gv_data[ens][obs] = bs_data[ens][obs] - np.mean(bs_data[ens][obs]) + bs_data[ens][obs][0]
                for _obs in self.y_obs:
                    gv_data[ens][_obs] = bs_data[ens][_obs] - np.mean(bs_data[ens][_obs]) + bs_data[ens][_obs][0]
                    # # #fit in lattice units 
                    # if self.conversion_factor:
                    #         gv_data[ens][_obs] = gv_data[ens][_obs]
                    # else:
                    #         gv_data[ens][_obs] = gv_data[ens][_obs] *bs_data[ens]['units']
                    # # fit in MeV or F_pi units
                    # else:
                    #     

            gv_data[ens] = gv.dataset.avg_data(gv_data[ens], bstrap=True) 
            for _obs in self.y_obs:
                    if self.conversion_factor:
                        gv_data[ens][_obs] = gv_data[ens][_obs] * bs_data[ens]['units_MeV']
                    else:
                        gv_data[ens][_obs] = gv_data[ens][_obs] *bs_data[ens]['units_MeV']

            gv_data[ens]['eps2_a'] = bs_data[ens]['eps2_a']
            gv_data[ens]['L'] = gv.gvar(bs_data[ens]['L'], bs_data[ens]['L'] / 10**6)
            # gv_data[ens]['units_MeV'] = bs_data[ens]['units_MeV']
            gv_data[ens]['units'] = bs_data[ens]['units']

            # gv_data[ens]['a_fm'] = bs_data[ens]['a_fm']

        output = {}
        for param in gv_data[self.ensembles[0]]:
            output[param] = np.array([gv_data[ens][param] for ens in self.ensembles])
        return output

    def get_data(self, div_lam_chi=False,svd_test=None):
        # gv_data = {}
        # for ens in self.ensembles:
        #     gv_data[ens] = gv.BufferDict()
        if svd_test:
            output = self.perform_svdcut()
            return output
        
        output = self.perform_gvar_processing()
        return output
            
    def get_data_phys_point(self, fpi_units=None,param=None):
        '''
        define physical point data
        '''
        data_phys_point = {
            'eps2_a' : gv.gvar(0),
            # 'a' : gv.gvar(0),
            'alpha_s' : gv.gvar(0.0),
            'L' : gv.gvar(np.infty),
            'hbarc' : gv.gvar(197.3269804, 0), # MeV-fm

            'lam_chi' : 4 *np.pi *gv.gvar('92.07(57)'),
            'm_pi' : gv.gvar('134.8(3)'), # '138.05638(37)'
            'm_k' : gv.gvar('494.2(3)'), # '495.6479(92)'
            'eps_pi' : gv.gvar('134.8(3)') / (4 *np.pi *gv.gvar('92.07(57)')),
            'eps_k' : gv.gvar('494.2(3)') / (4 *np.pi *gv.gvar('92.07(57)')),

            'm_lambda' : gv.gvar(1115.683, 0.006),
            'm_sigma' : np.mean([gv.gvar(g) for g in ['1189.37(07)', '1192.642(24)', '1197.449(30)']]),
            'm_sigma_st' : np.mean([gv.gvar(g) for g in ['1382.80(35)', '1383.7(1.0)', '1387.2(0.5)']]),
            'm_xi' : np.mean([gv.gvar(g) for g in ['1314.86(20)', '1321.71(07)']]),
            'm_xi_st' : np.mean([gv.gvar(g) for g in ['1531.80(32)', '1535.0(0.6)']]),
            'm_omega' : gv.gvar(1672.45,29),
            'm_proton' : gv.gvar(938.272,.0000058),
            'm_delta' : gv.gvar(1232, 2),

            
        }
        dim0_obs_to_m_baryon = {
        'eps_lambda': 'm_lambda',
        'eps_sigma': 'm_sigma',
        'eps_sigma_st': 'm_sigma_st',
        'eps_xi': 'm_xi',
        'eps_xi_st': 'm_xi_st',
        'eps_delta': 'm_delta'

    }
        # Compute new values for dim0_obs keys
        if fpi_units:
            for obs, m_baryon_key in dim0_obs_to_m_baryon.items():
                data_phys_point[m_baryon_key] = data_phys_point[m_baryon_key] / data_phys_point['lam_chi']
        if param is not None:
            return data_phys_point[param]
        return data_phys_point

    def get_posterior(self,fit_test=None,prior=None,param=None):
        '''
        get posteriors from resulting lsqfit multifitter object
        '''
        if param == 'all':
            return fit_test.fit.p
        elif param is not None:
            return fit_test.fit.p[param]
        else:
            output = {}
            for param in prior:
                if param in fit_test.fit.p:
                    output[param] = fit_test.fit.p[param]
            return output

    def make_prior(self,data,prior):
        '''
        reconstruct priors with second set coming from the bootstrapped input data file
        '''
        new_prior = {}
        for key in prior:
            new_prior[key] = prior[key]
        for key in ['m_pi', 'm_k', 'lam_chi', 'eps2_a']:
            new_prior[key] = data[key]
        return new_prior
    
def update_model_key_name(mdl_key, unit):
    if unit not in mdl_key:
        return f"{mdl_key}_{unit}"
    return mdl_key
    
def get_unit_description(unit):
    '''Returns a description based on the unit.'''
    if unit == 'phys':
        return "fitting in phys units"
    elif unit == 'fpi':
        return "fitting in fpi units"
    else:
        return f"fitting in {unit} units"
    
@property   
def _get_data_phys_point(param=None):
        '''
        define physical point data
        '''
        data_phys_point = {
            'eps2_a' : gv.gvar(0),
            # 'a' : gv.gvar(0),
            'alpha_s' : gv.gvar(0.0),
            'L' : gv.gvar(np.infty),
            'hbarc' : gv.gvar(197.3269804, 0), # MeV-fm

            'lam_chi' : 4 *np.pi *gv.gvar('92.07(57)'),
            'm_pi' : gv.gvar('134.8(3)'), # '138.05638(37)'
            'm_k' : gv.gvar('494.2(3)'), # '495.6479(92)'
            'eps_pi' : gv.gvar('134.8(3)') / (4 *np.pi *gv.gvar('92.07(57)')),
            'eps_k' : gv.gvar('494.2(3)') / (4 *np.pi *gv.gvar('92.07(57)')),

            'm_lambda' : gv.gvar(1115.683, 0.006),
            'm_sigma' : np.mean([gv.gvar(g) for g in ['1189.37(07)', '1192.642(24)', '1197.449(30)']]),
            'm_sigma_st' : np.mean([gv.gvar(g) for g in ['1382.80(35)', '1383.7(1.0)', '1387.2(0.5)']]),
            'm_xi' : np.mean([gv.gvar(g) for g in ['1314.86(20)', '1321.71(07)']]),
            'm_xi_st' : np.mean([gv.gvar(g) for g in ['1531.80(32)', '1535.0(0.6)']]),
            'm_omega' : gv.gvar(1672.45,29),
            'm_proton' : gv.gvar(938.272,.0000058),
            'm_delta' : gv.gvar(1232, 2),
        }

        return data_phys_point


    
    # def pickle_out(self,fit_info):
    #     model = fit_info['name']
    #     if not os.path.exists(self.project_path +'/results/'+ self.collection['name'] +'/pickles/'):
    #         os.makedirs(self.project_path +'/results/'+ self.collection['name'] +'/pickles/')
    #     filename = self.project_path +'/results/'+ self.collection['name'] +'/pickles/'+'_'+ model +'.p'

    #     output = {}

    #     output['logGBF'] = gv.gvar(fit_info['logGBF'])
    #     output['chi2/df'] = gv.gvar(fit_info['chi2/df'])
    #     output['Q'] = gv.gvar(fit_info['Q'])

    #     for key in fit_info['prior'].keys():
    #         output['prior:'+key] = fit_info['prior'][key]

    #     for key in fit_info['posterior'].keys():
    #         output['posterior:'+key] = fit_info['posterior'][key]

    #     for key in fit_info['phys_point'].keys():
    #         # gvar can't handle integers -- entries not in correlation matrix
    #         output['phys_point:'+key] = fit_info['phys_point'][key]

    #     for key in fit_info['error_budget']:
    #         output['error_budget:'+key] = gv.gvar(fit_info['error_budget'][key])

    #     gv.dump(output, filename)
    #     return None
    
    # def _unpickle_fit_info(self, mdl_key):
    #     filepath = self.project_path +'/results/'+ self.collection['name'] +'/pickles/'+ mdl_key +'.p'
    #     if os.path.isfile(filepath):
    #         return gv.load(filepath)
    #     else:
    #         return None
        
    # def get_fit_collection(self):
    #     if os.path.exists(self.project_path +'/results/'+ self.collection['name'] +'/pickles/'):
    #         output = {}

    #         pickled_models = []
    #         for file in os.listdir(self.project_path +'/results/'+ self.collection['name'] +'/pickles/'):
    #             if(file.endswith('.p')):
    #                 pickled_models.append(file.split('.')[0])

    #         for mdl_key in pickled_models:
    #             fit_info_mdl_key = self._unpickle_fit_info(mdl_key=mdl_key)
    #             model = mdl_key.split('_', 1)[1]

    #             obs = mdl_key.split('_')[0]
    #             if obs not in output:
    #                 output[obs] = {}

    #             output[obs][model] = {}
    #             output[obs][model]['name'] = model
    #             if obs == 'w0':
    #                 output[obs][model]['w0'] = fit_info_mdl_key['w0']
    #             elif obs == 't0':
    #                 output[obs][model]['sqrt_t0'] = fit_info_mdl_key['sqrt_t0']
    #             output[obs][model]['logGBF'] = fit_info_mdl_key['logGBF'].mean
    #             output[obs][model]['chi2/df'] = fit_info_mdl_key['chi2/df'].mean
    #             output[obs][model]['Q'] = fit_info_mdl_key['Q'].mean
    #             output[obs][model]['prior'] = {}
    #             output[obs][model]['posterior'] = {}
    #             output[obs][model]['phys_point'] = {}
    #             output[obs][model]['error_budget'] = {}

    #             for key in fit_info_mdl_key.keys():
    #                 if key.startswith('prior'):
    #                     output[obs][model]['prior'][key.split(':')[-1]] = fit_info_mdl_key[key]
    #                 elif key.startswith('posterior'):
    #                     output[obs][model]['posterior'][key.split(':')[-1]] = fit_info_mdl_key[key]
    #                 elif key.startswith('phys_point'):
    #                     output[obs][model]['phys_point'][key.split(':')[-1]] = fit_info_mdl_key[key]
    #                 elif key.startswith('error_budget'):
    #                     output[obs][model]['error_budget'][key.split(':')[-1]] = fit_info_mdl_key[key].mean

    #         return output
        
    # def save_fit_info(self, fit_info):
    #     self._pickle_fit_info(fit_info)
    #     return None
    



        




