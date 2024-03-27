import lsqfit
import numpy as np
import gvar as gv
import time
#import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import scipy.stats as stats

import fit_routine as fit
import lsqfitics
import i_o
import get_models

class Load_Fits:
    def __init__(self,fits):
        

class Model_Average:
    def __init__(self,strange:str,units:str):
        # self.fit_collection = fit_collection
            # self.fit_results = i_.get_fit_collection()
        self.strange = strange
        self.units = units
        self.model_data = {}
        if self.units == 'phys':
            _data = i_o.InputOutput(units='phys', strange=self.strange)
            self.data,self.prior,self.phys_data = _data.get_data_and_prior_for_unit()

        elif self.units == 'fpi':
            _data = i_o.InputOutput(units='fpi', strange=self.strange)
            self.data,self.prior,self.phys_data = _data.get_data_and_prior_for_unit()

        else:
            raise ValueError(f"Invalid value for 'units': {units}")


    def compute_models(self):
        model_data = {}

        # get models
        models = get_models.GenerateModels(strange=self.strange)
        all = models.generate_model_names()
        for info in all:
            units = info['units']
            # Conditionally load data and prior based on 'units'
            if units == 'phys':
                _data = i_o.InputOutput(units='phys', strange=self.strange)
            elif units == 'fpi':
                _data = i_o.InputOutput(units='fpi', strange=self.strange)
            else:
                raise ValueError(f"Invalid value for 'units': {units}")

            # Add the loaded data to the model_data dictionary
            model_data[info['name']] = {'data': _data, 'model_info': info}

        return model_data


    
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
        
        