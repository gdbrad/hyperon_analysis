import copy
import lsqfit
import numpy as np
import h5py as h5
import gvar as gv
import functools
from pathlib import Path
import matplotlib.pyplot as plt

# local modules
import non_analytic_functions as naf
# import fv_corrections as fv
import i_o as i_o
# import xpt.priors as priors


class Xi(lsqfit.MultiFitterModel):
    '''
    Constructs the mass extrapolation functions to the physical point using SU(2) hbxpt. 
    Note: the chiral order arising in the taylor expansion denotes inclusion of a chiral logarithm 
    '''
    def __init__(self, datatag, model_info):
        super().__init__(datatag)
        self.model_info = model_info

    def fitfcn(self, p, data=None,xdata = None):
        if xdata is None:
            xdata ={}
        if 'm_pi' not in xdata:
            xdata['m_pi'] = p['m_pi']
        if 'lam_chi' not in xdata:
            xdata['lam_chi'] = p['lam_chi']
        if self.model_info['units'] == 'phys':
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        else:
            xdata['eps_pi'] = p['eps_pi']
        xdata['eps_delta'] = (p['m_{xi_st,0}'] - p['m_{xi,0}']) / p['lam_chi']
        if 'eps2_a' not in xdata:
            xdata['eps2_a'] = p['eps2_a']
        #strange quark mass mistuning
        if self.model_info['order_strange'] is not None:
            xdata['d_eps2_s'] = ((2 * p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2) - 0.3513
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
    
        output = self.fitfcn_llo_ct(p,xdata)
        output += self.fitfcn_lo_ct(p, xdata)
        output += self.fitfcn_nlo_xpt(p, xdata) 
        output += self.fitfcn_n2lo_ct(p, xdata) 
        if self.model_info['units'] == 'fpi':
            output += self.fpi_corrections_n2lo(p,xdata)
        output += self.fitfcn_n2lo_xpt(p, xdata) 

        return output 
    
    def fitfcn_llo_ct(self,p,xdata):
        output = 0
        if self.model_info['order_light'] == 0:
            output+= p['m_{xi,0}']
        return output 

    def fitfcn_lo_ct(self, p, xdata):
        ''''pure taylor-type fit to O(m_pi^2)'''
        print("contents",p)
        output = 0
        if self.model_info['units'] == 'phys': # lam_chi dependence ON #
            if self.model_info['order_disc'] is not None and self.model_info['order_disc']<=3:
                output += p['m_{xi,0}'] * (p['d_{xi,a}'] * xdata['eps2_a'])

            if self.model_info['order_light'] is not None and self.model_info['order_light']<=3:
                print('True')
                output += (p['s_{xi}'] * xdata['lam_chi'] * xdata['eps_pi']**2)

            if self.model_info['order_strange'] is not None and self.model_info['order_strange']<=3:
                if self.model_info['order_strange'] <=3 and 'd_{xi,s}' in p:
                    output += p['m_{xi,0}']*(p['d_{xi,s}'] * xdata['d_eps2_s'])

        if self.model_info['units'] == 'fpi': # lam_chi dependence OFF #
            if self.model_info['order_disc'] <=3:
                output +=  (p['d_{xi,a}'] * xdata['eps2_a'])

            if self.model_info['order_light'] <=3:
                output += (p['s_{xi}'] * xdata['eps_pi']**2)

            if self.model_info['order_strange'] <=3:
                output += (p['d_{xi,s}'] * xdata['d_eps2_s'])

            if self.model_info['order_chiral'] <=3:
                if self.model_info['order_chiral'] <=3:
                    # if self.model_info['fv']:
                    #     output += p['c0'] * xdata['eps_pi']**2 * fv.fcn_I_m(xdata['eps_pi']**2,xdata['L'],xdata['lam_chi',10])
                    # else:
                    output += p['B_{xi,2}'] * xdata['eps_pi']**2
                    # output += p['c0'] * xdata['eps_pi']**2 * np.log(xdata['eps_pi']**2)

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
            if self.model_info['order_chiral'] >=3:
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
            if self.model_info['order_fpi'] in ['n2lo']:
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
            return base_term()
             
        if self.model_info['order_chiral'] is not None:
            if self.model_info['order_chiral'] >=3:
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
    '''
    SU(2) hbxpt extrapolation multifitter class for the Xi baryon
    '''
    def __init__(self, datatag, model_info):
        super().__init__(datatag)
        self.model_info = model_info

    def fitfcn(self, p, data=None,xdata=None):
        '''extraplation formulae'''
        if xdata is None:
            xdata ={}
        # if 'a_fm' not in xdata:
        #     xdata['a_fm'] = p['a_fm']
        if 'm_pi' not in xdata:
            xdata['m_pi'] = p['m_pi']
        if 'lam_chi' not in xdata:
            xdata['lam_chi'] = p['lam_chi']
        if self.model_info['units'] == 'phys':
            xdata['eps_pi'] = p['m_pi'] / p['lam_chi']
        else:
            xdata['eps_pi'] = p['eps_pi']
        xdata['eps_delta'] = (p['m_{xi_st,0}'] - p['m_{xi,0}']) / p['lam_chi']
        if 'eps2_a' not in xdata:
            xdata['eps2_a'] = p['eps2_a']
        #strange quark mass mistuning
        if self.model_info['order_strange'] is not None:
            xdata['d_eps2_s'] = ((2 * p['m_k']**2 - p['m_pi']**2) / p['lam_chi']**2) - 0.3513

        if data is not None:
            for key in data.keys():
                p[key] = data[key]

        output = p['m_{xi_st,0}'] #llo
        output += self.fitfcn_lo_ct(p, xdata)
        output += self.fitfcn_nlo_xpt(p, xdata)
        output += self.fitfcn_n2lo_ct(p, xdata)
        output += self.fitfcn_n2lo_xpt(p, xdata)

        return output

        # if self.model_info['units'] == 'fpi':

        #     return output * xdata['lam_chi'] * xdata['a_fm']
        # elif self.model_info['units'] == 'phys':

        
    
    def fitfcn_lo_ct(self, p, xdata):
        ''''taylor extrapolation to O(m_pi^2) without terms coming from xpt expressions'''
        output = 0
        if self.model_info['units'] == 'phys': # lam_chi dependence ON #
            if self.model_info['order_disc'] <=3:
                output += (p['m_{xi_st,0}'] * (p['d_{xi_st,a}']*xdata['eps2_a']))

            if self.model_info['order_light'] <=3:
                output += (p['s_{xi,bar}'] * xdata['lam_chi'] * xdata['eps_pi']**2)

            if self.model_info['order_strange'] is not None:
                if self.model_info['order_strange'] <=3 and 'd_{xi_st,s}' in p:
                    output += p['m_{xi_st,0}']*(p['d_{xi_st,s}'] * xdata['d_eps2_s'])
                    
        elif self.model_info['units'] == 'fpi': # lam_chi dependence OFF #
            if self.model_info['order_disc'] <=3:
                output += (p['d_{xi_st,a}']*xdata['eps2_a'])

            if self.model_info['order_chiral'] <=3:
                output += (p['s_{xi,bar}'] * xdata['eps_pi']**2)

            if self.model_info['order_strange'] <=3:
                output += (p['d_{xi_st,s}'] * xdata['d_eps2_s'])

        return output
    
    def fitfcn_nlo_xpt(self, p, xdata):
        '''xpt extrapolation to O(m_pi^3)'''
        if self.model_info['order_chiral'] is not None:
            if self.model_info['order_chiral'] >=2:
             
                term1 = (-5/6) * np.pi * p['g_{xi_st,xi_st}']**2 * xdata['eps_pi']**3
                term2 = 1/2* p['g_{xi_st,xi}']**2 * naf.fcn_F(xdata['eps_pi'], -xdata['eps_delta'])

                if self.model_info['units'] == 'phys':
                    return term1 * xdata['lam_chi'] - term2 * xdata['lam_chi']
                
                if self.model_info['units'] in ('fpi','lattice'):
                    return term1 - term2
        else:
            return 0


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
        
        if self.model_info['order_chiral'] is not None:
            if self.model_info['order_chiral'] >=3:
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
        return data[self.datatag]