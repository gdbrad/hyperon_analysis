"""
Generate information in the form of string for xpt models of interest. Passed to lsqfit MultiFitterModel
"""
import itertools

class GenerateModels:
    def __init__(self, strange:str):
        self.strange = strange

    def parse_model_name(self,name,strange):
        """pass a model name and construct the model info from this. Inversion of `generate_model_names`
        """
        components = name.split(':')
        if strange == '2':
            # particles = ['xi','xi_st']
            particles = ['xi_st']
        # Initialize order variables
        order_chiral = None
        order_strange = None
        order_disc = None
        order_light = None

        # Extract orders if present
        for component in components:
            if component.startswith('s_'):
                order_strange = int(component[2:])
            elif component.startswith('d_'):
                order_disc = int(component[2:])
            elif component.startswith('l_'):
                order_light = int(component[2:])
            elif component.startswith('x_'):
                order_chiral = int(component[2:])

        # Construct and return the model_info dictionary
        model_info = {
            'order_chiral': order_chiral,
            'order_strange': order_strange,
            'order_disc': order_disc,
            'order_light': order_light,
            'particles': particles,
            'units': 'phys'
        }

        return model_info

    
    def generate_model_names(self):
        """generating cartersian product of model options to form a model average
        order dictionary. If type of term is not included, this means it does not appear at the particular order:
            0/None: light terms = cnst
            1/LO:   
                light = m_pi^2
                strange = 2m_k^2 - m_pi^2
                disc = a^2
            2/NLO: 
                xpt = m_pi^3 
            3/N2LO:
                light = m_pi^4
                strange = (2m_k^2 - m_pi^2)^2
                disc = a^4
                xpt = m_pi^4 * log(m_pi^2)
                """
        orders_xpt = [0,2,3]
        orders_light = [0,1,3]
        orders_strange = [0,1,3]
        orders_disc = [0,1,3]
        scale_correlations = ['no','partial','full']
        conversion_factor = [False]
        # units = ['phys','fpi']
        units = ['phys']
        if self.strange == '2':
            # particles = ['xi','xi_st' ]
            particles = ['xi_st']
        elif self.strange == '1':
            particles = ['sigma','sigma_st','lam']
        elif self.strange == '0':
            particles = ['proton','delta']
        else:
            raise ValueError("Invalid input for strangeness. Should be either 0,1,2 as a string")
        model_names = {}
        unique_names = set()

        # for order_c in orders_chiral:
        for order_chiral,order_strange, order_disc, order_light,unit,cf in itertools.product(orders_xpt,orders_strange,orders_disc,orders_light,units,conversion_factor):

            # Build the model name
            parts = []
            for particle in particles:
                parts.append(particle)

            unit_list = []
            unit_list.append(unit)

            cf_list = []
            cf_list.append(cf)

            # scale_corr_list = []
            # scale_corr_list.append(scale_corr)
            # chiral order should not exceed that of the light order 
            if order_chiral and order_light and order_chiral < order_light:
                continue
            if order_strange in [1,3]:
                parts.append(f"s_{order_strange}")
            if order_disc in [1,3]:
                parts.append(f"d_{order_disc}")
            if order_light in [0,1,3]:
                parts.append(f"l_{order_light}")
            if order_chiral in [2,3]:
                parts.append(f"x_{order_chiral}")

            name = ':'.join(parts)
            name += ':conv_fact_'+(str(cf))
            # name = ':'.join(str(cf_list))

            if name not in unique_names:
                unique_names.add(name)

            # Add model info
            model_info = {
                'name': name,
                'order_chiral': order_chiral,
                'order_strange': order_strange,
                'order_disc': order_disc,
                'order_light': order_light,
                'particles': particles,
                'units': unit,
                'conversion_factor':cf
                # 'scale_corr': scale_corr
            }
            
            model_names[model_info['name']] = model_info
            del model_names[model_info['name']]['name']

        return model_names

    @classmethod
    def from_name(cls, name):
        model_info = cls.get_model_info_from_name(name)
        return cls(model_info)

    @staticmethod
    def get_model_info_from_name(name):
        model_info = {'name': name}
        orders = {'nlo': 'nlo', 'n2lo': 'n2lo', 'n3lo': 'n3lo', 'lo': 'lo','llo':'llo'}
        types = ['s', 'd', 'x', 'l']
        particles = ['xi','xi_st','sigma','sigma_st','lam']

        for t in types:
            for k, v in orders.items():
                if f'{t}_{k}' in name:
                    model_info[f'order_{t}'] = v
                    break
            else:
                model_info[f'order_{t}'] = 'lo'

        for p in particles:
            if f'{p}' in name:
                model_info['particles'] = p

        # Definition of eps2_a
        if '_w0orig' in name:
            model_info['eps2a_defn'] = 'w0_orig'
        elif ':w0imp' in name:
            model_info['eps2a_defn'] = 'w0_imp'
        elif '_t0orig' in name:
            model_info['eps2a_defn'] = 't0_orig'
        elif '_t0impr' in name:
            model_info['eps2a_defn'] = 't0_imp'
        elif '_variable' in name:
            model_info['eps2a_defn'] = 'variable'
        else:
            model_info['eps2a_defn'] = None

        model_info['fv'] = bool(':fv' in name)
        model_info['xpt'] = bool(':xpt' in name)

        # You can add more logic here to extract other information

        return model_info

    