def get_model_info_from_name(name,strange):
        model_info = {'name': name}
        orders = {'nlo': 'nlo', 'n2lo': 'n2lo', 'n3lo': 'n3lo', 'lo': 'lo','llo':'llo'}
        type_full_names = {
        'x': 'chiral',
        's': 'strange',
        'd': 'disc',
        'l': 'light',
        'f': 'fpi'  
        }
        if strange == '2':
            particles = ['xi','xi_st' ]
        elif strange == '1':
            particles == ['sigma','sigma_st','lam']
        elif strange == '0':
            particles == ['proton','delta']

        for t,full in type_full_names.items():
            for k, v in orders.items():
                if f'{t}_{k}' in name:
                    model_info[f'order_{full}'] = v
                    break
            else:
                model_info[f'order_{full}'] = 'lo'

        for p in particles:
            if f'{p}' in name:
                if 'particles' in model_info:
                    model_info['particles'].append(p)
                else:
                    model_info['particles'] = [p]

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

        # model_info['fv'] = bool(':fv' in name)
        model_info['xpt'] = bool(':xpt' in name)

        # You can add more logic here to extract other information

        return model_info