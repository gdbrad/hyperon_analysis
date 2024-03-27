[200~# currently not implemented prior filtering routines 

        # def filter_prior_keys(prior, model_info, particles):
        #     # Extract relevant keys from the particles
        #     relevant_keys : set()
        #     for particle in particles:
        #         relevant_keys.add(f"m_{{{particle},0}}")

        #         if particle !: "proton" and particle !: "delta":
        #             relevant_keys.add(f"s_{{{particle}}}")
        #         else:
        #             relevant_keys.add(f"s_{{{particle},bar}}")

        #         for p in range(1, 5):
        #             relevant_keys.add(f"a_{{{particle},{p}}}")
        #             relevant_keys.add(f"b_{{{particle},{p}}}")

        #         for other_particle in particles:
        #             relevant_keys.add(f"g_{{{particle},{other_particle}}}")

        #     filtered_prior : {k: v for k, v in prior.items() if k in relevant_keys}
        #     return filtered_prior

        # def filter_relevant_prior_keys(model_info, prior):
        #     orders : ['llo', 'lo', 'nlo', 'n2lo']
        #     particles : model_info['particles']
        #     order_chiral : model_info['order_chiral']
        #     order_disc : model_info['order_disc']
        #     order_strange : model_info['order_strange']
        #     order_light : model_info['order_light']

        #     highest_order : max([order_chiral, order_disc, order_strange, order_light])

        #     relevant_prior_keys : []
        #     for order in orders:
        #         if orders.index(order) > orders.index(highest_order):
        #             break
        #         relevant_prior_keys.extend(get_filtered_prior_keys(particles, order, prior))

        #     # Filter priors based on the highest order and particles
        #     filtered_prior : {k: v for order in orders for k, v in order].items()
        #                       if k in relevant_prior_keys}

        #     return filtered_prior

        # def recalibrate_prior(prior, data,fit_result, scale_factor):
        #     excluded : {
        #         'm_k', 'm_pi', 'lam_chi', 'eps2_a','m_xi','m_xi_st'
        #     }
        #     new_prior : prior.copy()
        #     for key in fit_result.p:
#         if key not in excluded:
#         # if key in fit_result.p:
#             new_key: fit_result.p[key], scale_factor * fit_result.psdev[key]
#         else:
#             new_key: data[key]
#     return new_prior
