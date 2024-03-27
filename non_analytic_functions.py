import gvar as gv
import numpy as np
import scipy.special as ss

### non-analytic functions that arise in extrapolation formulae for hyperon masses

def fcn_L(m, mu):
    output = m**2 * np.log(m**2 / mu**2)
    return output

def fcn_L_bar(m,mu):
    output = m**4 * np.log(m**2 / mu**2)
    return output

# def fcn_R(g):

# #if isinstance(g, gv._gvarcore.GVar):
#     x = g
#     conds = [(x > 0) & (x <= 1), x > 1]
#     funcs = [lambda x: np.sqrt(1-x) * np.log((1-np.sqrt(1-x))/(1+np.sqrt(1-x))),
#                 lambda x: 2*np.sqrt(x-1)*np.arctan(np.sqrt(x-1))
#                 ]

#     pieces = np.piecewise(x, conds, funcs)
#     return pieces

def fcn_R(g):
    def case1(g):
        output = np.sqrt(1 - g) *np.log( (1-np.sqrt(1 - g))/(1+np.sqrt(1 - g)) )
        return output
    def case2(g):
        output = 2 *np.sqrt(g - 1) *np.arctan(np.sqrt(g - 1))
        return output

    if hasattr(g, "__len__"):
        output = np.array([ case1(xi) if (0 < xi <= 1) else case2(xi) for xi in g])
        return output

    else:
        if (0 < g <= 1):
            return case1(g)
        else:
            return case2(g)
        
def fcn_dR(x):
    def case1(x):
        output = 1/x - np.log( (1 - np.sqrt(1-x))/(1 + np.sqrt(1-x)) )/(2 *np.sqrt(1 - x))
        return output

    def case2(x):
        return 2

    def case3(x):
        return 1/x + np.arctan(np.sqrt(x - 1)) / np.sqrt(x-1)
    
    if hasattr(x, "__len__"):
        output = np.array([fcn_dR(xi) for xi in x])
        return output

    else: 
        if (0 < x < 1):
            return case1(x)
        elif x == 1:
            return case2(x)
        else:
            return case3(x)
# def fcn_dR(g):
# #if isinstance(g, gv._gvarcore.GVar):
#     x = g
#     conds = [(x > 0) & (x < 1), x==1, x > 1]
#     funcs = [lambda x: 1/x - np.log((1-np.sqrt(1-x))/(np.sqrt(1-x)+1))/(2*np.sqrt(1-x)),
#                 lambda x: x==2,
#                 lambda x: 1/x + np.arctan(np.sqrt(x-1)) / np.sqrt(x-1)
#                 ]

#     pieces = np.piecewise(x, conds, funcs)
#     return pieces


def fcn_F(eps_pi, eps_delta):
    if hasattr(eps_pi, '__len__'):
        eps_pi = np.array(eps_pi)
    elif hasattr(eps_delta, '__len__'):
        eps_delta = np.array(eps_delta)
    output = (
        - eps_delta *(eps_delta**2 - eps_pi**2) *fcn_R((eps_pi/eps_delta**2))
        - (3/2) *eps_pi**2 *eps_delta *np.log(eps_pi**2)
        - eps_delta**3 *np.log(4 *(eps_delta/eps_pi)**2) 
    )
    return output

def fcn_dF(eps_pi, eps_delta):
    output = 0
    output += (
        + 2*eps_delta**3 / eps_pi
        - 3*eps_delta*eps_pi *np.log(eps_pi**2) 
        - 3*eps_delta*eps_pi
        + (2*eps_pi**3 / eps_delta - 2*eps_delta*eps_pi)*fcn_dR(eps_pi**2/eps_delta**2)
        + 2*eps_pi*eps_delta*fcn_R(eps_pi**2/eps_delta**2)
    )
    return output

def fcn_J(eps_pi, eps_delta):
    if hasattr(eps_pi, '__len__'):
        eps_pi = np.array(eps_pi)
    elif hasattr(eps_delta, '__len__'):
        eps_delta = np.array(eps_delta)
    output = 0
    output += eps_pi**2 * np.log(eps_pi**2)
    output += 2*eps_delta**2 * np.log((4*eps_delta**2)/ eps_pi**2)
    output += 2*eps_delta**2 * fcn_R(eps_pi**2/eps_delta**2)

    return output

def fcn_dJ(eps_pi,eps_delta):
    output = 0
    output -= 4*eps_delta**2/eps_pi 
    output += 4*eps_pi*fcn_dR(eps_pi**2/eps_delta**2) 
    output += 2*eps_pi*np.log(eps_pi**2) + 2*eps_pi
    return output

def safe_divide(num, denom, default=0):
    # This handles both scalars and arrays
    return np.where(denom != 0, num / denom, default)

    



    
