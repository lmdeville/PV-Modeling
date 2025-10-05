# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:32:14 2025

@author: lmdevil
"""

def cec_to_pvsyst(alpha_sc,a_ref,I_L_ref,I_o_ref,R_sh_ref,R_s,Adjust,NcelS,
                  initial=None,
                  bounds=None,
                  options=None,
                  method='Nelder-Mead',
                  R_sh_exp=5.5, EgRef=1.121, dEgdT=-0.0002677): 

    '''
    This function translates CEC parameter values to PVsyst parameter values
    using methods described by Deville et al [1]

    Parameters
    ----------
   alpha_sc : float
        The short-circuit current temperature coefficient of the
        module in units of A/C.
        
   a_ref : float
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at reference
        conditions, in units of V.
        
   I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions,
        in amperes.

    I_o_ref : float
        The dark or diode reverse saturation current at reference conditions,
        in amperes.

    R_sh_ref : float
        The shunt resistance at reference conditions, in ohms.

    R_s : float
        The series resistance at reference conditions, in ohms.

    Adjust : float
        The adjustment to the temperature coefficient for short circuit
        current, in percent
        
    NCels : integer
        The number of cells connected in series

    initial : list
        A list defining the initial guess for the PVsyst parameters.
        See the notes section for more details.

    bounds : list
        A list defining the upper and lower bounds for the PVsyst parameters
        within the optimization.
        See the notes section for more details.

    options : dict, default {'maxiter':5000, 'maxfev':5000, 'xatol':0.001, 'disp': True}
        A dictionary of solver options, details on method specific options available in 
        scipy.optimize.minimize documentation

    method : str or callable, default method=Nelder-Mead
        Type of solver, should be one of listed options for scipy.optimize.minimize

    R_sh_exp : float, default R_sh_exp=5.5
        The exponent in the equation for shunt resistance, unitless. 
    
    EgRef : float, default EgRef=1.121
        The energy bandgap at reference temperature in units of eV.
        1.121 eV for crystalline silicon. EgRef must be >0.  For parameters
        from the SAM CEC module database, EgRef=1.121 is implicit for all
        cell types in the parameter estimation algorithm used by NREL.

    dEgdT : float, default dEgdT=-0.0002677
        The temperature dependence of the energy bandgap at reference
        conditions in units of 1/K. May be either a scalar value
        (e.g. -0.0002677 as in [3]) or a DataFrame (this may be useful if
        dEgdT is a modeled as a function of temperature). For parameters from
        the SAM CEC module database, dEgdT=-0.0002677 is implicit for all cell
        types in the parameter estimation algorithm used by NREL.


    Returns
    -------
    alpha_sc : float
        The short-circuit current temperature coefficient of the
        module in units of A/C.

    gamma_ref : float
        The diode ideality factor

    mu_gamma : float
        The temperature coefficient for the diode ideality factor, 1/K

    I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions,
        in amperes.

    I_o_ref : float
        The dark or diode reverse saturation current at reference conditions,
        in amperes.

    R_sh_0 : float
        The shunt resistance at zero irradiance conditions, in ohms.

    R_sh_ref : float
        The shunt resistance at reference conditions, in ohms.

    R_s : float
        The series resistance at reference conditions, in ohms.


    Notes
    -----
    The order of the parameters in the initial guess & bounds series:
    [alpha_sc, gamma_ref, mu_gamma, I_L_ref, I_o_ref, R_sh_mult, R_sh_ref, R_s]
    


    [1] L. Deville, C. W. Hansen, K. S. Anderson, T. L. Chambers 
    and M. Theristis, "Parameter Translation for Photovoltaic Single-Diode 
    Models," in IEEE Journal of Photovoltaics, vol. 15, no. 3, pp. 451-457,
    May 2025, doi: 10.1109/JPHOTOV.2025.3539319. 
'''
    if initial == None:
        initial=[alpha_sc, 1.2, 0.0001,I_L_ref,I_o_ref,4,R_sh_ref,R_s]
    if bounds == None:
        bounds=[(-1,1), (1,2), (-1,1), (1e-12,100), (1e-24,0.1), 
                (1,20), (100,1e6), (1e-12,10)]
    if options == None:
        options={'maxiter':5000, 'maxfev':5000, 'xatol':0.001, 'disp': True},

    ee = np.array([100, 100, 100, 100, 200, 200, 200, 200, 400, 400, 400, 400,
                   600, 600, 600, 600,800, 800, 800, 800, 1000, 1000, 1000, 
                   1000, 1100, 1100, 1100, 1100]).T
    tc = np.array([15, 25, 50, 75,15, 25, 50, 75,15, 25, 50, 75,15, 25, 50, 75,
                   15, 25, 50, 75,15, 25, 50, 75,15, 25, 50, 75]).T
    

    cec_params = pvlib.pvsystem.calcparams_cec(ee, tc, alpha_sc, 
                                a_ref,I_L_ref,I_o_ref,R_sh_ref,R_s,Adjust)
    cec_ivs = pvlib.pvsystem.singlediode(*cec_params)
    
    def pvs_objfun(pvs_mod,cec_ivs, ee, tc, cs, r_sh_exp, egref):
        I_L_ref=pvs_mod[3]
        I_o_ref=pvs_mod[4]
        gamma_ref=pvs_mod[1]
        mu_gamma=pvs_mod[2]
        R_sh_mult=pvs_mod[5]
        R_s=pvs_mod[7]
        R_sh_ref=pvs_mod[6]
        alpha_sc=pvs_mod[0]
        R_sh_0=R_sh_ref*R_sh_mult

        pvs_params=pvlib.pvsystem.calcparams_pvsyst(ee, tc, alpha_sc, 
                    gamma_ref, mu_gamma, I_L_ref, I_o_ref, R_sh_ref, R_sh_0,
                    R_s, cs, R_sh_exp=r_sh_exp, EgRef = egref)
        
        pvsyst_ivs=pvlib.pvsystem.singlediode(photocurrent=pvs_params[0],
            saturation_current=pvs_params[1], resistance_series=pvs_params[2], 
            resistance_shunt=pvs_params[3], nNsVth=pvs_params[4])
    
        isc_diff=abs((pvsyst_ivs['i_sc']-cec_ivs['i_sc'])/cec_ivs['i_sc']).mean()
        imp_diff=abs((pvsyst_ivs['i_mp']-cec_ivs['i_mp'])/cec_ivs['i_mp']).mean()
        voc_diff=abs((pvsyst_ivs['v_oc']-cec_ivs['v_oc'])/cec_ivs['v_oc']).mean()
        vmp_diff=abs((pvsyst_ivs['v_mp']-cec_ivs['v_mp'])/cec_ivs['v_mp']).mean()
        pmp_diff=abs((pvsyst_ivs['p_mp']-cec_ivs['p_mp'])/cec_ivs['p_mp']).mean()
        
        mean_diff = ((isc_diff+imp_diff+voc_diff+vmp_diff+pmp_diff)/5)
        
        return mean_diff

    pvs_params = minimize(pvs_objfun, initial,
                    args=(cec_ivs, ee, tc, NcelS ,5.5, 1.121), 
                    method=method, bounds=bounds, 
                    options=options)['x']    

    return tuple([pvs_params[0], pvs_params[1],pvs_params[2],pvs_params[3],
                  pvs_params[4],pvs_params[6],(pvs_params[5]*pvs_params[6]),
                  pvs_params[7]])