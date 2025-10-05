# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:56:49 2025

@author: lmdevil
"""

def pvsyst_to_cec(alpha_sc,gamma_ref,mu_gamma,I_L_ref,I_o_ref,R_sh_ref,R_sh_0,R_s,NcelS,
                  initial=None,
                  bounds=None,
                  options=None,
                  method='Nelder-Mead',
                  R_sh_exp=5.5, EgRef=1.121): 
    
  '''
    This function translates PVsyst parameter values to CEC parameter values
    using methods described by Deville et al [1]

    Parameters
    ----------
    
    alpha_sc : float
         The short-circuit current temperature coefficient of the
         module in units of A/C.
         
         
  gamma_ref : float
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at reference
        conditions, in units of V.
        
   mu_gamma : float
        The temperature coefficient for the diode ideality factor, 1/K
        
   I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions,
        in amperes.

   I_o_ref : float
        The dark or diode reverse saturation current at reference conditions,
        in amperes.

   R_sh_ref : float
        The shunt resistance at reference conditions, in ohms.

   R_sh_0 : float
        The series resistance at zero irradiance conditions, in ohms.

   R_s : float
        The series resistance at reference conditions, in ohms
        
   NCels : integer
        The number of cells connected in series

   initial : list
        A list defining the initial guess for the CEC parameters.
        See the notes section for more details.

   bounds : list
        A list defining the upper and lower bounds for the CEC parameters
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


    Returns
    -------

    I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions,
        in amperes.

    I_o_ref : float
        The dark or diode reverse saturation current at reference conditions,
        in amperes.
        
    a_ref : float
         The product of the usual diode ideality factor (n, unitless),
         number of cells in series (Ns), and cell thermal voltage at reference
         conditions, in units of V.
         
    R_sh_ref : float
        The shunt resistance at reference conditions, in ohms.

    R_s : float
        The series resistance at reference conditions, in ohms.
        
    Adjust : float
        The adjustment to the temperature coefficient for short circuit
        current, in percent


    Notes
    -----
    The order of the parameters in the initial guess & bounds series:
    [I_L_ref, I_o_ref, a_ref, R_sh_ref, R_s, Adjust]
    


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
    
    
    
    pvs_params = pvlib.pvsystem.calcparams_pvsyst(ee, tc, alpha_sc,gamma_ref,
                    mu_gamma, I_L_ref, I_o_ref, R_sh_ref, R_sh_0, R_s, NCelS, 
                    R_sh_exp, EgRef)

    pvsyst_ivs = pvlib.pvsystem.singlediode(*pvs_params)
    
    
    def cec_objfun(cec_mod,pvs_ivs, ee, tc, alpha_sc):
    
        I_L_ref = cec_mod[0]
        I_o_ref = cec_mod[1]
        a_ref = cec_mod[2]
        R_sh_ref = cec_mod[3]
        R_s = cec_mod[4]
        alpha_sc = alpha_sc
        Adjust = cec_mod[5]
        
        cec_params = pvlib.pvsystem.calcparams_cec(ee, tc,alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s, Adjust)
        cec_ivs = pvlib.pvsystem.singlediode(*cec_params)

        isc_rss = np.sqrt(sum((cec_ivs['i_sc']-pvs_ivs['i_sc'])**2))
        imp_rss = np.sqrt(sum((cec_ivs['i_mp']-pvs_ivs['i_mp'])**2))
        voc_rss = np.sqrt(sum((cec_ivs['v_oc']-pvs_ivs['v_oc'])**2))
        vmp_rss = np.sqrt(sum((cec_ivs['v_mp']-pvs_ivs['v_mp'])**2))
        pmp_rss = np.sqrt(sum((cec_ivs['p_mp']-pvs_ivs['p_mp'])**2))
    
        mean_diff = ((isc_rss+imp_rss+voc_rss+vmp_rss+pmp_rss)/5)
        
        return mean_diff
    
    
    cec_params = minimize(cec_objfun, initial, args=(pvsyst_ivs, ee, tc, 
                alpha_sc), method=method, bounds=bounds,options=options)['x']
    
    return tuple([cec_params[0], cec_params[1],cec_params[2],cec_params[3],
                  cec_params[4],cec_params[5]])