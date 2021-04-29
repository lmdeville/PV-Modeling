# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 08:10:33 2021

@author: lmdevil
"""

def sys_met_data(mettablename, systablename, start, end):
    engine = datatools.database.create_mss_engine()
    
    met_sql = f"select * from {mettablename} where TmStamp between '{start}' and '{end}';"
    met = pd.read_sql(met_sql, engine, index_col='TmStamp')
    met.index = pd.DatetimeIndex(met.index)
    met.index = met.index.tz_localize('MST') 
    
    sys_sql = f"select * from {systablename} where TmStamp between '{start}' and '{end}';"
    sys = pd.read_sql(sys_sql, engine, index_col='TmStamp')
    sys.index = pd.DatetimeIndex(sys.index)
    sys.index = sys.index.tz_localize('MST')

    df = pd.merge(met,sys,how='inner', left_index=True, right_index=True)
    
    return df

def calc_sol_data(time, latitude, longitude, tilt, pressure):
    sdf = pvlib.solarposition.get_solarposition(time, latitude, longitude, pressure)
    sdf['dni_extra'] = pvlib.irradiance.get_extra_radiation(time)
    return sdf


def  calc_env_data(tilt, altitude, sur_azimuth, sol_azimuth, zenith, DNI, GHI, DHI, DNI_extra, model):
    edf = pvlib.irradiance.get_total_irradiance(tilt, sur_azimuth, zenith, sol_azimuth,
                                                               DNI, GHI, DHI, DNI_extra, model)
    edf['aoi'] = pvlib.irradiance.aoi(tilt, sur_azimuth, zenith, sol_azimuth)
    edf['airmass'] = pvlib.atmosphere.get_relative_airmass(zenith)
    edf['pressure'] = pvlib.atmosphere.alt2pres(altitude)
    return edf


def meas_val(str_v, str_i):
    str_p = str_v*str_i
    results = str_p.resample('H').mean()
    return results


def sapm_param(POA_GHI, POA_DNI, POA_DHI, amb_temp, ws_avg, am_abs, aoi, module):
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    sapm_p = pvlib.temperature.sapm_cell(POA_GHI,
            amb_temp, ws_avg,
            **temperature_model_parameters)
    sapm_p = sapm_p.to_frame()
    sapm_p = sapm_p.rename(columns = {0 : 'tcell'})
    sapm_p['eff_irr'] = pvlib.pvsystem.sapm_effective_irradiance(
           POA_DNI, POA_DHI, am_abs, aoi, module)
    return sapm_p

def sapm(eff_irr, tcell, module):
    dc = pvlib.pvsystem.sapm(eff_irr, tcell, module)
    return dc

def sapm_p(v_mp, i_mp):
    str_v = v_mp*module['str_len']
    str_p = i_mp*str_v
    h_str_p = str_p.resample('H').mean()
    return h_str_p

def pvwatts(POA_eff, cell_temp, stc_mod_p, Gpmp, temp_ref):
    p = pvlib.pvsystem.pvwatts_dc(POA_eff, cell_temp, stc_mod_p, Gpmp, temp_ref)
    str_p = p*module['str_len']
    h_str_p = str_p.resample('H').mean()
    return h_str_p

def sdm(p_mp):
    str_p = p_mp*module['str_len']
    h_str_p = str_p.resample('H').mean()
    return h_str_p
