#retrieved from JPL Horizons on 2022-12-13

R_eq_saturn = 60268
a_saturn = 1.432e9
hill_saturn = 61.6e6
a_sat = {}
a_sat['mimas'] = 185540
a_sat['enceladus'] = 238040
a_sat['tethys'] = 294670
a_sat['dione'] = 377420
a_sat['rhea'] = 527070
a_sat['titan'] = 1221870
a_sat['hyperion'] = 1481009
a_sat['iapetus'] = 3560840

m_saturn_kg = 5.6834e26
m_sat = {}
m_sat['mimas'] = 3.75e19/m_saturn_kg
m_sat['enceladus'] = 10.805e19/m_saturn_kg
m_sat['tethys'] = 61.76e19/m_saturn_kg
m_sat['dione'] = 109.572e19/m_saturn_kg
m_sat['rhea'] = 230.9e19/m_saturn_kg
m_sat['titan'] = 13455.3e19/m_saturn_kg
m_sat['hyperion'] = 0.5551e19/m_saturn_kg
m_sat['iapetus'] = 180.59e19/m_saturn_kg
m_sun = 3499

J2_sat = 0.0163
J2_inner_satellites = 0.5*sum([m_sat[sat]*(a_sat[sat]/R_eq_saturn)**2 for sat in ['mimas', 'enceladus', 'tethys', 'dione', 'rhea']])
J2_titan = 0.5*m_sat['titan']*(a_sat['titan']/R_eq_saturn)**2
