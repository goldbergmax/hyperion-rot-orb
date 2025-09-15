import argparse
import os
from pathlib import Path
import numpy as np
import rebound
import reboundx
from multiprocessing import Pool
import pickle
from itertools import product
from saturn_system import a_sat, m_sat, R_eq_saturn, J2_sat, J2_inner_satellites, a_saturn

def titan_iapetus_resonance(titan_ecc=0.028, iapetus_ecc=0.0293, init_a=0.995, tau_a=1e10, tau_e=np.inf, J2=True, sun=False):
    sim = rebound.Simulation()
    sim.units = ('msaturn', 'km', 'yr')
    sim.add(m=1)
    sim.add(m=m_sat['titan'], a=a_sat['titan']*init_a, e=titan_ecc, inc=np.radians(0.306), Omega='uniform', pomega='uniform', l='uniform')
    sim.add(m=m_sat['iapetus'], a=a_sat['iapetus'], e=iapetus_ecc, inc=np.radians(15.20-8), Omega=solar_Omega, pomega='uniform', l='uniform')
    rebx = reboundx.Extras(sim)
    mof = rebx.load_force("modify_orbits_forces")
    rebx.add_force(mof)
    gh = rebx.load_force("gravitational_harmonics")
    rebx.add_force(gh)
    sim.move_to_com()
    ps = sim.particles
    if J2:
        ps[0].params['J2'] = J2_sat + J2_inner_satellites
        ps[0].params['R_eq'] = R_eq_saturn
    if sun:
        solar = rebx.load_force('solar_force')
        rebx.add_force(solar)
        solar.params['sf_inc'] = solar_inc
        solar.params['sf_Omega'] = solar_Omega
        solar.params['sf_a'] = solar_a
        solar.params['sf_m'] = solar_m
        solar.params['sf_P'] = (4*np.pi**2*solar_a**3/(sim.G*solar_m))**0.5 
    sim.integrator = 'whfast'
    sim.dt = ps[1].P/20
    try:
        len(tau_a)
        assert len(tau_a) == N, 'tau_a must be a scalar or a list of length N'
        tau_as = tau_a
    except TypeError:
        tau_as = [tau_a]*N
    try:
        len(tau_e)
        assert len(tau_e) == N, 'tau_e must be a scalar or a list of length N'
        tau_es = tau_e
    except TypeError:
        tau_es = [tau_e]*N
    titan_orbit = np.zeros((N,6))
    iap_orbit = np.zeros((N,6))
    for i, time in enumerate(times):
        ps[1].params['tau_a'] = tau_as[i]
        ps[1].params['tau_e'] = tau_es[i]
        sim.integrate(time)
        titan_orbit[i] = ps[1].a, ps[1].e, ps[1].inc, ps[1].pomega, ps[1].Omega, ps[1].l
        iap_orbit[i] = ps[2].a, ps[2].e, ps[2].inc, ps[2].pomega, ps[2].Omega, ps[2].l
    return times, titan_orbit, iap_orbit

parser = argparse.ArgumentParser()
parser.add_argument('--N_sims', type=int, default=25)
parser.add_argument('--t_end', type=float, default=50e6)
parser.add_argument('--init_a', type=float, default=0.995)
parser.add_argument('--iap_ecc', type=float, default=0.05)
parser.add_argument('--tau_a', type=float, default=1e10)
parser.add_argument('--tau_e', type=float, default=np.inf)
parser.add_argument('--J2', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--sun', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

solar_m = 3499
solar_a = a_saturn
solar_inc = np.radians(26.7)
solar_Omega = np.radians(0)
times = np.arange(0, args.t_end, 1000)
N = len(times)

pool = Pool(31)
titan_orbits = []
iap_orbits = []
func_args = product([0.005, 0.03, 0.05], [args.iap_ecc]*(args.N_sims//3), [args.init_a], [args.tau_a], [args.tau_e], [args.J2], [args.sun])
results = pool.starmap(titan_iapetus_resonance, func_args)
sun_path = '_sun' if args.sun else ''
write_dir = Path(f'/mnt/data-big/mgoldberg/satellites/titan_iapetus_capture/e_iap_{args.iap_ecc:.2f}' + sun_path)
os.makedirs(write_dir, exist_ok=True)
pickle.dump(results, open(write_dir / f'titan_iapetus_capture_tau_a_{args.tau_a:.0e}.p', 'wb'))