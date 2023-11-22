import numpy as np
import argparse
from multiprocessing import Pool
import pickle
from rotation_tools import RotationSimulation, RotationalState

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=1000)
parser.add_argument('--spher_scale', type=float, default=1.0)
parser.add_argument('--method', type=str, default='DOP853')
parser.add_argument('--output', type=str, default='')
cli_args = parser.parse_args()

A, B, C = 0.314, 0.474, 0.542 # from Harbison et al. (2011)
# setting spher_scale less than 1.0 makes the ellipsoid more spherical
avg_mom = (A+B+C)/3
A -= (1-cli_args.spher_scale)*(A-avg_mom)
B -= (1-cli_args.spher_scale)*(B-avg_mom)
C -= (1-cli_args.spher_scale)*(C-avg_mom)

e_period = 2022.5 # secular oscillation period in units where hyperion's orbital period is 2pi
M0 = np.radians(295)
P = 2*np.pi
n = 2*np.pi/P
ts = np.arange(0, cli_args.N*P, P)
true_P = 21.28

theta0, phi0, psi0 = 0.004, 1.441, 0.427
# omega0 = 4.433
omega0 = 1
omegaa0 = omega0*0.890
omegab0 = omega0*0.067
omegac0 = omega0*0.451
initial_state = RotationalState([theta0, phi0, psi0, omegaa0, omegab0, omegac0])

args = [(0.010,0.0), (0.0158,0.0), (0.0251,0.0), (0.0398,0.0), (0.0631,0.0),
        (0.100,0.0), (0.1580,0.0), (0.2510,0.0), (0.3980,0.0), (0.6310,0.0)]
sims = [RotationSimulation(A, B, C, e_forced, e_free, e_period, n, M0) for e_forced, e_free in args]
def solve_rot(sim: RotationSimulation):
    sim.integrate(ts, initial_state, int_method=cli_args.method)
    return sim
pool = Pool(20)
sims = pool.map(solve_rot, sims)
output = '_' + cli_args.output if cli_args.output else ''
pickle.dump(sims, open(f'/mnt/data-big/mgoldberg/satellites/hyperion_rotation/sols_{cli_args.spher_scale}{output}.p', 'wb'))
