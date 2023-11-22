import argparse
from multiprocessing import Pool
from pathlib import Path
import pickle
from hyp_capture import Simulation
import numpy as np

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument('--Q_over_k2', type=float, default=np.inf)
group.add_argument('--titan_tau_a', type=float, default=np.inf)
parser.add_argument('--t_lock', type=float, default=np.inf)
parser.add_argument('--scaling', type=float)
parser.add_argument('--J2', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--sun', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--titan_ecc', type=float)
parser.add_argument('--init_per_rat', type=float)
parser.add_argument('--init_param', type=float)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--t_end', type=float)
parser.add_argument('--init', choices=['uniform', 'maxwell'], default='uniform')
parser.add_argument('--hyp_damping', choices=['constant', 'constant_NPA', 'realistic_NPA'])
parser.add_argument('--hyp_Q', type=str, default='100')
parser.add_argument('--threads', type=int, default=1)
args = parser.parse_args()

hyp_Qs = [float(Q) for Q in args.hyp_Q.split(',')]
def run_sim(Q_hyp):
    hyp_cap = Simulation(n=args.n, t_end=args.t_end, time_scaling=args.scaling, init_per_rats=args.n*[args.init_per_rat], init_type=args.init, 
                         init_param=args.init_param, titan_ecc=args.titan_ecc, Q_over_k2=args.Q_over_k2, titan_tau_a=args.titan_tau_a, t_lock=args.t_lock,
                         J2=args.J2, sun=args.sun, hyp_damping_type=args.hyp_damping, Q_hyp=Q_hyp, k2_hyp=1.5e-3, R_hyp=150)
    hyp_cap.run_sim()
    save_dir = Path(f'/mnt/data-big/mgoldberg/satellites/hyperion_capture/{args.init}_{args.init_param}_titan_e_{args.titan_ecc}_per_{args.init_per_rat}/')
    save_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump(hyp_cap, open(save_dir / f'{hyp_cap}.pkl', 'wb'))

pool = Pool(args.threads)
pool.map(run_sim, hyp_Qs)
