import numpy as np
from scipy.stats import maxwell
import rebound, reboundx
from saturn_system import a_sat, m_sat, a_saturn, m_sun, R_eq_saturn, J2_sat, J2_inner_satellites

solar_inc = np.radians(26.7)
solar_Omega = np.radians(0)

class HyperionDamping:
    def __init__(self, damping_type: str, Q_hyp: float, k2_hyp: float, R_hyp: float, time_scaling: float, rot_fit: tuple[float, float]):
        self.damping_type = damping_type
        self.Q_hyp, self.k2_hyp, self.R_hyp = Q_hyp, k2_hyp, R_hyp
        self.time_scaling = time_scaling
        self.rot_fit = rot_fit

    def compute_tau_e(self, p):
        if np.isposinf(self.Q_hyp):
            return np.inf
        if self.damping_type == 'constant':
            return (-21/2 * self.k2_hyp/self.Q_hyp / m_sat['hyperion'] * (self.R_hyp/p.a)**5 * p.n)**-1 / self.time_scaling
        elif self.damping_type == 'constant_NPA':
            return p.e**2 * (-21/2 * self.k2_hyp/self.Q_hyp / m_sat['hyperion'] * (self.R_hyp/p.a)**5 * p.n)**-1 / self.time_scaling
        elif self.damping_type == 'realistic_NPA':
            omega = self.rot_fit[0]*self.rot_fit[1]**(p.e/0.1)
            return p.e**2 * omega**-4 * (-21/2 * self.k2_hyp/self.Q_hyp / m_sat['hyperion'] * (self.R_hyp/p.a)**5 * p.n)**-1 / self.time_scaling

class TitanMigration:
    def __init__(self, Q_over_k2: float, tau_a: float, t_lock: float, time_scaling: float):
        if np.isfinite([Q_over_k2, tau_a]).sum() != 1:
            raise ValueError('must specify either Q_over_k2 or tau_a')
        self.time_scaling = time_scaling
        if np.isfinite(Q_over_k2):
            self.Q_over_k2 = Q_over_k2/time_scaling
            self.mig_type = 'constant_Q'
        elif np.isfinite(tau_a):
            self.tau_a = tau_a/time_scaling
            if np.isfinite(t_lock):
                self.t_lock = t_lock/time_scaling
                self.mig_type = 'realistic_lock'
            else:
                self.mig_type = 'constant_tau'

    def compute_titan_mig(self, p: rebound.Particle, t: float) -> tuple[float, float]:
        if self.mig_type == 'constant_Q':
            tau_a = self.Q_over_k2/3/m_sat['titan']*(p.a/R_eq_saturn)**5 / p.n
            return tau_a, -tau_a
        elif self.mig_type == 'constant_tau':
            return self.tau_a, -self.tau_a
        elif self.mig_type == 'realistic_lock':
            if t < self.t_lock:
                return np.inf, -np.inf
            else:
                return t/self.B, -t/self.B
        
    def compute_initial_a(self, duration, G) -> float:
        if self.mig_type == 'constant_Q':
            return (a_sat['titan']**(13/2) - 39/2/self.Q_over_k2*m_sat['titan']*np.sqrt(G)*R_eq_saturn**5*duration)**(2/13)
        elif self.mig_type == 'constant_tau':
            return a_sat['titan']*np.exp(-duration/self.tau_a)
        elif self.mig_type == 'realistic_lock':
            self.B = duration/self.tau_a
            return a_sat['titan']*(self.t_lock/duration)**self.B


class Simulation():
    def __init__(self, n, t_end, time_scaling, init_per_rats, init_type, init_param, titan_ecc, Q_over_k2, titan_tau_a, t_lock, J2=True, sun=True, 
                 hyp_damping_type='realistic_NPA', Q_hyp=100, k2_hyp=1.5e-3, R_hyp=150, rot_fit=(2.88, 1.37)):
        self.titan_mig = TitanMigration(Q_over_k2=Q_over_k2, tau_a=titan_tau_a, t_lock=t_lock, time_scaling=time_scaling)
        self.hyp_damping = HyperionDamping(damping_type=hyp_damping_type, Q_hyp=Q_hyp, k2_hyp=k2_hyp, R_hyp=R_hyp, time_scaling=time_scaling, rot_fit=rot_fit)
        self.n, self.t_end, self.time_scaling = n, t_end, time_scaling
        self.init_per_rats, self.init_type, self.init_param = init_per_rats, init_type, init_param
        self.titan_ecc, self.J2, self.sun = titan_ecc, J2, sun

    def __repr__(self) -> str:
        if self.titan_mig.mig_type == 'constant_Q':
            titan_mig_str = self.titan_mig.mig_type + '_' + str(self.titan_mig.Q_over_k2*self.time_scaling)
        elif self.titan_mig.mig_type == 'constant_tau':
            titan_mig_str = self.titan_mig.mig_type + f'_{self.titan_mig.tau_a*self.time_scaling:.0g}'
        elif self.titan_mig.mig_type == 'realistic_lock':
            titan_mig_str = self.titan_mig.mig_type + f'_{self.titan_mig.tau_a*self.time_scaling:.0g}_{self.titan_mig.t_lock*self.time_scaling:.0g}'
        hyp_damp_str = f'hyp_damp_{self.hyp_damping.damping_type}_{self.hyp_damping.Q_hyp}'
        return f'{titan_mig_str}_{hyp_damp_str}'

    def set_up_sim(self) -> tuple[rebound.Simulation, list[str]]:
        self.times = np.arange(0, self.t_end, 0.5)
        self.N_times = len(self.times)
        sim = rebound.Simulation()
        sim.units = ('msaturn', 'km', 'yr')
        sim.add(m=1)
        self.G = sim.G
        titan_init_a = self.titan_mig.compute_initial_a(self.times[-1], sim.G)
        assert np.isfinite(titan_init_a), 'Initial Titan sma is not finite'
        print(f'Q = {self.hyp_damping.Q_hyp} Initial Titan sma: {titan_init_a/R_eq_saturn:.3f} R_eq')
        sim.add(m=m_sat['titan'], a=titan_init_a, e=self.titan_ecc, inc=np.radians(0.306), Omega='uniform')
        ps = sim.particles
        hyp_hashes = [f'hyperion_{j}' for j in range(self.n)]
        if self.init_type == 'uniform':
            init_es = np.logspace(-3, np.log10(self.init_param), self.n)
            init_is = np.linspace(0, 0, self.n)
            for (init_per_rat, e, inc, hyp_hash) in zip(self.init_per_rats, init_es, init_is, hyp_hashes):
                sim.add(m=0, a=titan_init_a*init_per_rat**(2/3), e=e, inc=inc, 
                        Omega='uniform', pomega='uniform', l='uniform', hash=hyp_hash, primary=ps[0])
        elif self.init_type == 'maxwell':
            r = self.init_param*maxwell.rvs(scale=1, size=self.n)
            phi = np.random.uniform(0, 2*np.pi, self.n)
            costheta = np.random.uniform(-1, 1, self.n)
            theta = np.arccos(costheta)
            x = r*np.sin(theta)*np.cos(phi)
            y = r*np.sin(theta)*np.sin(phi)
            z = r*np.cos(theta)
            l_init = np.random.uniform(0, 2*np.pi)
            for (j, init_per_rat, hyp_hash) in zip(range(self.n), self.init_per_rats, hyp_hashes):
                sim.add(m=0, a=titan_init_a*init_per_rat**(2/3), e=0, inc=0, Omega=0, pomega=0, l=l_init, hash=hyp_hash, primary=ps[0])
                v = ps[hyp_hash].v
                ps[hyp_hash].vx += v*x[j]
                ps[hyp_hash].vy += v*y[j]
                ps[hyp_hash].vz += v*z[j]
        else:
            raise ValueError(f'init_type {self.init_type} not recognized')
        if self.sun:
            sim.add(m=m_sun, a=a_saturn, inc=solar_inc, Omega=solar_Omega)
        sim.move_to_com()
        ps = sim.particles
        rebx = reboundx.Extras(sim)
        mof = rebx.load_force("modify_orbits_forces")
        rebx.add_force(mof)
        if self.J2:
            gh = rebx.load_force("gravitational_harmonics")
            rebx.add_force(gh)
            ps[0].params['J2'] = J2_sat + J2_inner_satellites
            ps[0].params['R_eq'] = R_eq_saturn
        sim.integrator = 'whfast'
        sim.dt = min(p.P for p in ps[1:])/20
        print(f'Q = {self.hyp_damping.Q_hyp} Integrator timestep: {sim.dt:.3f} yr')
        return sim, hyp_hashes

    def run_sim(self):
        sim, hyp_hashes = self.set_up_sim()
        ps = sim.particles
        self.titan_orbit = np.full((self.N_times,7), np.nan)
        self.hyp_orbits = np.full((self.N_times, self.n,7), np.nan)
        for i, time in enumerate(self.times):
            titan_tau_a, titan_tau_e = self.titan_mig.compute_titan_mig(ps[1], sim.t)
            ps[1].params['tau_a'], ps[1].params['tau_e'] = titan_tau_a, titan_tau_e
            while True:
                try:
                    sim.integrate(time)
                    break
                except rebound.Escape:
                    for hyp_hash in hyp_hashes:
                        if ps[hyp_hash]**ps[0] > sim.exit_max_distance:
                            break
                    sim.remove(hash=hyp_hash)
                    hyp_hashes.remove(hyp_hash)
            self.titan_orbit[i] = ps[1].a, ps[1].e, ps[1].inc, ps[1].pomega, ps[1].Omega, ps[1].l, titan_tau_a
            for hyp_hash in hyp_hashes:
                p = ps[hyp_hash]
                orbit = p.calculate_orbit()
                hyp_tau_e = self.hyp_damping.compute_tau_e(p)
                p.params['tau_e'] = hyp_tau_e
                self.hyp_orbits[i, int(hyp_hash.split('_')[-1])] = orbit.a, orbit.e, orbit.inc, orbit.pomega, orbit.Omega, orbit.l, hyp_tau_e
            if i % (self.N_times//4) == 0:
                print(f'Q = {self.hyp_damping.Q_hyp} {i/self.N_times*100:.1f}% complete')
        print(f'Q = {self.hyp_damping.Q_hyp} 100% complete')
