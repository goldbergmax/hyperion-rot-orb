import numpy as np
from scipy.integrate import solve_ivp
from rebound import M_to_f

def euler_to_quat(theta, phi, psi):
    lambda0 = np.cos(phi/2)*np.cos((theta+psi)/2)
    lambda1 = np.sin(phi/2)*np.cos((theta-psi)/2)
    lambda2 = np.sin(phi/2)*np.sin((theta-psi)/2)
    lambda3 = np.cos(phi/2)*np.sin((theta+psi)/2)
    return lambda0, lambda1, lambda2, lambda3

def quat_to_euler(lambda0, lambda1, lambda2, lambda3):
    theta_quat = np.arctan2(lambda1*lambda3 + lambda0*lambda2, 
                            lambda0*lambda1 - lambda2*lambda3)
    phi_quat = 2*np.arctan2(np.sqrt(lambda1**2 + lambda2**2), 
                            np.sqrt(lambda0**2 + lambda3**2))
    psi_quat = np.arctan2(lambda1*lambda3 - lambda0*lambda2,
                          lambda0*lambda1 + lambda2*lambda3)
    return theta_quat, phi_quat, psi_quat

def direction_cosines_quat(f, lambda0, lambda1, lambda2, lambda3):
    alpha = (lambda0**2 + lambda1**2 - lambda2**2 - lambda3**2)*np.cos(f) + 2*(lambda0*lambda3 + lambda1*lambda2)*np.sin(f)
    beta = 2*(lambda1*lambda2 - lambda0*lambda3)*np.cos(f) + (lambda0**2 - lambda1**2 + lambda2**2 - lambda3**2)*np.sin(f)
    gamma = 2*(lambda0*lambda2 + lambda1*lambda3)*np.cos(f) + 2*(-lambda0*lambda1 + lambda2*lambda3)*np.sin(f)
    quat_norm_sq = lambda0**2 + lambda1**2 + lambda2**2 + lambda3**2
    alpha /= quat_norm_sq
    beta /= quat_norm_sq
    gamma /= quat_norm_sq
    return alpha, beta, gamma

def direction_cosines_euler(f, theta, phi, psi):
    alpha = np.cos(theta - f)*np.cos(psi) - np.sin(theta - f)*np.cos(phi)*np.sin(psi)
    beta = -np.cos(theta - f)*np.sin(psi) - np.sin(theta - f)*np.cos(phi)*np.cos(psi)
    gamma = np.sin(theta - f)*np.sin(phi)
    return alpha, beta, gamma

class RotationalState():
    """
    Represent a rotational state in either Euler angles or quaternions.
    """
    def __init__(self, state, t=0.0, A=None, B=None, C=None):
        """
        state: array-like with first axis of length 6 or 7, depending on whether Euler angles or quaternions are used
        """
        self.t = t
        self.A, self.B, self.C = A, B, C
        _state = np.asarray(state)
        if _state.shape[0] == 6:
            self.theta, self.phi, self.psi, self.omegaa, self.omegab, self.omegac = _state
            self.lambda0, self.lambda1, self.lambda2, self.lambda3 = euler_to_quat(self.theta, self.phi, self.psi)
        elif _state.shape[0] == 7:
            self.lambda0, self.lambda1, self.lambda2, self.lambda3, self.omegaa, self.omegab, self.omegac = _state
            self.theta, self.phi, self.psi = quat_to_euler(self.lambda0, self.lambda1, self.lambda2, self.lambda3)
        else:
            raise ValueError("Invalid number of parameters")
        
    @property
    def euler(self):
        return [self.theta, self.phi, self.psi, self.omegaa, self.omegab, self.omegac]
    
    @property
    def quat(self):
        return [self.lambda0, self.lambda1, self.lambda2, self.lambda3, self.omegaa, self.omegab, self.omegac]
    
    @property
    def omega(self):
        return np.linalg.norm([self.omegaa, self.omegab, self.omegac], axis=0)
    
    def cosines(self, f):
        return direction_cosines_quat(f, self.lambda0, self.lambda1, self.lambda2, self.lambda3)

    def cut_NPA(self, threshold, window_size=200):
        self.window_size = window_size
        self.omega_reshape = self.omega.reshape(-1, window_size)
        self.omega_std = np.std(self.omega_reshape, axis=1)
        self.npa_mask = np.repeat(self.omega_std > threshold, window_size)
        self.omega_npa = self.omega[self.npa_mask]
        self.omega_pa = self.omega[~self.npa_mask]

    @property
    def omega_prec(self):
        omega_prec = {}
        omega_prec['A'] = (np.sqrt(self.B*self.C)-self.A)/np.sqrt(self.B*self.C)*self.omegaa
        omega_prec['B'] = (np.sqrt(self.A*self.C)-self.B)/np.sqrt(self.A*self.C)*self.omegab
        omega_prec['C'] = (np.sqrt(self.A*self.B)-self.C)/np.sqrt(self.A*self.B)*self.omegac
        return omega_prec


class RotationSimulation():
    def __init__(self, A, B, C, e_forced, e_free=0, e_period=np.inf, n=1.0, M0=0.0):
        self.A = A
        self.B = B
        self.C = C
        self.e_forced = e_forced
        self.e_free = e_free
        self.e_period = e_period
        self.n = n
        self.M0 = M0

    def get_e(self, t):
        return self.e_forced + self.e_free*np.sin(t*2*np.pi/self.e_period)

    def get_rf(self, t):
        e = self.get_e(t)
        M = np.remainder(self.n*t, 2*np.pi) + self.M0
        f = M_to_f(e, M)
        r = (1 - e**2)/(1 + e*np.cos(f))
        return r, f

    def deriv_euler(self, t, y):
        theta, phi, psi, omegaa, omegab, omegac = y
        r, f = self.get_rf(t)

        theta_dot = (omegaa*np.sin(psi) + omegab*np.cos(psi))/np.sin(phi)
        phi_dot = omegaa*np.cos(psi) - omegab*np.sin(psi)
        psi_dot = omegac - theta_dot*np.cos(phi)

        alpha, beta, gamma = direction_cosines_euler(f, theta, phi, psi)
        omegaa_dot = (self.B-self.C)/self.A*(omegab*omegac - 3*beta*gamma/r**3)
        omegab_dot = (self.C-self.A)/self.B*(omegac*omegaa - 3*gamma*alpha/r**3)
        omegac_dot = (self.A-self.B)/self.C*(omegaa*omegab - 3*alpha*beta/r**3)
        return [theta_dot, phi_dot, psi_dot, omegaa_dot, omegab_dot, omegac_dot]

    def deriv_quat(self, t, y):
        lambda0, lambda1, lambda2, lambda3, omegaa, omegab, omegac = y
        r, f = self.get_rf(t)
        lambda0_dot = 0.5*(-lambda1*omegaa - lambda2*omegab - lambda3*omegac)
        lambda1_dot = 0.5*(lambda0*omegaa - lambda3*omegab + lambda2*omegac)
        lambda2_dot = 0.5*(lambda3*omegaa + lambda0*omegab - lambda1*omegac)
        lambda3_dot = 0.5*(-lambda2*omegaa + lambda1*omegab + lambda0*omegac)

        alpha, beta, gamma = direction_cosines_quat(f, lambda0, lambda1, lambda2, lambda3)
        omegaa_dot = (self.B-self.C)/self.A*(omegab*omegac - 3*beta*gamma/r**3)
        omegab_dot = (self.C-self.A)/self.B*(omegac*omegaa - 3*gamma*alpha/r**3)
        omegac_dot = (self.A-self.B)/self.C*(omegaa*omegab - 3*alpha*beta/r**3)

        return [lambda0_dot, lambda1_dot, lambda2_dot, lambda3_dot, omegaa_dot, omegab_dot, omegac_dot]
    
    def integrate(self, ts: np.ndarray, initial_state: RotationalState, int_method='DOP853', coords='quat', rtol=1e-6, atol=1e-10):
        if coords == 'quat':
            self._sol = solve_ivp(self.deriv_quat, [ts[0], ts[-1]], initial_state.quat, 
                                 t_eval=ts, rtol=rtol, atol=atol, method=int_method)
        elif coords == 'euler':
            self._sol = solve_ivp(self.deriv_euler, [ts[0], ts[-1]], initial_state.euler, 
                                 t_eval=ts, rtol=rtol, atol=atol, method=int_method)
        else:
            raise ValueError("Invalid integration coordinates")
        self.sol = RotationalState(self._sol.y, self._sol.t, self.A, self.B, self.C)
