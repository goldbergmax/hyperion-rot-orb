import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R
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
    
    @classmethod
    def from_euler(cls, angles, omega_body=None, omega_lab=None, t=0.0, A=None, B=None, C=None):
        if omega_body is not None:
            omegaa, omegab, omegac = omega_body
        elif omega_lab is not None:
            r_body_to_lab = R.from_euler('ZXZ', angles)
            omega_body = r_body_to_lab.inv().apply(omega_lab)
            omegaa, omegab, omegac = omega_body
        else:
            raise ValueError("Either omega_body or omega_lab must be provided")
        theta, phi, psi = angles
        return cls([theta, phi, psi, omegaa, omegab, omegac], t, A, B, C)
    
    @classmethod
    def from_quat(cls, lambda0, lambda1, lambda2, lambda3, omegaa, omegab, omegac, t=0.0, A=None, B=None, C=None):
        return cls([lambda0, lambda1, lambda2, lambda3, omegaa, omegab, omegac], t, A, B, C)
    
    @classmethod
    def from_andoyer(cls, G, Lambda, L, g, lambda_node, l, A, B, C, permute_axes=np.eye(3)):
        r_ang_mom_to_lab = R.from_euler('ZX', np.array([lambda_node, np.arccos(Lambda/G)]).T)
        r_body_to_ang_mom = R.from_euler('ZXZ', np.array([g, np.arccos(L/G), l]).T)
        r_permute_axes = R.from_matrix(permute_axes)
        r_body_to_lab = r_ang_mom_to_lab * r_body_to_ang_mom * r_permute_axes
        G_body = np.array([0, 0, G])
        G_lab = r_body_to_lab.apply(G_body)
        omegaa = G_lab[0]/A
        omegab = G_lab[1]/B
        omegac = G_lab[2]/C
        theta, phi, psi = r_body_to_lab.inv().as_euler('ZXZ')
        return cls([theta, phi, psi, omegaa, omegab, omegac], A=A, B=B, C=C)

    @property
    def euler(self):
        return np.asarray([self.theta, self.phi, self.psi, self.omegaa, self.omegab, self.omegac]).T
    
    @property
    def quat(self):
        return np.asarray([self.lambda0, self.lambda1, self.lambda2, self.lambda3, self.omegaa, self.omegab, self.omegac]).T
    
    @property
    def omega(self):
        return np.linalg.norm([self.omegaa, self.omegab, self.omegac], axis=0)
    
    @property
    def omega_lab(self):
        r_body_to_lab = R.from_euler('ZXZ', np.array([self.theta, self.phi, self.psi]).T)
        omega_body = np.array([self.omegaa, self.omegab, self.omegac]).T
        omega_lab = r_body_to_lab.apply(omega_body)
        return omega_lab

    @property
    def G_lab(self):
        if self.A is None or self.B is None or self.C is None:
            raise ValueError("Cannot calculate Andoyer coordinates without principal moments of inertia")
        r_body_to_lab = R.from_euler('ZXZ', np.array([self.theta, self.phi, self.psi]).T)
        G_body = np.array([self.A*self.omegaa, self.B*self.omegab, self.C*self.omegac]).T
        G_lab = r_body_to_lab.apply(G_body)
        return G_lab

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

    def andoyer(self, permute_axes=np.eye(3)):
        """
        Calculate the Andoyer canonical coordinates, according to Deprit (1967)
        These make use of an intermediate plane defined by the angular momentum vector
        We must arbitrarily choose the lab z axis and body c axis as the preferred directions
        This can be changed by providing a permutation matrix permute_axes
        
        G is the magnitude of the angular momentum vector
        Lambda is the z-component of the angular momentum vector
        L is the magnitude of the angular momentum vector projected onto the body c axis
        g is the argument of the intersection of the body equatorial plane with the intermediate plane
        lambda_node is the longitude of the ascending node of the angular momentum vector
        l is the argument of the a axis in the equatorial plane
        """
        if self.A is None or self.B is None or self.C is None:
            raise ValueError("Cannot calculate Andoyer coordinates without principal moments of inertia")
        r_body_to_lab = R.from_quat(self.quat[...,[1,2,3,0]])
        G_body = np.array([self.A*self.omegaa, self.B*self.omegab, self.C*self.omegac]).T
        G = np.linalg.norm(G_body, axis=-1)
        G_lab = r_body_to_lab.apply(G_body)
        Lambda = G_lab[...,2]
        Inc = np.arccos(Lambda/G)
        lambda_node = np.arctan2(G_lab[...,0], -G_lab[...,1])
        r_ang_mom_to_lab = R.from_euler('ZX', np.array([lambda_node, Inc]).T)
        assert np.linalg.det(permute_axes) == 1
        r_permute_axes = R.from_matrix(permute_axes)
        r_body_to_ang_mom = r_ang_mom_to_lab.inv() * r_body_to_lab * r_permute_axes.inv()
        g = np.remainder(r_body_to_ang_mom.as_euler('ZXZ')[...,0], 2*np.pi)
        l = np.remainder(r_body_to_ang_mom.as_euler('ZXZ')[...,2], 2*np.pi)
        L_axis = r_permute_axes.inv().apply([0,0,1])
        if L_axis[0]:
            L = self.A*self.omegaa
        elif L_axis[1]:
            L = self.B*self.omegab
        elif L_axis[2]:
            L = self.C*self.omegac
        else:
            raise ValueError("Invalid permutation matrix")
        J = np.arccos(L/G)
        assert np.allclose(np.remainder(r_body_to_ang_mom.as_euler('ZXZ')[...,1], 2*np.pi), J)
        return np.asarray([G, Lambda, L, g, lambda_node, l])

class RotationSimulation():
    def __init__(self, A, B, C, e_forced, e_free=0, e_period=np.inf, n=1.0, M0=0.0, potential=True):
        '''
        A, B, C: principal moments of inertia
        e_forced: forced eccentricity
        e_free: free eccentricity
        e_period: period of free eccentricity oscillation
        n: mean motion
        M0: initial mean anomaly
        potential: whether to use the potential function (True) or not (False)
        '''
        self.A = A
        self.B = B
        self.C = C
        self.e_forced = e_forced
        self.e_free = e_free
        self.e_period = e_period
        self.n = n
        self.M0 = M0
        self.potential = potential

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
        omegaa_dot = (self.B-self.C)/self.A*(omegab*omegac - self.potential*3*beta*gamma/r**3)
        omegab_dot = (self.C-self.A)/self.B*(omegac*omegaa - self.potential*3*gamma*alpha/r**3)
        omegac_dot = (self.A-self.B)/self.C*(omegaa*omegab - self.potential*3*alpha*beta/r**3)
        return [theta_dot, phi_dot, psi_dot, omegaa_dot, omegab_dot, omegac_dot]

    def deriv_quat(self, t, y):
        lambda0, lambda1, lambda2, lambda3, omegaa, omegab, omegac = y
        r, f = self.get_rf(t)
        lambda0_dot = 0.5*(-lambda1*omegaa - lambda2*omegab - lambda3*omegac)
        lambda1_dot = 0.5*(lambda0*omegaa - lambda3*omegab + lambda2*omegac)
        lambda2_dot = 0.5*(lambda3*omegaa + lambda0*omegab - lambda1*omegac)
        lambda3_dot = 0.5*(-lambda2*omegaa + lambda1*omegab + lambda0*omegac)

        alpha, beta, gamma = direction_cosines_quat(f, lambda0, lambda1, lambda2, lambda3)
        omegaa_dot = (self.B-self.C)/self.A*(omegab*omegac - self.potential*3*beta*gamma/r**3)
        omegab_dot = (self.C-self.A)/self.B*(omegac*omegaa - self.potential*3*gamma*alpha/r**3)
        omegac_dot = (self.A-self.B)/self.C*(omegaa*omegab - self.potential*3*alpha*beta/r**3)

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
