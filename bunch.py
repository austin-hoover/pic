import numpy as np
import numpy.linalg as la
from scipy.stats import truncnorm
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from scipy.constants import epsilon_0, elementary_charge, speed_of_light, pi
proton_mass = 0.938272029 # [GeV/c^2]
classical_proton_radius = 1.53469e-18 # [m]
k0 = 0.35 # [m^-1]


def rotation_matrix(phi):
    """2D rotation matrix (cw)."""
    C, S = np.cos(phi), np.sin(phi)
    return np.array([[C, S], [-S, C]])


def apply(M, X):
    """Apply matrix M to all rows of X."""
    return np.apply_along_axis(lambda x: np.matmul(M, x), 1, X)


def norm_rows(X):
    """Normalize all rows of X to unit length."""
    return np.apply_along_axis(lambda x: x/la.norm(x), 1, X)
    
    
def rand_rows(X, k):
    """Return k random rows of X."""
    idx = np.random.choice(X.shape[0], k, replace=False)
    return X[idx, :]


def Vmat2D(alpha, beta):
    """Return symplectic normalization matrix for 2D phase space."""
    return np.sqrt(1/beta)* np.array([[beta, 0], [alpha, 1]])
    
    
def get_perveance(line_density, beta, gamma):
    """Return the dimensionless space charge perveance."""
    return 2 * classical_proton_radius * line_density / (beta**2 * gamma**3)
    
    
def get_sc_factor(charge, mass, beta, gamma):
    """Return factor defined by x'' = factor * (x electric field component).
    
    Units of charge are Coulombs, and units of mass are GeV/c^2.
    """
    mass_kg = mass * 1.782662e-27
    velocity = beta * speed_of_light
    return charge / (mass_kg * velocity**2 * gamma**3)


class DistGenerator:
    """Class to generate particle distributions in 4D phase space.
    
    The four dimensions are the transverse positions and slopes {x, x', y, y'}.
    Note that 'normalized space', is referring to normalization in the 2D
    sense, in which the x-x' and y-y' ellipses are upright, as opposed the 4D
    sense, in which the whole covariance matrix is diagonal. In other words,
    only the regular Twiss parameters are used.
    
    Attributes
    ----------
    ex, ey : float
        Rms emittances: eu = sqrt(<u^2><u'^2> - <uu'>^2)
    ax, ay, bx, by : float
        Alpha and beta functions: au = <uu'> / eu, bu = <u^2> / eu
    V : ndarray, shape (4, 4)
        Symplectic normalization matrix.
    A : ndarray, shape (4, 4)
        Emittance scaling matrix.
    kinds : list
        List of the available distributions.
    """
    
    def __init__(self, twiss=(0., 0., 10., 10.), eps=(100e-6, 100e-6)):
        self.set_eps(*eps)
        self.set_twiss(*twiss)
        self.kinds = ['kv', 'gauss', 'waterbag', 'danilov']
        self._gen_funcs = {'kv':self._kv,
                           'gauss':self._gauss,
                           'waterbag':self._waterbag,
                           'danilov':self._danilov}
        
    def set_twiss(self, ax, ay, bx, by):
        """Set Twiss parameters"""
        self.ax, self.ay, self.bx, self.by = ax, ay, bx, by
        self.V = np.zeros((4, 4))
        self.V[:2, :2] = Vmat2D(ax, bx)
        self.V[2:, 2:] = Vmat2D(ay, by)
        
    def set_eps(self, ex, ey):
        """Set emittance."""
        self.ex, self.ey = ex, ey
        self.A = np.sqrt(np.diag([ex, ex, ey, ey]))
        
    def get_cov(self, ex, ey):
        Sigma = np.zeros((4, 4))
        gx = (1 + self.ax**2) / self.bx
        gy = (1 + self.ay**2) / self.by
        Sigma[:2, :2] = ex * np.array([[self.bx, -self.ax], [-self.ax, gx]])
        Sigma[2:, 2:] = ey * np.array([[self.by, -self.ay], [-self.ay, gy]])
        return Sigma
    
    def unnormalize(self, X):
        """Transform coordinates out of normalized phase space.
        
        X : ndarray, shape (nparts, 4)
        """
        return apply(np.matmul(self.V, self.A), X)
    
    def normalize(self, X):
        """Transform coordinates into normalized phase space.
        
        X : ndarray, shape (nparts, 4)
        """
        return apply(la.inv(np.matmul(self.V, self.A)), X)
    
    def generate(self, kind='gauss', nparts=1, eps=None, **kwargs):
        """Generate a distribution.
        
        Parameters
        ----------
        kind : {'kv', 'gauss', 'danilov'}
            The kind of distribution to generate.
        **kwargs
            Key word arguments passed to the generating function.
        
        Returns
        -------
        X : ndarray, shape (nparts, 4)
            The corodinate array
        """
        if type(eps) is float:
            eps = [eps, eps]
        if kind == 'kv':
            eps = [4 * e for e in eps]
        if eps is not None:
            self.set_eps(*eps)
        Xn = self._gen_funcs[kind](int(nparts), **kwargs)
        return self.unnormalize(Xn)
    
    def _kv(self, nparts, **kwargs):
        """Generate a KV distribution in normalized space.
        
        Particles uniformly populate the boundary of a 4D sphere. This is
        achieved by normalizing the radii of all particles in 4D Gaussian
        distribution to unit length.
        """
        Xn = np.random.normal(size=(nparts, 4))
        return norm_rows(Xn)
        
    def _gauss(self, nparts, cut=None, **kwargs):
        """Gaussian distribution in normalized space.
        
        cut: float or None
            Cut off the distribution after this many standard devations."""
        if cut:
            Xn = truncnorm.rvs(a=4*[-cut], b=4*[cut], size=(nparts, 4))
        else:
            Xn = np.random.normal(size=(nparts, 4))
        return Xn
    
    def _danilov(self, nparts, phase_diff=90., **kwargs):
        """Danilov distribution in normalized space.
        
        This is defined by the conditions {y' = ax + by, x' = cx + dy} for all
        particles. Note that it is best to use the 4D Twiss parameters instead.
        
        phase_diff : float
            Difference between x and y phases.
        """
        r = np.sqrt(np.random.random(nparts))
        theta = 2 * np.pi * np.random.random(nparts)
        x, y = r * np.cos(theta), r * np.sin(theta)
        xp, yp = -y, x
        Xn = np.vstack([x, xp, y, yp]).T
        P = np.identity(4)
        P[:2, :2] = rotation_matrix(np.radians(phase_diff - 90))
        return apply(P, Xn)
    
    def _waterbag(self, **kwargs):
        """Waterbag distribution in normalized space.
        
        Particles uniformly populate the interior of a 4D sphere. First,
        a KV distribution is generated. Then, particle radii are scaled by
        the factor r^(1/4), where r is a uniform random variable in the range
        [0, 1].
        """
        Xn = norm_rows(np.random.normal(size=(nparts, 4)))
        r = np.random.random(nparts)**(1/4)
        r = r.reshape(nparts, 1)
        return r * Xn
    
    
class Bunch:
    """Container for 2D distribution of particles.

    Attributes
    ----------
    intensity : float
        Number of physical particles in the bunch. Default 1.5e14.
    length : float
        Length of the bunch [m]. Default: 250.
    mass, charge, kin_energy : float
        Mass [GeV/c^2], charge [C], and kinetic energy [GeV] per particle.
    line_density : float
        Longitudinal particle density [1 / m]. Default 1.5e14 / 250.
    line_charge_density : float
        Longitudinal charge density [C / m].
    nparts : float
        Number of macroparticles in the bunch.
    macrosize : float
        Number of physical particles represented by each macroparticle.
    macrocharge : float
        Charge represented by each macroparticle [C].
    perveance : float
        Dimensionless space charge perveance.
    sc_factor : float
        Factor for space charge kicks such that that x'' = factor * Ex.
    X : ndarray, shape (nparts, 4)
        Array of particle coordinates. Columns are [x, x', y, y']. Units are
        meters and radians.
    positions : ndarray, shape (nparts, 2):
        Just the x and y positions (for convenience).
    """
    def __init__(self, intensity=1e14, length=250., mass=0.938, kin_energy=1.,
                 charge=elementary_charge):
        self.intensity, self.length = intensity, length
        self.mass, self.kin_energy, self.charge = mass, kin_energy, charge
        self.gamma = 1 + (self.kin_energy / self.mass)
        self.beta = np.sqrt(1 - (1 / self.gamma)**2)
        self.nparts = 0
        self.line_density = intensity / length
        self.line_charge_density = charge * self.line_density
        self.perveance = get_perveance(self.line_density, self.beta, self.gamma)
        self.sc_factor = get_sc_factor(charge, mass, self.beta, self.gamma)
        self.compute_macrosize()
        self.X, self.positions = None, None
        
    def compute_macrosize(self):
        """Update the macrosize and macrocharge."""
        if self.nparts > 0:
            self.macrosize = self.intensity // self.nparts
        else:
            self.macrosize = 0
        self.macrocharge = self.charge * self.macrosize
                                
    def fill(self, X):
        """Fill with particles."""
        self.X = X if self.X is None else np.vstack([self.X, X])
        self.positions = self.X[:, [0, 2]]
        self.nparts = self.X.shape[0]
        self.compute_macrosize()

    def compute_extremum(self):
        """Get extreme x and y coorinates."""
        self.xmin, self.ymin = np.min(self.positions, axis=0)
        self.xmax, self.ymax = np.max(self.positions, axis=0)
        self.xlim, self.ylim = (self.xmin, self.xmax), (self.ymin, self.ymax)
