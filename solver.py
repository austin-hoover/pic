import numpy as np
import pandas as pd
from scipy.fft import fft2, ifft2
from scipy.constants import epsilon_0, pi
from scipy.integrate import solve_ivp

from grid import Grid
from utils import mat2vec


k0 = 0.35 # [m^-1]


class FODO:
    """Class for FODO lattice.
    
    The order is : half-qf, drift, qd, drift, half-qf. Both magnets are
    upright and have the same strength.
    
    Attributes
    ----------
    k0 : float
        Focusing strength of both quadrupoles [m^-1].
    length : float
        Period length [m].
    fill_fac : float
        Fraction of cell filled with quadrupoles.
    """
    def __init__(self, k0=k0, length=5.0, fill_fac = 0.5):
        self.k0, self.length, self.fill_fac = k0, length, fill_fac
        
    def foc_strength(self, s):
        """Return x and y focusing strength at position `s`. We assume the
        lattice repeats forever."""
        kx, ky = 0., 0
        s %= self.length # assume infinite repeating cells
        s /= self.length # fractional position in cell
        delta = 0.25 * self.fill_fac
        if s < delta or s > 1 - delta:
            kx, ky = self.k0, -self.k0
        elif 0.5 - delta <= s < 0.5 + delta:
            kx, ky = -self.k0, +self.k0
        return kx, ky
        
        
class EnvelopeSolver:
    """Class to track the rms beam envelope assuming a uniform density ellipse.
    
    Attributes
    ----------
    positions : ndarray, shape (nsteps + 1,)
        Positions at which to evaluate.
    Sigma0 : ndarray, shape (4, 4):
        Initial covariance matrix.
    sigma0 : ndarray, shape (10,)
        Initial moment vector (upper-triangular elements of `Sigma`. Order is :
        ['x2','xxp','xy','xyp','xp2','yxp','xpyp','y2','yyp','yp2'].
    sigma : ndarray, shape (nsteps + 1, 10)
        Beam moment vector at each position.
    perveance : float
        Dimensionless beam perveance.
    ext_foc : callable
        Function which returns the external focusing strength at position s.
        Call signature is `kx, ky = ext_foc(s)`.
    mm_mrad : bool
        Whether to convert units to mm-mrad.
    """
    def __init__(self, Sigma0, positions, perveance, ext_foc=None,
                 mm_mrad=True, atol=1e-14):
        self.sigma0 = Sigma0[np.triu_indices(4)]
        self.positions, self.perveance = positions, perveance
        self.mm_mrad = mm_mrad
        self.atol = atol
        self.ext_foc = ext_foc
        if self.ext_foc is None:
            self.ext_foc = lambda s: (0.0, 0.0)
        
    def reset(self):
        self.moments = []
        
    def set_perveance(self, perveance):
        self.perveance = perveance
        
    def derivs(self, sigma, s):
        """Return derivative of 10 element moment vector."""
        k0x, k0y = self.ext_foc(s)
        k0xy = 0.
        # Space charge terms
        s11, s12, s13, s14, s22, s23, s24, s33, s34, s44 = sigma
        S0 = np.sqrt(s11*s33 - s13**2)
        Sx, Sy = s11 + S0, s33 + S0
        D = S0 * (Sx + Sy)
        psi_xx, psi_yy, psi_xy = Sy/D, Sx/D, -s13/D
        # Modified focusing strength
        kx = k0x - 0.5 * self.perveance * psi_xx
        ky = k0y - 0.5 * self.perveance * psi_yy
        kxy = k0xy + 0.5 * self.perveance * psi_xy
        # Derivatives
        sigma_p = np.zeros(10)
        sigma_p[0] = 2 * s12
        sigma_p[1] = s22 - kx*s11 + kxy*s13
        sigma_p[2] = s23 + s14
        sigma_p[3] = s24 + kxy*s11 - ky*s13
        sigma_p[4] = -2*kx*s12 + 2*kxy*s23
        sigma_p[5] = s24 - kx*s13 + kxy*s33
        sigma_p[6] = -kx*s14 + kxy*(s34+s12) - ky*s23
        sigma_p[7] = 2 * s34
        sigma_p[8] = s44 + kxy*s13 - ky*s33
        sigma_p[9] = 2*kxy*s14 - 2*ky*s34
        return sigma_p
                
    def integrate(self):
        """Integrate the envelope equations."""
        self.moments = odeint(self.derivs, self.sigma0, self.positions, atol=self.atol)
        if self.mm_mrad:
            self.moments *= 1e6
            


class PoissonSolver:
    """Class to solve Poisson's equation on a 2D grid.

    Attributes
    ----------
    rho, phi, G : ndarray, shape (2*Nx, 2*Ny)
        Charge density (rho), potential (phi), and Green's function (G) at each
        grid point on a doubled grid. Only one quadrant (i < Nx, j < Ny)
        corresponds to to the real potential.
    """
    def __init__(self, grid, sign=-1.):
        self.grid = grid
        new_shape = (2 * self.grid.Nx, 2 * self.grid.Ny)
        self.rho, self.G = np.zeros(new_shape), np.zeros(new_shape)
        self.phi = np.zeros(new_shape)
        
    def set_grid(self, grid):
        self.__init__(grid)
        
    def compute_greens_function(self, line_charge_density):
        """Compute Green's function on doubled grid."""
        Nx, Ny = self.grid.Nx, self.grid.Ny
        Y, X = np.meshgrid(self.grid.x - self.grid.xmin,
                           self.grid.y - self.grid.ymin)
        self.G[:Nx, :Ny] = np.log(X**2 + Y**2, out=np.zeros_like(X),
                                  where=(X + Y > 0))
        self.G *= 0.5 * line_charge_density / (2 * pi * epsilon_0)
        self.G[Nx:, :] = np.flip(self.G[:Nx, :], axis=0)
        self.G[:, Ny:] = np.flip(self.G[:, :Ny], axis=1)
                
    def get_potential(self, rho, line_charge_density):
        """Compute the electric potential on the grid.
        
        Parameters
        ----------
        rho : ndarray, shape (Nx, Ny)
            Charge density at each grid point.
        line_charge_density : float
            Longitudinal charge density.
        
        Returns
        -------
        phi : ndarray, shape (Nx, Ny)
            Electric potential at each grid point.
        """
        Nx, Ny = self.grid.Nx, self.grid.Ny
        self.rho[:Nx, :Ny] = rho
        self.compute_greens_function(line_charge_density)
        self.phi = ifft2(fft2(self.G) * fft2(self.rho)).real
        return self.phi[:Nx, :Ny]
