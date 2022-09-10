import numpy as np
from tqdm import trange

from grid import Grid
from solver import PoissonSolver
from bunch import Bunch
from utils import mat2vec


class History:
    """Class to store bunch data over time.

    Atributes
    ---------
    moments : list
        Second-order bunch moments. Each element is ndarray of shape (10,).
    coords : list
        Bunch coordinate arrays. Each element is ndarray of shape (nparts, 4)
    moment_positions, coord_positions : list
        Positions corresponding to each element of `moments` or `coords`.
    """
    def __init__(self, bunch, samples=None):
        self.X = bunch.X
        self.moments, self.coords = [], []
        self.moment_positions, self.coord_positions = [], []
        if samples is None or samples >= bunch.nparts:
            self.idx = np.arange(bunch.nparts)
        else:
            self.idx = np.random.choice(bunch.nparts, samples, replace=False)
        
    def store_moments(self, s):
        Sigma = np.cov(self.X.T)
        self.moments.append(Sigma[np.triu_indices(4)])
        self.moment_positions.append(s)
        
    def store_coords(self, s):
        self.coords.append(np.copy(self.X[self.idx, :]))
        self.coord_positions.append(s)
        
    def package(self, mm_mrad):
        self.moments = np.array(self.moments)
        self.coords = np.array(self.coords)
        if mm_mrad:
            self.moments *= 1e6
            self.coords *= 1e3
        
        
class Simulation:
    """Class to simulate the evolution of a charged particle bunch.
        
    Attributes
    ----------
    bunch : Bunch:
        The bunch to track.
    length : float
        Total tracking distance [m].
    step_size : float
        Distance between force calculations [m].
    nsteps : float
        Total number of steps = int(length / ds).
    steps_performed : int
        Number of steps performed so far.
    s : float
        Current bunch position.
    positions : ndarray, shape (nsteps + 1,)
        Positions at which coordinates are updated.
    history : History object
        Object storing bunch data at each position.
    meas_every : dict or int
        Keys should be 'moments' and 'coords'. Values correspond to the
        number of simulations steps between storing these quantities. For
        example, `meas_every = {'coords':4, 'moments':2}` will store the
        moments every 4 steps and the moments every other step. If an
        int is provided, this will be applied to both. Defaults to start
        and end of simulation.
    samples : int
        Number of bunch particles to store when measuring phase space
        coordinates. Defaults to the entire coordinate array.
    mm_mrad : bool
        If True, use units of mm-mrad. Otherwise use m-rad.
    ext_foc : callable
        Function returning external kx and ky at the current position
        such that u'' = -ku. Call signature is `kx, ky = ext_foc(s)`.
    """
    def __init__(self, bunch, length, step_size, grid_size, meas_every={},
                 samples=None, mm_mrad=True, ext_foc=None):
        self.bunch = bunch
        self.length, self.ds = length, step_size
        self.nsteps = int(length / step_size)
        self.positions = np.linspace(0, length, self.nsteps + 1)
        self.grid = Grid(size=grid_size)
        self.solver = PoissonSolver(self.grid)
        self.fields = np.zeros((bunch.nparts, 2))
        self.history = History(bunch, samples)
        self.ext_foc = ext_foc
        if type(meas_every) is int:
            meas_every = {'moments': meas_every, 'coords':meas_every}
        meas_every.setdefault('moments', self.nsteps)
        meas_every.setdefault('coords', self.nsteps)
        for key in meas_every.keys():
             if meas_every[key] is None:
                meas_every[key] = self.nsteps
        self.meas_every = (meas_every['moments'], meas_every['coords'])
        self.mm_mrad = mm_mrad
        self.s, self.steps_performed = 0.0, 0
        
    def set_grid(self):
        """Determine grid limits."""
        self.bunch.compute_extremum()
        self.grid.set_lims(self.bunch.xlim, self.bunch.ylim)
        self.solver.set_grid(self.grid)
        
    def compute_electric_field(self):
        """Compute the self-generated electric field."""
        self.set_grid()
        rho = self.grid.distribute(self.bunch.positions)
        rho *= self.bunch.line_charge_density * 4 # unknown origin
        phi = self.solver.get_potential(rho, self.bunch.line_charge_density)
        Ex, Ey = self.grid.gradient(-phi)
        self.fields[:, 0] = self.grid.interpolate(Ex, self.bunch.positions)
        self.fields[:, 1] = self.grid.interpolate(Ey, self.bunch.positions)
                            
    def kick(self, ds):
        """Update particle slopes."""
        # Space charge
        dxp_ds = self.bunch.sc_factor * self.fields[:, 0]
        dyp_ds = self.bunch.sc_factor * self.fields[:, 1]
        # External forces
        if self.ext_foc is not None:
            kx, ky = self.ext_foc(self.s)
            dxp_ds -= kx * self.bunch.X[:, 0]
            dyp_ds -= ky * self.bunch.X[:, 2]
        self.bunch.X[:, 1] += dxp_ds * ds
        self.bunch.X[:, 3] += dyp_ds * ds
        
    def push(self, ds):
        """Update particle positions."""
        self.bunch.X[:, 0] += self.bunch.X[:, 1] * ds
        self.bunch.X[:, 2] += self.bunch.X[:, 3] * ds
        
    def store(self):
        """Store bunch coordinates or statistics."""
        store_moments = self.steps_performed % self.meas_every[0] == 0
        store_coords = self.steps_performed % self.meas_every[1] == 0
        if not (store_moments or store_coords):
            return
        Xp = np.copy(self.bunch.X[:, [1, 3]])
        self.kick(+0.5 * self.ds) # sync positions/slopes
        if store_moments:
            self.history.store_moments(self.s)
        if store_coords:
            self.history.store_coords(self.s)
        self.bunch.X[:, [1, 3]] = Xp
        
    def run(self):
        """Run the simulation."""
        self.compute_electric_field()
        self.store()
        self.kick(-0.5 * self.ds) # desync positions/slopes
        for i in trange(self.nsteps):
            self.compute_electric_field()
            self.kick(self.ds)
            self.push(self.ds)
            self.s += self.ds
            self.steps_performed += 1
            self.store()
        self.history.package(self.mm_mrad)
