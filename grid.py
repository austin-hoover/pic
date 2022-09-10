import numpy as np
from scipy.interpolate import RegularGridInterpolator

class Grid:
    """Class for 2D grid.

    Attributes
    ----------
    xmin, ymin, xmax, ymax : float
        Minimum and maximum coordinates.
    Nx, Ny : int
        Number of grid points.
    dx, dy : int
        Spacing between grid points.
    x, y : ndarray, shape (Nx,) or (Ny,)
        Positions of each grid point.
    cell_area : float
        Area of each cell.
    """
    def __init__(self, xlim=(-1, 1), ylim=(-1, 1), size=(64, 64)):
        self.xlim, self.ylim = xlim, ylim
        (self.xmin, self.xmax), (self.ymin, self.ymax) = xlim, ylim
        self.size = size
        self.Nx, self.Ny = size
        self.dx = (self.xmax - self.xmin) / (self.Nx - 1)
        self.dy = (self.ymax - self.ymin) / (self.Ny - 1)
        self.cell_area = self.dx * self.dy
        self.x = np.linspace(self.xmin, self.xmax, self.Nx)
        self.y = np.linspace(self.ymin, self.ymax, self.Ny)
        
    def set_lims(self, xlim, ylim):
        """Set the min and max grid coordinates."""
        self.__init__(xlim, ylim, self.size)
        
    def zeros(self):
        """Create array of zeros with same size as the grid."""
        return np.zeros((self.size))

    def distribute(self, positions):
        """Distribute points on the grid using the cloud-in-cell (CIC) method.
        
        Note: Cython is not yet used, so this will be very slow.
        
        Parameters
        ----------
        positions : ndarray, shape (n, 2)
            List of (x, y) positions.
            
        Returns
        -------
        rho : ndarray, shape (Nx, Ny)
            Density at each grid point.
        """
        # Compute area overlapping with 4 nearest neighbors
        ivals = np.floor((positions[:, 0] - self.xmin) / self.dx).astype(int)
        jvals = np.floor((positions[:, 1] - self.ymin) / self.dy).astype(int)
        ivals[ivals > self.Nx - 2] = self.Nx - 2
        jvals[jvals > self.Ny - 2] = self.Ny - 2
        x_i, x_ip1 = self.x[ivals], self.x[ivals + 1]
        y_j, y_jp1 = self.y[jvals], self.y[jvals + 1]
        _A1 = (positions[:, 0] - x_i) * (positions[:, 1] - y_j)
        _A2 = (x_ip1 - positions[:, 0]) * (positions[:, 1] - y_j)
        _A3 = (positions[:, 0] - x_i) * (y_jp1 - positions[:, 1])
        _A4 = (x_ip1 - positions[:, 0]) * (y_jp1 - positions[:, 1])
        # Distribute areas for each point
        rho = self.zeros()
        for i, j, A1, A2, A3, A4 in zip(ivals, jvals, _A1, _A2, _A3, _A4):
            rho[i, j] += A4
            rho[i + 1, j] += A3
            rho[i, j + 1] += A2
            rho[i + 1, j + 1] += A1
        return rho / self.cell_area

    def interpolate(self, grid_vals, positions):
        """Interpolate values from the grid using the CIC method.
        
        Parameters
        ----------
        positions : ndarray, shape (n, 2)
            List of (x, y) positions.
        grid_vals : ndarray, shape (n, 2)
            Scalar value at each coordinate point.
            
        Returns
        -------
        int_vals : ndarray, shape (nparts,)
            Interpolated value at each position.
        """
        int_func = RegularGridInterpolator((self.x, self.y), grid_vals)
        return int_func(positions)

    def gradient(self, grid_vals):
        """Compute gradient using 2nd order centered differencing.
        
        Parameters
        ----------
        grid_vals : ndarray, shape (Nx, Ny)
            Scalar values at each grid point.
        neg : Bool
            If True, return the negative of the gradient.
            
        Returns
        -------
        gradx, grady : ndarray, shape (Nx, Ny)
            The x and y gradient at each grid point.
        """
        return np.gradient(grid_vals, self.dx, self.dy)
