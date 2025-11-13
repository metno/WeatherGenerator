
import numpy as np
from scipy.spatial import KDTree, Delaunay

def convert_coordinates(coords):
    """
    Convert lat-lon coordinates to cartesian coordinates in a unit box
    """

    xyz_coords = np.ndarray((coords.shape[0],3))

    xyz_coords[:,0] = np.cos(np.pi*coords[:,0]/180.0)*np.cos(np.pi*coords[:,1]/180.0)
    xyz_coords[:,1] = np.cos(np.pi*coords[:,0]/180.0)*np.sin(np.pi*coords[:,1]/180.0)
    xyz_coords[:,2] = np.sin(np.pi*coords[:,0]/180.0)

    return xyz_coords


class verif_interpolator:
    """
    Interpolator class that's either a wrapper for scipys LinearNDInterpolator 
    or uses the handmade approximate 2D linear interpolator
    """

    def __init__(self, grid_points, obs_points):
        """
        Initialise the class and store gridpoints
        """

        self.grid_points = grid_points
        self.obs_points = obs_points


class verif_2D_interpolator(verif_interpolator):
    """
    Class that does approximate 2D interpolation
    """

    def prepare(self):
        """
        Do the setup required for interpolation
        """

        self.grid_xyz = convert_coordinates(self.grid_points)
        self.obs_xyz = convert_coordinates(self.obs_points)

        tree = KDTree(self.grid_xyz)
        _, self.indices = tree.query(self.obs_xyz, k = 5)

    def compute_weights():
        """
        Compute the weights of the three nearest grid points
        by computing the barycentric coordinates,
        assuming that the observations are close enough to the plane through the grid points.
        """

        self.weights = np.ndarray((self.obs_xyz.shape[0],3))
        eps = 0.01
