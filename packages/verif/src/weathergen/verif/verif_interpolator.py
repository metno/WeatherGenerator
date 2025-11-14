
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


def normalise(x):
    return x[:]/np.sum(x[:])


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

        self.compute_weights()


    def compute_weights(self):
        """
        Compute the weights of the three nearest grid points
        by computing the barycentric coordinates,
        assuming that the observations are close enough to the plane through the grid points.
        """

        self.weights = np.ndarray((self.obs_xyz.shape[0],3))

        print()
        print(self.weights.shape)
        print(type(self.weights))

        eps = 0.01

        for i, (obs, indix) in enumerate(zip(self.obs_xyz, self.indices)):

            AB = self.grid_xyz[indix[1]] - self.grid_xyz[indix[0]]
            AC = self.grid_xyz[indix[2]] - self.grid_xyz[indix[0]]
            BC = self.grid_xyz[indix[2]] - self.grid_xyz[indix[1]]
            AP = obs                     - self.grid_xyz[indix[0]]
            BP = obs                     - self.grid_xyz[indix[1]]

            area_tot          = np.linalg.norm(np.cross(AB, AC))
            self.weights[i,0] = np.linalg.norm(np.cross(BC, BP))
            self.weights[i,1] = np.linalg.norm(np.cross(AC, AP))
            self.weights[i,2] = np.linalg.norm(np.cross(AB, AP))

            if (1 - area_tot/np.sum(self.weights[i,:]) < eps):
                continue

            indix[2] = indix[3]

            AC = self.grid_xyz[indix[2]] - self.grid_xyz[indix[0]]
            BC = self.grid_xyz[indix[2]] - self.grid_xyz[indix[1]]

            area_tot          = np.linalg.norm(np.cross(AB, AC))
            self.weights[i,0] = np.linalg.norm(np.cross(BC, BP))
            self.weights[i,1] = np.linalg.norm(np.cross(AC, AP))

            if (1 - area_tot/np.sum(self.weights[i,:]) < eps):
                continue

            indix[2] = indix[4]

            AC = self.grid_xyz[indix[2]] - self.grid_xyz[indix[0]]
            BC = self.grid_xyz[indix[2]] - self.grid_xyz[indix[1]]

            self.weights[i,0] = np.linalg.norm(np.cross(BC, BP))
            self.weights[i,1] = np.linalg.norm(np.cross(AC, AP))

        self.weights = self.weights/self.weights.sum(axis=1)[:, np.newaxis]


    def interpolate(self, values):
        """
        Interpolate values to points
        """

        wvalues = np.ndarray((self.obs_points.shape[0]))

        print('weight 0: ', self.weights[0,0])
        print('weight 1: ', self.weights[0,1])
        print('weight 2: ', self.weights[0,2])

        print('indice 0: ', self.indices[0,0])
        print('indice 1: ', self.indices[0,1])
        print('indice 2: ', self.indices[0,2])

        print('value 0:  ', values[self.indices[0,0]])
        print('value 1:  ', values[self.indices[0,1]])
        print('value 2:  ', values[self.indices[0,2]])

        wvalues[:] = self.weights[:,0]*values[self.indices[:,0]]\
                   + self.weights[:,1]*values[self.indices[:,1]]\
                   + self.weights[:,2]*values[self.indices[:,2]]

        return wvalues

