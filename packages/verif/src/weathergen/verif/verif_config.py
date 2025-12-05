
import yaml

__all__ = ['Variable']


class Variable:
    """
    Object representing a variable
    """

    def __init__(self, **kwargs):
        self.name       = kwargs.get("name")     
        self.attributes = kwargs.get("attributes")
        self.zarr_name  = kwargs.get("zarr_name")
        self.zarr_units = kwargs.get("zarr_units")
        self.obs_name   = kwargs.get("obs_name")
        self.obs_units  = kwargs.get("obs_units")

    def __repr__(self):
        return self.name


class Variables:
    """
    Utility class to read a verif configuration from .yaml file
    """

    def __init__(self, filename):
        print("Reading configuration from file: %s" % filename)

        with open(filename, 'r') as stream:
            self.schema = yaml.safe_load(stream)

    def __iter__(self):
        return self.variables.__iter__()

    @property
    def variables(self):
        """
        Get a list of variables from the locally stored configuration
        """
        return [Variable(**var) for var in self.schema]

