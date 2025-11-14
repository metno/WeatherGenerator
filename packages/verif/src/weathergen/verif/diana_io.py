from pathlib import Path

class diana_io:
    """
    Class to print locations in Diana format
    """

    def __init__(self, path):
        """
        Store the path
        """

        self.path = path


    def write(self, points):
        """
        Print points to Diana file
        """

        with self.path.open('w') as dianio:

            dianio.write('[NAME ZARRCOORDS]\n')
            dianio.write('\n')
            dianio.write('[COLUMNS Lat:r Lon:r]\n')
            dianio.write('\n')
            dianio.write('[DATA]\n')

            for (lat,lon) in points:
                dianio.write("   %3.5f  %3.5f\n"%(lat, lon))



#        dianio.write("   %3.5f  %3.5f\n"%(obs_coords[0,0], obs_coords[0,1]))
#        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[0,0],0], zarr_coords[indices[0,0],1]))
#        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[0,1],0], zarr_coords[indices[0,1],1]))
#        dianio.write("   %3.5f  %3.5f\n"%(zarr_coords[indices[0,2],0], zarr_coords[indices[0,2],1]))
