
# Core object
from tyssue import Sheet
# Simple 2D geometry
from tyssue import PlanarGeometry as geom
# Visualisation
from tyssue.draw import sheet_view
import matplotlib.pyplot as plt

# Read the HDF5 file of the middle bulge.# Read the HDF5 file of the middle bulge.
from tyssue.io import hdf5
dsets = hdf5.load_datasets('trilayer_bulge_data.hdf5')
sheet = Sheet('reloaded', dsets)
# Plot the geometry
fig, ax = sheet_view(sheet)
# Fix axis limits and aspect
ax.set_xlim(-5, 20)
ax.set_ylim(-5, 7)
ax.set_aspect('equal')
plt.show()


# Add the dynamic specs



# Do a T1 transition on the edge we want, find the edge number first.


# Now run another t= 1 to see relaxation.




print('\n This is the end of this script. (＾• ω •＾) ')
