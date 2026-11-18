"""
This script takes the HDF5 file of a trophoblast bilayer system, then plots the outer surface of the STB layer.

"""
# Load all required modules.
import random
import numpy as np
import re
import matplotlib.pyplot as plt
from tyssue import Sheet, PlanarGeometry
from tyssue.topology.sheet_topology import remove_face
from tyssue import PlanarGeometry as geom #for simple 2d geometry
from tyssue.dynamics import effectors, model_factory
from tyssue.io import hdf5 # For saving the datasets
from tyssue.solvers.viscous import EulerSolver
from tyssue.behaviors import EventManager

# 2D plotting
from tyssue.draw import sheet_view
from tyssue.io import hdf5

# Starts reloading the hdf5 files, after each reloading, we check the reloading by genereating a plot.
dsets = hdf5.load_datasets('uniform_alpha.hdf5')
uniform_sheet = Sheet('reloaded', dsets)
fig,ax = sheet_view(uniform_sheet)
ax.set_title('plot of uniform alpha simulation')
plt.show()



dsets2 = hdf5.load_datasets('80percent_alpha.hdf5')
reduced_sheet = Sheet('reloaded', dsets2)
fig, ax = sheet_view(reduced_sheet)
ax.set_title('plot of reduced alpha simulation')
plt.show()

dsets3 = hdf5.load_datasets('50percent_alpha.hdf5')
half_sheet = Sheet('reloaded', dsets3)
fig, ax = sheet_view(half_sheet)
ax.set_title('plot of half alpha simulation')
plt.show()





print("\n This is the end of the script. :) \n")
