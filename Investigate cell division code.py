# -*- coding: utf-8 -*-
"""
This script is to investigate the data structure of the Tyssue package.
"""

# =============================================================================
# First we need to surpress the version warnings from Pandas.
import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning) 
# =============================================================================

# Load all required modules.


import pandas as pd
import numpy as np
import os
import json
import matplotlib as matplot
import matplotlib.pylab as plt
import ipyvolume as ipv

from tyssue import Sheet, config #import core object
from tyssue import PlanarGeometry as geom #for simple 2d geometry

# For cell topology/configuration
from tyssue.topology.sheet_topology import type1_transition
from tyssue.topology.base_topology import collapse_edge, remove_face
from tyssue.topology.sheet_topology import split_vert as sheet_split
from tyssue.topology.bulk_topology import split_vert as bulk_split
from tyssue.topology import condition_4i, condition_4ii

## model and solver
from tyssue.dynamics.planar_vertex_model import PlanarModel as pmodel
from tyssue.solvers.quasistatic import QSSolver
from tyssue.generation import extrude
from tyssue.dynamics import model_factory, effectors
from tyssue.topology.sheet_topology import remove_face, cell_division


# 2D plotting
from tyssue.draw import sheet_view, highlight_cells

#I/O
from tyssue.io import hdf5


'''
First we generate a bilayer structure and a sample cell sheet.
Bilayer is generated via the function Sheet.planar_sheet_2d().
'''
# Generate a bilayer structure.
bilayer = Sheet.planar_sheet_2d(identifier = 'basic2D', nx = 30, ny = 4, distx = 2, disty = 2)

bilayer.sanitize(trim_borders=True, order_edges=True)
geom.update_all(bilayer)

fig, ax = sheet_view(bilayer, mode = '2D')
fig.set_size_inches(10,10)

# Generate sample sheet, Investigate the data structure for cell division now. 
solver = QSSolver()
sheet = Sheet.planar_sheet_2d('division', 6, 6, 1, 1)
sheet.sanitize(trim_borders=True, order_edges=True)
geom.update_all(sheet)

#sheet.get_opposite() # Don't know why this line is here, seems unnecessary.

# Set up the model
# First use the default specification for the dyanmics of a sheet vertex model.
# This way, we can assign properties such as line_tension into the edge_df.
nondim_specs = config.dynamics.quasistatic_plane_spec()
dim_model_specs = pmodel.dimensionalize(nondim_specs)
# No differences between two specs.
# We can use either specs.

# udpate the new specs (contain line_tension, etc) into the cell data.
sheet.update_specs(dim_model_specs, reset=True)

# Show number of cells, edges and vertices of the sheet.
print("Number of cells: {}\n"
      "          edges: {}\n"
      "          vertices: {}\n".format(sheet.Nf, sheet.Ne, sheet.Nv))

# ## Minimize energy
res = solver.find_energy_min(sheet, geom, pmodel)

# ## View the result
draw_specs = config.draw.sheet_spec()
draw_specs['vert']['visible'] = False
draw_specs['edge']['head_width'] = 0  # values other than 0 gives error.
fig, ax = sheet_view(sheet, **draw_specs)
fig.set_size_inches(12, 5)

# Now we perform cell division.

# Cause a cell to divide. second parameter is the face index of mother cell.
# The function returns the face index of new cell.
daughter = cell_division(sheet, 1, geom, angle=np.pi)

# calculate energy min state.
res = solver.find_energy_min(sheet, geom, pmodel)
print(res['success'])

fig, ax = sheet_view(sheet, **draw_specs)
fig.set_size_inches(12, 5)










'''
This is the end of the script.
'''
