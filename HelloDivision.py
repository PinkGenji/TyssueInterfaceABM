# -*- coding: utf-8 -*-
"""
This is for first drawing of proliferation and make sure I have parameter control of different layers.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Supress pandas warning

import matplotlib as matplot

from tyssue import Sheet, config #import core object
from tyssue import PlanarGeometry as geom #for simple 2d geometry
from tyssue.draw import sheet_view #for sheet view

import pandas as pd
import numpy as np

# What we're here for
from tyssue.topology.sheet_topology import type1_transition
from tyssue.topology.base_topology import collapse_edge, remove_face
from tyssue.topology.sheet_topology import split_vert as sheet_split
from tyssue.topology.bulk_topology import split_vert as bulk_split
from tyssue.topology import condition_4i, condition_4ii

## model and solver
from tyssue.dynamics.sheet_vertex_model import SheetModel as model
from tyssue.solvers.quasistatic import QSSolver
from tyssue.generation import extrude
from tyssue.dynamics import model_factory, effectors
from tyssue.topology.sheet_topology import remove_face, cell_division


# 2D plotting
from tyssue.draw import sheet_view, highlight_cells

#I/O
from tyssue.io import hdf5
#plt.style.use('bmh')

import logging






'''
The following code draw a bilayer structure, with each layer only have a single
layer of cells. With more cells than the preivous bilayer.

'''
#start with specifying the properties of the sheet:
#the sheet is named 'basic2D', cell number on x-axis =6, y-axis=7 and distance between 2 cells along x and y are both 1
bilayer = Sheet.planar_sheet_2d(identifier = 'basic2D', nx = 30, ny = 4, distx = 2, disty = 2)
geom.update_all(bilayer) #generate the sheet

# =============================================================================
# #sheet_view() function displays the created object in a matplotlib figure
# fig,ax = sheet_view(sheet) 
# fig.set_size_inches(10,10)
# =============================================================================

bilayer.sanitize(trim_borders=True, order_edges=True)
geom.update_all(bilayer)
# We pass an option to display the edge directions:
fig, ax = sheet_view(bilayer, mode = '2D')
fig.set_size_inches(10,10)


'''
Analyse the datastrcture
'''
solver = QSSolver()
sheet = Sheet.planar_sheet_2d('division', 6, 6, 1, 1)
sheet.sanitize(trim_borders=True, order_edges=True)
geom.update_all(sheet)

fig, ax = sheet_view(sheet, mode = '2D')   #shows the plot

sheet.get_opposite()

help(sheet.get_opposite)

# ## Set up the model
nondim_specs = config.dynamics.quasistatic_plane_spec()
dim_model_specs = model.dimensionalize(nondim_specs)
sheet.update_specs(dim_model_specs, reset=True)
fig, ax = sheet_view(sheet, mode = '2D')

print("Number of cells: {}\n"
      "          edges: {}\n"
      "          vertices: {}\n".format(sheet.Nf, sheet.Ne, sheet.Nv))

# ## Minimize energy
res = solver.find_energy_min(sheet, geom, model)
help(geom)
# ## View the result
draw_specs = config.draw.sheet_spec()
draw_specs['vert']['visible'] = False
draw_specs['edge']['head_width'] = 0   # value other than 0 gives error.
fig, ax = sheet_view(sheet, **draw_specs)
fig.set_size_inches(12, 5)

draw_specs['edge']['head_width'] 

# Generate the daughter cell.

daughter = cell_division(sheet, 7, geom, angle=np.pi/2)

res = solver.find_energy_min(sheet, geom, model)
print(res['success'])

fig, ax = sheet_view(sheet, **draw_specs)
fig.set_size_inches(12, 5)



'''
This is the end of the script.
'''