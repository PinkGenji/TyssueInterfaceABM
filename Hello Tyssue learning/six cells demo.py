# -*- coding: utf-8 -*-
"""
This script uses a six cell system as a demo and for me to work around.
"""

# =============================================================================
# First we need to surpress the version warnings from Pandas.
import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning) 
# =============================================================================

# Load all required modules.

import numpy as np
import pandas as pd

import os
import json
import matplotlib as matplot
import matplotlib.pylab as plt
import ipyvolume as ipv

from tyssue import Sheet, config #import core object
from tyssue import PlanarGeometry as geom #for simple 2d geometry

# For cell topology/configuration
from tyssue.topology.sheet_topology import type1_transition
from tyssue.topology.base_topology import collapse_edge, remove_face, add_vert
from tyssue.topology.sheet_topology import split_vert as sheet_split
from tyssue.topology.bulk_topology import split_vert as bulk_split
from tyssue.topology import condition_4i, condition_4ii

## model and solver
from tyssue.dynamics.planar_vertex_model import PlanarModel as smodel
from tyssue.solvers.quasistatic import QSSolver
from tyssue.generation import extrude
from tyssue.dynamics import model_factory, effectors
from tyssue.topology.sheet_topology import remove_face, cell_division, face_division

# 2D plotting
from tyssue.draw import sheet_view, highlight_cells

# import my own functions
from my_headers import delete_face, xprod_2d, put_vert

""" start the project """
# Generate the cell sheet as three cells.
num_x = 3
num_y = 2
sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = num_x, ny = num_y, distx = 1, disty = 1)
geom.update_all(sheet)

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Plot the figure to see the index.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
    
delete_face(sheet, 4)
delete_face(sheet, 3)
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise

# Plot figures to check.
# Draw the cell mesh with face labelling and edge arrows.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)

# Draw with vertex labelling.
fig, ax= sheet_view(sheet, edge = {'head_width':0.1})
for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)

# Draw with edge labelling.
fig, ax= sheet_view(sheet)
for edge, data in sheet.edge_df.iterrows():
    ax.text((data.sx+data.tx)/2, (data.sy+data.ty)/2, edge)
    
    
""" Add mechanical properties for energy minimization """
# We are using the specs from 2007 Farhadifar model.
specs = {
    'edge': {
        'is_active': 1,
        'line_tension': 0.12,
        'ux': 0.0,
        'uy': 0.0,
        'uz': 0.0
    },
   'face': {
       'area_elasticity': 1.0,
       'contractility': 0.04,
       'is_alive': 1,
       'prefered_area': 1.0},
   'settings': {
       'grad_norm_factor': 1.0,
       'nrj_norm_factor': 1.0
   },
   'vert': {
       'is_active': 1
   }
}
sheet.update_specs(specs, reset = True)
geom.update_all(sheet)


print(sheet.edge_df)

# energy minimisation.
solver = QSSolver()
res = solver.find_energy_min(sheet, geom, smodel)

fig, ax = sheet_view(sheet)   # Draw cell mesh.
for vert, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)

""" Assign the cell type to face_df and edge_df """
num_ct = num_x
# First we assign cell type in face data frame.
sheet.face_df.loc[0:num_ct-1, "cell_type"] = 'CT'
sheet.face_df.loc[num_ct:, "cell_type"] = 'ST'

# Then update the edge data frame via 'face' column.

sheet.edge_df['cell_type'] = 'to be set'

for i in list(range(len(sheet.edge_df))):
    sheet.edge_df.loc[i, 'cell_type'] = sheet.face_df.loc[sheet.edge_df.loc[i,'face'],'cell_type']


""" Evolve the cell division and see the behaviour. """
# =============================================================================
# The logic of cell proliferation is:
# Over "x" amount of time, we have a division rate "y" for CT cells. (Division)
# The range of growth time varies within a range or mean +- SD. (Growth)
# =============================================================================


division_rate = 0.01



















"""
This is the end of the script.
"""
