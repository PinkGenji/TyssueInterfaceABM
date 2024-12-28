#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is a unit test for T3 function.
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
from tyssue.draw.plt_draw import plot_forces

# import my own functions
from my_headers import *

from T3_function import *

""" Start programming. """
# Generate the cell sheet as three cells.
num_x = 5
num_y = 4
sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = num_x, ny = num_y, distx = 1, disty = 1)
geom.update_all(sheet)

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Plot the figure to see the index.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
    
delete_face(sheet, 5)
delete_face(sheet, 6)
delete_face(sheet, 17)
delete_face(sheet,18)

sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise
geom.update_all(sheet)
sheet.get_extra_indices()   # extra_indices are not included in update_all.
# Plot figures to check.
# Draw the cell mesh with face labelling and edge arrows.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)
    


""" Make adjustments to the geometry for unit test.  """

# Setup d_min and d_sep values.
d_min = sheet.edge_df.loc[:,'length'].min()/10
d_sep = d_min*1.5
print(f'd_min is set: {d_min}, d_sep is set: {d_sep}')

# Case 1 from Fletcher 2013, changing the position of vertex 3.
sheet.vert_df.loc[3,'x'] = sheet.vert_df.loc[29,'x'] - d_min*0.5
sheet.vert_df.loc[3,'y'] = sheet.vert_df.loc[29,'y'] - d_min*0.5


geom.update_all(sheet)
sheet.get_extra_indices()  
# Plot figures to check.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)



vert1 = sheet.vert_df.loc[3,['x','y']].to_numpy(dtype = float) 
vert2 = sheet.vert_df.loc[29,['x','y']].to_numpy(dtype = float)
d = np.linalg.norm(vert2-vert1)
print(f'The np.lingal.norm computes the distance as: {d}.')

# We know edge 3 is the colliding edge and vertex 3 is the incoming vertex.
# Check if the dist_computer function is working as expected.
sheet.edge_df.loc[3,['srce','trgt']]
distance, nearest = dist_computer(sheet, 3, 3, d_sep)
print(f'The distance computed by dist_computer is: {distance}.')







""" This is the end of the script. """
