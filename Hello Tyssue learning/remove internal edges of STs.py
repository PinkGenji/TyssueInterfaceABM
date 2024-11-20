# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:34:56 2024

@author: lyu195
"""

# -*- coding: utf-8 -*-
"""
This script draws a cell evolution of 10 cells with lateral split.
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
from my_headers import *

""" Start programming. """
# Generate the cell sheet as three cells.
num_x = 10
num_y = 2
sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = num_x, ny = num_y, distx = 1, disty = 1)
geom.update_all(sheet)

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Plot the figure to see the index.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
    
delete_face(sheet, num_x)
delete_face(sheet, num_x+1)
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise

# Plot figures to check.
fig, ax = sheet_view(sheet)

num_ct = num_x
# First we assign cell type in face data frame.
sheet.face_df.loc[0:num_ct-1, "cell_type"] = 'CT'
sheet.face_df.loc[num_ct:, "cell_type"] = 'ST'

# Then update the edge data frame via 'face' column.

sheet.edge_df['cell_type'] = 'to be set'

for i in list(range(len(sheet.edge_df))):
    sheet.edge_df.loc[i, 'cell_type'] = sheet.face_df.loc[sheet.edge_df.loc[i,'face'],'cell_type']

sheet.get_opposite()

# First, filter rows where 'cell_type' is 'ST' and 'opposite' is not -1
rows_to_drop = []
for i in range(len(sheet.edge_df)):
    if (sheet.edge_df['cell_type'].iloc[i] == 'ST') and (sheet.edge_df['opposite'].iloc[i] != -1):
        rows_to_drop.append(i)

# Drop all selected rows at once
sheet.edge_df.drop(rows_to_drop, inplace=True)

geom.update_all(sheet)
sheet_view(sheet)

# Plot figures to check
fig, ax = sheet_view(sheet)

# Remove the box and axis lines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Hide the x and y axis ticks and labels
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

plt.show()


fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)
    
edge = sheet.edge_df[(sheet.edge_df['srce'] == 12)].index
sheet.edge_df.loc[edge, 'srce'] = 13
geom.update_all(sheet)
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)

edge2 = sheet.edge_df[(sheet.edge_df['trgt'] == 12)].index
sheet.edge_df.loc[edge2, 'trgt'] = 30
geom.update_all(sheet)
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)

T3_detection(sheet, 2, 0.8)

sheet.get_extra_indices()
sheet.sgle_edges
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'opposite'] == -1:
        boundary

T3_detection(sheet, 2, 0.3)






"""This is the end of the script"""
