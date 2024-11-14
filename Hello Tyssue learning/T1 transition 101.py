# -*- coding: utf-8 -*-
"""
This code file contains the primary design of my own T1 transition
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
from tyssue.topology.base_topology import collapse_edge, remove_face, add_vert, merge_vertices
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
from my_headers import delete_face, xprod_2d, put_vert, lateral_split, divisibility_check

"""For the interior edge case."""
# Generate the cell sheet as three cells.
num_x = 4
num_y = 4

sheet = Sheet.planar_sheet_2d('face', nx = num_x, ny=num_y, distx=2, disty=2)

geom.update_all(sheet)

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Plot the figure to see the index.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
    
for i in list(range(num_x, num_y*(num_x+1), 2*(num_x+1) )):
    delete_face(sheet, i)
    delete_face(sheet, i+1)
sheet.get_extra_indices()
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise

# Visualize the sheet.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)
    

for index, series in sheet.edge_df.iterrows():
    if series['srce'] == 14 and series['trgt'] == 45:
        print(f'The index is: {index}')

sheet.vert_df.loc[45,'y'] +=0.5
geom.update_all(sheet)
sheet_view(sheet)
print('The length of the shortest edge is: '+ str(sheet.edge_df.loc[:,'length'].min()))

# Store the single edges as a deparate df
single_edge_index = sheet.edge_df.index.intersection(sheet.sgle_edges).tolist()
for index in sheet.edge_df.index :
    if sheet.edge_df.loc[index,'length'] < 1.05:
        print(f'Edge {index} is shorter than threshold')
        type1_transition(sheet, index, remove_tri_faces=False, multiplier=1.5)
    else:
        continue
geom.update_all(sheet)

sheet_view(sheet)


"""For the boundary edge case, both vertices have rank 2."""
# Generate the cell sheet as three cells.
num_x = 4
num_y = 4

sheet = Sheet.planar_sheet_2d('face', nx = num_x, ny=num_y, distx=2, disty=2)

geom.update_all(sheet)

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Plot the figure to see the index.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
    
for i in list(range(num_x, num_y*(num_x+1), 2*(num_x+1) )):
    delete_face(sheet, i)
    delete_face(sheet, i+1)
sheet.get_extra_indices()
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise

# Visualize the sheet.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)
    
sheet_view(sheet)

merge_vertices(sheet, 20, 22)
geom.update_all(sheet)
sheet_view(sheet)

new_boundary_coord = sheet.vert_df.loc[20,['x','y']]
new_boundary_coord[0]+=0.5



for i in list(range(len(sheet.edge_df))):
    if sheet.edge_df.loc[i,'opposite'] == -1 and sheet.edge_df.loc[i,'trgt'] == 20:
            print(f'The edge {i} is of interest.')
    elif sheet.edge_df.loc[i,'opposite'] == -1 and sheet.edge_df.loc[i,'srce'] == 20:
            print(f'The edge {i} is of interest.')
    else:
        continue


put_vert(sheet, 90, new_boundary_coord)
put_vert(sheet, 45, new_boundary_coord)
geom.update_all(sheet)
sheet_view(sheet)



""" For the boundary edge case, but one vertex rank = 1, second vertex rank = 2  """












""" This is the end of the file. """
