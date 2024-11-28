# -*- coding: utf-8 -*-
"""
This script is the fundamental for T3 transitions
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
from tyssue.draw.plt_draw import plot_forces
from tyssue.config.draw import sheet_spec
# import my own functions
from my_headers import *

# Generate the cell sheet as three cells.
num_x = 4
num_y = 4

sheet = Sheet.planar_sheet_2d('face', nx = num_x, ny=num_y, distx=2, disty=2)

geom.update_all(sheet)

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

for i in list(range(num_x, num_y*(num_x+1), 2*(num_x+1) )):
    delete_face(sheet, i)
    delete_face(sheet, i+1)
sheet.get_extra_indices()
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise

# Visualize the sheet.
sheet_view(sheet)

fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)

sheet.vert_df.loc[26,'x'] -=0.5

boundary_vert = set()
boundary_edge = set()
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'opposite'] == -1:
        boundary_vert.add(sheet.edge_df.loc[i,'srce'])
        boundary_vert.add(sheet.edge_df.loc[i,'trgt'])
        boundary_edge.add(i)
    

for e in boundary_edge:
    # First detect the distance
    # Extract source and target vertex IDs
    srce_id, trgt_id = sheet.edge_df.loc[e, ['srce', 'trgt']]
    # Extract source and target positions as numpy arrays
    endpoint1 = sheet.vert_df.loc[srce_id, ['x', 'y']].values
    endpoint2 = sheet.vert_df.loc[trgt_id, ['x', 'y']].values
    for v in boundary_vert:
        if v != srce_id and v!= trgt_id:
            vertex = sheet.vert_df.loc[v,['x','y']].values
            dist, nearest = pnt2line(vertex, endpoint1 , endpoint2)
            # Check the distance from the vertex to the edge.
            if dist < 1.1:
                # Check if need to extend the line avoid collision.
                towards_end1 = distance(endpoint1, nearest)
                towards_end2 = distance(endpoint2, nearest)
                for i in (endpoint1, endpoint2):
                    if distance(i, nearest) < 0.5:
                        print(f'vert {i} and vert {v} are colliding ')
                        a_hat = unit(vector(endpoint1, endpoint2))
                        
                
                print(f'edge {e} and vert {v} are too close. With nearest point at {nearest}.')

geom.update_all(sheet)
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)


""" This is the end of the script """
