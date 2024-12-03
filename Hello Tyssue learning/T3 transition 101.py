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


# =============================================================================
# The following code is how you move a single vertex position.
# move vert 27 lower.

# sheet.vert_df.loc[27, ['x','y']] = [9,2]
# geom.update_all(sheet)
# =============================================================================

# =============================================================================
# The following code is when you create a new vertex, then reconnect an edge
# to the newly created vertex.

# new_id = put_vert(sheet, 20, [9,2])[0]
# sheet.edge_df.loc[43,'srce'] = new_id #change the trgt vert_i to new_id
# geom.update_all(sheet)
# =============================================================================

# =============================================================================
# The following code is how to compute the rank.
# The logic is total rank = (2*double edges)/2 + single edges


# edge_associated = sheet.edge_df[(sheet.edge_df['srce'] == 27) | (sheet.edge_df['trgt'] == 27)]
# rank = (len(edge_associated) - len(edge_associated[edge_associated['opposite'] == -1]))/2 + len(edge_associated[edge_associated['opposite'] == -1])
# print(rank)
# =============================================================================
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



"""Check which vertex will collide with which edge"""    

for e in boundary_edge:
    # Extract source and target vertex IDs
    srce_id, trgt_id = sheet.edge_df.loc[e, ['srce', 'trgt']]
    # Extract source and target positions as numpy arrays
    endpoint1 = sheet.vert_df.loc[srce_id, ['x', 'y']].values
    endpoint2 = sheet.vert_df.loc[trgt_id, ['x', 'y']].values
    endpoints = [endpoint1, endpoint2]
    for v in boundary_vert:
        if v != srce_id and v!= trgt_id:
            vertex = sheet.vert_df.loc[v,['x','y']].values
            dist, nearest = pnt2line(vertex, endpoint1 , endpoint2)
            # Check the distance from the vertex to the edge.
            if dist < 1.1:
                print(f'srce = {srce_id}, trgt = {trgt_id}, v = {v}')
                # store the associated edges, aka, rank
                edge_associated = sheet.edge_df[(sheet.edge_df['srce'] == v) | (sheet.edge_df['trgt'] == v)]
                rank = (len(edge_associated) - len(edge_associated[edge_associated['opposite'] == -1]))/2 + len(edge_associated[edge_associated['opposite'] == -1])

                
                    
                closest_end_dist, closest_end = closest_pair_dist(nearest, endpoint1, endpoint2)
                for i in endpoints:
                    if closest_end_dist < 0.5:
                        # Then the nearest point is moved to be 0.5 away from the endpoint.
                        remaining = [item for item in endpoints if not (item == i).all()]
                        i2 = remaining[0]
                        a = vector(i2, i)
                        a_hat = a / round(np.linalg.norm(a),4)
                        nearest = i + 0.5*a_hat 
                sheet.vert_df.loc[v,['x','y']] = nearest
                print(f'edge {e} and vert {v} are too close. With nearest point at {nearest}. \n')

geom.update_all(sheet)
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)


""" Collect the rank of the vertex """






""" Check if the incoming vertex is from the adjacent cell """







""" This is the end of the script """
