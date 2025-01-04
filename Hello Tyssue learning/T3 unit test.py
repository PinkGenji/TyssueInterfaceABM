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
# Setup d_min and d_sep values.
d_min = sheet.edge_df.loc[:,'length'].min()/10
d_min=0.13
d_sep = d_min*3

print(f'd_min is set: {d_min}, d_sep is set: {d_sep}')

# Plot figures to check.
# Draw the cell mesh with face labelling and edge arrows.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)

fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)

""" Make adjustments to the geometry for unit test.  
Note: the code for case 1 and case 3 were wrote first, since creating case 2 
and case 4 involves vertex reindex. Hence I wrote case 1 & 3 first.
"""


# Case 1 from Fletcher 2013, move vertex number 13 close enough to 
# edge number 57, which is connecting vertex 51 and 50.
sheet.vert_df.loc[13,'x'] = 5.7
sheet.vert_df.loc[13,'y'] = 2.55

geom.update_all(sheet)
sheet.get_extra_indices()  
# Plot figures to check.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)


# Plot figures to check.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)

# =============================================================================
# 
# while True:
#     T3_todo = None
# 
#     boundary_vert, boundary_edge = find_boundary(sheet)
#     
#     for edge_e in boundary_edge:
#         # Extract source and target vertex IDs
#         srce_id, trgt_id = sheet.edge_df.loc[edge_e, ['srce', 'trgt']]
#         for vertex_v in boundary_vert:
#             if vertex_v == srce_id or vertex_v == trgt_id:
#                 continue
#             
#             distance, nearest = dist_computer(sheet, edge_e, vertex_v, d_sep)
#             if distance < d_min:
#                 T3_todo = vertex_v
#                 print(f'Found incoming vertex: {vertex_v} and colliding edge: {edge_e}')
#                 T3_swap(sheet, edge_e, vertex_v, nearest, d_sep)
#                 sheet.reset_index()
#                 geom.update_all(sheet)
#                 sheet_view(sheet, mode='quick')
#                 
# 
#                 break
#     if T3_todo is None:
#         break
# =============================================================================
                

# Case 3 from Fletcher 2013, changing the position of vertex 3.
sheet.vert_df.loc[3,'x'] = 0.9
sheet.vert_df.loc[3,'y'] = 2.3

geom.update_all(sheet)
sheet.get_extra_indices()  
sheet_view(sheet)

# Plot figures to check.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)



dist, nearest = dist_computer(sheet, 35, 3, d_sep)
v29 = sheet.vert_df.loc[29,['x','y']].to_numpy(dtype = float)
v26 = sheet.vert_df.loc[26,['x','y']].to_numpy(dtype = float)
dist29 = np.linalg.norm(v29-nearest)
dist26 = np.linalg.norm(v26-nearest)
dist29
dist26  
sheet.vert_df.loc[29,'x'] = nearest[0]
sheet.vert_df.loc[29,'y'] = nearest[1]

    
merge_vertices(sheet,3, 29, reindex=False)
sheet.reset_index()
geom.update_all(sheet)
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)


''' Do T3 '''
while True:
    T3_todo = None
    print('computing boundary indices.')
    boundary_vert, boundary_edge = find_boundary(sheet)
    
    for edge_e in boundary_edge:
        # Extract source and target vertex IDs
        srce_id, trgt_id = sheet.edge_df.loc[edge_e, ['srce', 'trgt']]
        for vertex_v in boundary_vert:
            if vertex_v == srce_id or vertex_v == trgt_id:
                continue
            
            distance, nearest = dist_computer(sheet, edge_e, vertex_v, d_sep)
            if distance < d_min:
                T3_todo = vertex_v
                print(f'Found incoming vertex: {vertex_v} and colliding edge: {edge_e}')
                T3_swap(sheet, edge_e, vertex_v, nearest, d_sep)
                sheet.reset_index()
                geom.update_all(sheet)
                sheet.get_extra_indices()
                fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
                for face, data in sheet.vert_df.iterrows():
                    ax.text(data.x, data.y, face)
                break
        
        if T3_todo is not None:
            break  # Exit outer loop to restart with updated boundary

            
    if T3_todo is None:
        break

    





# Case 4 from Fletcher 2013, first put an edge cuts face 16 by connecting
# vertex 12 and 10. then adjust the position of vertex 41, and
# move vertex 12 close enough to the edge formed by 
# vertex 42 and vertex 41, which is edge 106. Lastly adjust the position of vertex 10.

divide = face_division(sheet, 16, 12, 10)
sheet.reset_index()
geom.update_all(sheet)

sheet.vert_df.loc[41,'y'] = 5
sheet.vert_df.loc[12,'x'] = 3.3
sheet.vert_df.loc[12,'y'] = 4.8
sheet.vert_df.loc[10,'y'] =3.8

geom.update_all(sheet)
sheet.get_extra_indices()
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)
    
   
    

# Case 2 from Fletcher 2013, first，adjust the position of vertex 22, 47and 56.

sheet.vert_df.loc[22,'y'] = -1
sheet.vert_df.loc[47,'y'] = 0.5
sheet.vert_df.loc[56,'x'] = 3.2
sheet.vert_df.loc[56,'y'] = -0.5
daughter = face_division(sheet, 4,  56, 54)
geom.update_all(sheet)
sheet.get_extra_indices()
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)
    

while True:
    T3_todo = None
    boundary_vert, boundary_edge = find_boundary(sheet)
    
    for edge_e in boundary_edge:
        # Extract source and target vertex IDs
        srce_id, trgt_id = sheet.edge_df.loc[edge_e, ['srce', 'trgt']]
        for vertex_v in boundary_vert:
            if vertex_v == srce_id or vertex_v == trgt_id:
                continue
            
            distance, nearest = dist_computer(sheet, edge_e, vertex_v, d_sep)
            if distance < d_min:
                T3_todo = vertex_v
                print(f'Found incoming vertex: {vertex_v} and colliding edge: {edge_e}')
                T3_swap(sheet, edge_e, vertex_v, nearest, d_sep)
                sheet.reset_index()
                geom.update_all(sheet)
                sheet_view(sheet, mode='quick')
                break
            
    if T3_todo is None:
        break






""" Implement T3 transition """

# First, compute all boundary edge and boundary vertices.

boundary_vert, boundary_edge = find_boundary(sheet)

for edge_e in boundary_edge:
    # Extract source and target vertex IDs
    srce_id, trgt_id = sheet.edge_df.loc[edge_e, ['srce', 'trgt']]
    for vertex_v in boundary_vert:
        if vertex_v == srce_id or vertex_v == trgt_id:
            continue
        else:
            distance, nearest = dist_computer(sheet, edge_e, vertex_v, d_sep)
            if distance < d_min:
                print(f'Found incoming vertex: {vertex_v}')

                # T3_swap(sheet, edge_e, vertex_v, nearest, d_sep)
                # geom.update_all(sheet)
                # sheet_view(sheet, mode='quick')









""" This is the end of the script. """
