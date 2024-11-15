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
from tyssue.draw.plt_draw import plot_forces

# import my own functions
from my_headers import *

"""For the interior edge case."""
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

for index, series in sheet.edge_df.iterrows():
    if series['srce'] == 14 and series['trgt'] == 45:
        print(f'The index is: {index}')
    elif series['srce'] == 33 and series['trgt'] == 9:
        print(f'The index is: {index}')
    elif series['srce'] == 16 and series['trgt'] == 2 :
        print(f'The index is: {index}')

sheet.vert_df.loc[45,'y'] +=0.5
sheet.vert_df.loc[9,'y'] +=0.5
sheet.vert_df.loc[35,'x'] +=0.5
geom.update_all(sheet)
sheet_view(sheet)
print('The length of the shortest edge is: '+ str(sheet.edge_df.loc[:,'length'].min()))

model = model_factory([
    effectors.LineTension, # This defines cell cell adhesion force.
    effectors.FaceContractility, # This defines deformation energy coefficient
    effectors.FaceAreaElasticity # This defines memrbane surface energy coefficient.
    ])
sheet.vert_df['viscosity'] = 1.0
sheet.update_specs(model.specs, reset=True)
geom.update_all(sheet)
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i, 'opposite'] == -1:
        sheet.edge_df.loc[i, 'line_tension'] /= 2
    else:
        continue
geom.update_all(sheet)

# Store the single edges as a deparate df
t=0
dt=1
while t <3:
    for index in sheet.edge_df.index :
        if sheet.edge_df.loc[index,'length'] < 1.05:
            print(f'Edge {index} is shorter than threshold')
            type1_transition(sheet, index, remove_tri_faces=False, multiplier=1.5)
        else:
            continue
    geom.update_all(sheet)
    
    # Force computing and updating positions.
    valid_active_verts = sheet.active_verts[sheet.active_verts.isin(sheet.vert_df.index)]
    pos = sheet.vert_df.loc[valid_active_verts, sheet.coords].values
    # Compute the moving direction.
    dot_r = my_ode(sheet)
    new_pos = pos + dot_r*dt
    # Save the new positions back to `vert_df`
    sheet.vert_df.loc[valid_active_verts , sheet.coords] = new_pos
    geom.update_all(sheet)
    
    fig, ax = plot_forces(sheet, geom, model, ['x', 'y'], scaling=0.1)
    t+=1


"""For the boundary edge case, both vertices have rank 2."""
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
    
# Now perform transitions.
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
    

merge_vertices(sheet, 37, 4)
geom.update_all(sheet)
sheet_view(sheet)

new_boundary_coord = sheet.vert_df.loc[4,['x','y']]
new_boundary_coord[0]-=0.5

for i in list(range(len(sheet.edge_df))):
    if sheet.edge_df.loc[i,'opposite'] == -1 and sheet.edge_df.loc[i,'trgt'] == 4:
            print(f'The edge {i} is of interest.')
    elif sheet.edge_df.loc[i,'opposite'] == -1 and sheet.edge_df.loc[i,'srce'] == 4:
            print(f'The edge {i} is of interest.')
    else:
        continue


put_vert(sheet, 52, new_boundary_coord)
geom.update_all(sheet)
sheet_view(sheet)


""" 
More on the vertex rank =1, and =2 case.
"""
from tyssue.topology.sheet_topology import split_vert
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

# We pick the edge 77.
print(sheet.edge_df.loc[52,['srce','trgt']])
srce, trgt, face = sheet.edge_df.loc[52, ["srce", "trgt", "face"]].astype(int)

vert = min(srce, trgt) # find the vertex that won't be reindexed
ret_code = collapse_edge(sheet, 52)
geom.update_all(sheet)
sheet_view(sheet)
split_vert(sheet, vert)
geom.update_all(sheet)
sheet_view(sheet)


type1_transition(sheet, 52)
geom.update_all(sheet)
sheet_view(sheet)




""" This is the end of the file. """
