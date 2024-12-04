#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single fusion on a bilayer
"""

# -*- coding: utf-8 -*-
"""
This script qualitatively shows biased cell division effect.
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

from tyssue.draw.plt_draw import plot_forces

# 2D plotting
from tyssue.draw import sheet_view, highlight_cells

# import my own functions
from my_headers import *

# Generate the cell sheet as three cells.

rng = np.random.default_rng(70)

num_x = 20
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
sheet.get_extra_indices()
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise

# Plot figures to check.
# Draw the cell mesh with face labelling and edge arrows.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)



""" Assign the cell type to face_df and edge_df """
num_ct = num_x
# First we assign cell type in face data frame.
sheet.face_df.loc[0:num_ct-1, "cell_type"] = 'CT'
sheet.face_df.loc[num_ct:, "cell_type"] = 'ST'

# Then update the edge data frame via 'face' column.

sheet.edge_df['cell_type'] = 'to be set'

for i in list(range(len(sheet.edge_df))):
    sheet.edge_df.loc[i, 'cell_type'] = sheet.face_df.loc[sheet.edge_df.loc[i,'face'],'cell_type']

# First, we need a way to compute the energy, then use gradient descent.
specs = {
    'edge': {
        'is_active': 1,
        'line_tension': 5,
        'ux': 0.0,
        'uy': 0.0,
        'uz': 0.0
    },
   'face': {
       'area_elasticity': 55,
       'contractility': 0,
       'is_alive': 1,
       'prefered_area': 1},
   'settings': {
       'grad_norm_factor': 1.0,
       'nrj_norm_factor': 1.0
   },
   'vert': {
       'is_active': 1
   }
}
sheet.vert_df['viscosity'] = 1.0
# Update the specs (adds / changes the values in the dataframes' columns)
sheet.update_specs(specs, reset = True)
geom.update_all(sheet)
sheet.get_opposite()

# Fuse cell 9 and 29
fusing_edge = neighbour_edge(sheet, 9, 29)

new_id = edge_remover(sheet, fusing_edge)

geom.update_all(sheet)

update_cell_type(sheet, new_id)

fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)



# Define the list of faces to check
target_faces = [0, 19, 20, 39]

# Step 1: Identify edges associated with target faces
edges_with_target_faces = sheet.edge_df[sheet.edge_df['face'].isin(target_faces)]

# Step 2: Extract all unique vertices (`srce` and `trgt`) from these edges
vertices_to_deactivate = pd.concat([edges_with_target_faces['srce'], 
                                     edges_with_target_faces['trgt']]).unique()

# Step 3: Update `is_active` in `vert_df` for these vertices
sheet.vert_df.loc[sheet.vert_df.index.isin(vertices_to_deactivate), 'is_active'] = 0


# Adjust for cell-boundary adhesion force.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i, 'opposite'] == -1:
        sheet.edge_df.loc[i, 'line_tension'] *=2
    else:
        continue

for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i, 'cell_type'] == 'ST':
        sheet.edge_df.loc[i, 'line_tension'] *=0.04
    else:
        continue


geom.update_all(sheet)

fig, ax = plot_forces(sheet, geom, smodel, ['x', 'y'], scaling=0.1)

# We need set the all the threshold value first.
t1_threshold = sheet.edge_df.loc[:,'length'].mean()/10
t2_threshold = sheet.face_df.loc[:,'area'].mean()/10
division_threshold = sheet.face_df.loc[:,'area'].mean()*1.5
growth_speed = sheet.face_df.loc[:,'area'].mean() *0.8
max_movement = t1_threshold/2
daughter = lateral_split(sheet, mother = 10)
geom.update_all(sheet)
sheet_view(sheet)
# Now assume we want to go from t = 0 to t= 0.2, dt = 0.1
t0 = 0
t_end = 30
dt = 0.01
time_points = np.linspace(t0, t_end, int((t_end - t0) / dt) + 1)
snapshots = [0.5,2.5, 5, 10, t_end]
print(f'time points are: {time_points}.')

for t in time_points:
    print(f'start at t= {round(t, 5)}.')
    
    # Mesh restructure check
    # T1 transition, edge rearrangment check
    while True:
    # Check for any edge below the threshold, starting from index 0 upwards
        edge_to_process = None
        for index in sheet.edge_df.index:
            if sheet.edge_df.loc[index, 'length'] < t1_threshold:
                edge_to_process = index
                edge_length = sheet.edge_df.loc[edge_to_process,'length']
                print(f'Edge {edge_to_process} is too short: {edge_length}')
                # Process the identified edge with T1 transition
                type1_transition(sheet, edge_to_process,remove_tri_faces=False, multiplier=1.5)
                break  
        # Exit the loop if no edges are below the threshold
        if edge_to_process is None:
            break
    geom.update_all(sheet)

    # T2 transition check.
    tri_faces = sheet.face_df[(sheet.face_df["num_sides"] < 4) & 
                          (sheet.face_df["area"] < t2_threshold)].index
    while len(tri_faces):
        remove_face(sheet, tri_faces[0])
        # Recompute the list of triangular faces below the area threshold after each removal
        tri_faces = sheet.face_df[(sheet.face_df["num_sides"] < 4) & 
                                  (sheet.face_df["area"] < t2_threshold)].index
    sheet.reset_index(order = True)
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
    
    # Plot with title contain time.
    if t in snapshots:
        fig, ax = sheet_view(sheet)
        ax.title.set_text(f'time = {round(t, 5)}')










"""
This is the end of the script.
"""
