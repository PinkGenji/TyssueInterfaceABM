# -*- coding: utf-8 -*-
"""
This script shows basic logic of cell cycle during division.
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
from tyssue.config.draw import sheet_spec
# import my own functions
from my_headers import *

rng = np.random.default_rng(70)

# Generate the cell sheet as three cells.
num_x = 1
num_y = 1
sheet = Sheet.planar_sheet_2d('face', nx = num_x, ny=num_y, distx=1, disty=1)
geom.update_all(sheet)
# remove non-enclosed faces
sheet.remove(sheet.get_invalid())  
delete_face(sheet, 1)
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise
sheet_view(sheet)
sheet.get_extra_indices()
# We need to creata a new colum to store the cell cycle time, default a 0, then minus.
sheet.face_df['T_cycle'] = 0
# Visualize the sheet.
fig, ax = sheet_view(sheet,  mode = '2D')
# First, we need a way to compute the energy, then use gradient descent.
specs = {
    'edge': {
        'is_active': 1,
        'line_tension': 10,
        'ux': 0.0,
        'uy': 0.0,
        'uz': 0.0
    },
   'face': {
       'area_elasticity': 110,
       'contractility': 0,
       'is_alive': 1,
       'prefered_area': 2},
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

# Adjust for cell-boundary adhesion force.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i, 'opposite'] == -1:
        sheet.edge_df.loc[i, 'line_tension'] *=2
    else:
        continue
geom.update_all(sheet)

fig, ax = plot_forces(sheet, geom, smodel, ['x', 'y'], scaling=0.1)

# We need set the all the threshold value first.
t1_threshold = 0.1
t2_threshold = 0.1
division_threshold = 1
inhibition_threshold = 0.8
max_movement = t1_threshold/2

# Now assume we want to go from t = 0 to t= 0.2, dt = 0.1
t0 =0
t_end = 50
dt = 0.001
time_points = np.linspace(t0, t_end, int((t_end - t0) / dt) + 1)
print(f'time points are: {time_points}')
t = t0
while t <= t_end:
    dt = 0.001
    print(f'start at t= {round(t, 5)}')

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
    
    # Cell division.
    # Store the centroid before iteration of cells.
    unique_edges_df = sheet.edge_df.drop_duplicates(subset='face')
    centre_data = unique_edges_df.loc[:,['face','fx','fy']]
    # Loop over all the faces.
    cells_can_divide = sheet.face_df[(sheet.face_df['area'] >= division_threshold) & (sheet.face_df['T_cycle'] == 0)]
    for index, series in cells_can_divide.iterrows():
        daughter_index = division_2(sheet,rng=rng, cent_data= centre_data, cell_id = index)
    sheet.reset_index(order = True)
    geom.update_all(sheet)

    # Force computing and updating positions.
    valid_active_verts = sheet.active_verts[sheet.active_verts.isin(sheet.vert_df.index)]
    pos = sheet.vert_df.loc[valid_active_verts, sheet.coords].values
    # get the movement of position based on dynamical dt.
    dt, movement = time_step_bot(sheet, dt, max_dist_allowed = max_movement )
    new_pos = pos + movement
    # Save the new positions back to `vert_df`
    sheet.vert_df.loc[valid_active_verts , sheet.coords] = new_pos
    geom.update_all(sheet)
    
    #Need to update the T_cycle value based on their compression time.
    become_free = sheet.face_df[(sheet.face_df['area'] >= inhibition_threshold) & (sheet.face_df['T_cycle'] > 0)]
    for i in become_free.index:
        sheet.face_df.loc[i,'T_cycle'] = round(sheet.face_df.loc[i,'T_cycle']- dt, 3)
        T_cyc = sheet.face_df.loc[i,'T_cycle']
        print(f'T_cycle for {i} is {T_cyc}')
    geom.update_all(sheet)

    mean_area = sheet.face_df.loc[:,'area'].mean()
    max_area = sheet.face_df.loc[:,'area'].max()
    min_area = sheet.face_df.loc[:,'area'].min()
    print(f'At time {round(t, 3)}, mean area: {mean_area}, max area: {max_area}, min area: {min_area}')
    # # Plot with title contain time.
    if t in time_points[::1000]:
        fig, ax = sheet_view(sheet)
        ax.title.set_text(f'time = {round(t, 5)}, mean area: {mean_area}')
        
    t +=dt
    
# =============================================================================
# 
# draw_specs = sheet_spec()
# draw_specs['face']['visible'] = True
# okay = sheet.face_df[(sheet.face_df['area'] >= division_threshold) & (sheet.face_df['T_cycle'] > 0)]
# for i in sheet.face_df.index:
#     if i in okay.index:
#         sheet.face_df['color'] = 0
#     else:
#         sheet.face_df['color'] = 1
#         
# draw_specs['face']['color'] = sheet.face_df['color']
# draw_specs['face']['alpha'] = 0.5
# fig, ax = sheet_view(sheet,['x', 'y'], **draw_specs)
# =============================================================================





""" This is the end of the script. """
