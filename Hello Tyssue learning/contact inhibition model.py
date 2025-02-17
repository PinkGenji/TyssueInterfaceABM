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
sheet.face_df['T_age'] = 0
# Visualize the sheet.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
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
t1_threshold = 0.01
t2_threshold = 0.1
d_min = 0.0008
d_sep = 0.011
division_threshold = 1
inhibition_threshold = 0.8
max_movement = t1_threshold/2
time_stamp = []
cell_counter = []
area_intotal = []
cell_ave_intime = []
# Now assume we want to go from t = 0 to t= 0.2, dt = 0.1
t0 = 0
t_end = 50
dt = 0.001
time_point = np.linspace(t0, t_end, int((t_end - t0) / dt) + 1)
t = t0
while t <= t_end:
    dt = 0.001
    #print(f'start at t= {round(t, 5)}')

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
    
    # T3 transition.
    while True:
        T3_collision = None
        boundary_vert, boundary_edge = find_boundary(sheet)
        if not boundary_edge:  # Exit if no boundary edges are found
            break
        
        for e in boundary_edge:
            # Extract source and target vertex IDs
            srce_id, trgt_id = sheet.edge_df.loc[e, ['srce', 'trgt']]
            # Extract source and target positions as numpy arrays
            endpoint1 = sheet.vert_df.loc[srce_id, ['x', 'y']].values
            endpoint2 = sheet.vert_df.loc[trgt_id, ['x', 'y']].values
            endpoints = [endpoint1, endpoint2]
            for v in boundary_vert:
                #compute the dist needed for threshold comparing.
                if v != srce_id and v!= trgt_id:
                    if are_vertices_in_same_face(sheet, v, srce_id)==True and are_vertices_in_same_face(sheet, v, trgt_id)==True:
                        break
                    else:
                        vertex = sheet.vert_df.loc[v,['x','y']].values
                        dist, nearest1 = pnt2line(vertex, endpoint1 , endpoint2)
                        print(f'end1: {srce_id}, end2: {trgt_id}, v: {v}  ')
                        # Check the distance from the vertex to the edge.
                        if dist < 1.1:
                            T3_collision = e
                            T3_transition(sheet, e, v, d_min, d_sep, nearest1)
                            geom.update_all(sheet)
                            sheet.reset_index()
            break

        if T3_collision is None:
            break  # Exit loop if no edge was found to process

    # Cell division.
    # Store the centroid before iteration of cells.
    unique_edges_df = sheet.edge_df.drop_duplicates(subset='face')
    centre_data = unique_edges_df.loc[:,['face','fx','fy']]
    # Loop over all the faces.
    cells_can_divide = sheet.face_df[(sheet.face_df['area'] >= division_threshold) & (sheet.face_df['T_age'] == sheet.face_df['T_cycle'])]
    for index, series in cells_can_divide.iterrows():
        daughter_index = division_mt(sheet,rng=rng, cent_data= centre_data, cell_id = index, dt = dt)
    # Update the T_age in mitosis.
    cells_are_mitosis = sheet.face_df[(sheet.face_df['T_age'] != sheet.face_df['T_cycle'])]
    
# =============================================================================
#     for i in cells_are_mitosis.index:
#         new_prefered_area =  1/2*(sheet.face_df.loc[i,'T_age']/sheet.face_df.loc[i,'T_cycle'])+1/2
#         sheet.face_df.loc[i,'prefered_area'] = new_prefered_area
#         print(f'new_prefered_area: {new_prefered_area}')
#     sheet.reset_index(order = True)
#     geom.update_all(sheet)
# =============================================================================
    
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
    become_free = sheet.face_df[(sheet.face_df['area'] >= inhibition_threshold) & (sheet.face_df['T_age'] < sheet.face_df['T_cycle'])]
    for i in become_free.index:
        sheet.face_df.loc[i,'T_age'] = round(sheet.face_df.loc[i,'T_age']+ dt, 3)
        T_age = sheet.face_df.loc[i,'T_age']
        
    geom.update_all(sheet)
    
    # Add trackers for quantify.
    cell_num_count = len(sheet.face_df)
    mean_area = sheet.face_df.loc[:,'area'].mean()
    total_area = sheet.face_df.loc[:,'area'].sum()
    
    time_stamp.append(t)
    cell_counter.append(cell_num_count)
    cell_ave_intime.append(mean_area)
    area_intotal.append(total_area)

    print(f'At time {t}, total cell: {cell_num_count}, total_area: {total_area}\n')

    # if t in time_point[::10]:
    #     fig, ax = sheet_view(sheet)
    #     ax.title.set_text(f'time = {round(t, 5)}')
    
    # Update time_point
    t += dt
    t = round(t,5)





""" This is the end of the script. """
