# -*- coding: utf-8 -*-
"""
This script simulates a petri dish case (contact inhibition) after my PhD PYR. 
In this simulation, the new T3 is implemented.
"""

# =============================================================================
# First we need to surpress the version warnings from Pandas.
import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning) 
# =============================================================================

# Load all required modules.

import numpy as np
import pandas as pd
from decimal import Decimal

import os
import json
import matplotlib.pyplot as plt

import ipyvolume as ipv

from tyssue import Sheet, config #import core object
from tyssue import PlanarGeometry as geom #for simple 2d geometry

# For cell topology/configuration
from tyssue.topology.sheet_topology import type1_transition
from tyssue.topology.base_topology import collapse_edge, add_vert
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
from T3_function import *

# Set up the random number generator (RNG)
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
t = Decimal("0")

dt = Decimal("0.001")


t_end = Decimal("100")


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
                    sheet.reset_index(order=False)
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

    # Cell division.
    # Store the centroid before iteration of cells.
    unique_edges_df = sheet.edge_df.drop_duplicates(subset='face')
    centre_data = unique_edges_df.loc[:,['face','fx','fy']]    
    # only the cells that have large enough area and completed its mitosis cycle should be divided.
    cells_can_divide = sheet.face_df[(sheet.face_df['area'] >= division_threshold) & (sheet.face_df['T_cycle'] == 0)]
    for index, series in cells_can_divide.iterrows():
        daughter_index = division_mt(sheet,rng=rng, cent_data= centre_data, cell_id = index)
# =============================================================================
#     I commented out this part of the code since I don't think we need T_age anymore.
#
#     # Update the T_age in mitosis.
#     cells_are_mitosis = sheet.face_df[(sheet.face_df['T_age'] != sheet.face_df['T_cycle'])]
#     for i in cells_are_mitosis.index:
#         sheet.face_df.loc[i,'prefered_area'] = 1/2*(sheet.face_df.loc[i,'T_age']/sheet.face_df.loc[i,'T_cycle']+1)
# =============================================================================
    
    sheet.reset_index(order = True)
    geom.update_all(sheet)
    
    
    # Force computing and updating positions.
    valid_active_verts = sheet.active_verts[sheet.active_verts.isin(sheet.vert_df.index)]
    pos = sheet.vert_df.loc[valid_active_verts, sheet.coords].values
    # get the movement of position based on dynamical dt.
    dt, movement = time_step_bot(sheet, dt, max_dist_allowed = max_movement )
    new_pos = pos + movement
    dt = Decimal(dt)
    # Save the new positions back to `vert_df`
    sheet.vert_df.loc[valid_active_verts , sheet.coords] = new_pos
    geom.update_all(sheet)
    
    # We loop over every cell in the system.
    # If T_cycle of the cell is zero, then do nothing.
    # If T_cycle of the cell is smaller than zero, then correct it to be zero.
    # If T_cycle of the cell is larger than zero, then minus it by dt.
    for cell in sheet.face_df.index:
        if sheet.face_df.loc[cell, 'T_cycle' ] < 0:
            sheet.face_df.loc[cell,'T_cycle'] = 0
        if sheet.face_df.loc[cell, 'T_cycle'] > 0:
            sheet.face_df.loc[cell,'T_cycle'] -= dt

        
    geom.update_all(sheet)
    
    # Add trackers for quantify.
    cell_num_count = len(sheet.face_df)
    mean_area = sheet.face_df.loc[:,'area'].mean()
    total_area = sheet.face_df.loc[:,'area'].sum()
    
    time_stamp.append(t)
    cell_counter.append(cell_num_count)
    cell_ave_intime.append(mean_area)
    area_intotal.append(total_area)

    print(f'At time {t}, there are {cell_num_count} cells, total_area: {total_area}\n')

    # if t in time_point[::10]:
    #     fig, ax = sheet_view(sheet)
    #     ax.title.set_text(f'time = {round(t, 5)}')
    
    # Update time_point
    t += dt
    t = t.quantize(Decimal("0.0001"))  # Keeps t rounded to 5 decimal places


fig, ax = sheet_view(sheet)
ax.title.set_text(f'time = {round(t, 5)}')












"""
This is the end of the script.
"""
