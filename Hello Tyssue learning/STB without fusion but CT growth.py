#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script describes the movement of STB without fusion, but only the growth
of CTs underneath it.
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
from tyssue.topology.base_topology import collapse_edge, remove_face, add_vert, get_num_common_edges
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
from tyssue.draw.plt_draw import plot_forces, plot_forces2
from tyssue.config.draw import sheet_spec
# import my own functions
from my_headers import delete_face, lateral_split, time_step_bot

rng = np.random.default_rng(70)


""" Initialize the geometry """

num_x = 20
num_y = 2
sheet = Sheet.planar_sheet_2d('face', nx = num_x, ny=num_y, distx=1, disty=1)
geom.update_all(sheet)
# remove non-enclosed faces
sheet.remove(sheet.get_invalid())  
delete_face(sheet, face_deleting = 20)
delete_face(sheet, face_deleting = 21)
sheet.reset_index(order=True) 
geom.update_all(sheet)
  
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
sheet.get_extra_indices()


""" Assign cell properties """

# First we assign cell type in face data frame.
num_ct = num_x
sheet.face_df.loc[0:num_ct-1, "cell_type"] = 'CT'
sheet.face_df.loc[num_ct:, "cell_type"] = 'ST'

# Then update the edge data frame via 'face' column.

sheet.edge_df['cell_type'] = 'to be set'

for i in list(range(len(sheet.edge_df))):
    sheet.edge_df.loc[i, 'cell_type'] = sheet.face_df.loc[sheet.edge_df.loc[i,'face'],'cell_type']

sheet.edge_df['cell_type'] = 'to be set'

for i in list(range(len(sheet.edge_df))):
    sheet.edge_df.loc[i, 'cell_type'] = sheet.face_df.loc[sheet.edge_df.loc[i,'face'],'cell_type']


sheet.face_df['division_status'] = 'ready'

# Change the division_status for ST cells.
for i in list(range(len(sheet.face_df))):
    if sheet.face_df.loc[i, 'cell_type'] == 'ST':
        sheet.face_df.loc[i, 'division_status'] = 'N/A'
    else:
        continue

specs = {
    'edge': {
        'is_active': 1,
        'line_tension': 0.12,
        'ux': 0.0,
        'uy': 0.0,
        'uz': 0.0
    },
   'face': {
       'area_elasticity': 1.0,
       'contractility': 0.04,
       'is_alive': 1,
       'prefered_area': 1.0},
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
# =============================================================================
# 
# # Adjust for cell-boundary adhesion force.
# for i in sheet.edge_df.index:
#     if sheet.edge_df.loc[i, 'opposite'] == -1:
#         sheet.edge_df.loc[i, 'line_tension'] *=2
#     else:
#         continue
# 
# for i in sheet.edge_df.index:
#     if sheet.edge_df.loc[i, 'cell_type'] == 'ST':
#         sheet.edge_df.loc[i, 'is_alive'] =0
#     else:
#         continue
#     
# =============================================================================

    
# Disable left and right hand side vertices.
# =============================================================================
# for edge_id in sheet.edge_df.index:
#     if sheet.edge_df.loc[edge_id,'face'] == 20:
#         srce = sheet.edge_df.loc[edge_id,'srce']
#         sheet.vert_df.loc[srce,'is_active'] = 0
#     if sheet.edge_df.loc[edge_id,'face'] == 39:
#         srce = sheet.edge_df.loc[edge_id,'srce']
#         sheet.vert_df.loc[srce,'is_active'] = 0
#     if sheet.edge_df.loc[edge_id,'face'] == 0:
#         srce = sheet.edge_df.loc[edge_id,'srce']
#         sheet.vert_df.loc[srce,'is_active'] = 0
#     if sheet.edge_df.loc[edge_id,'face'] == 19:
#         srce = sheet.edge_df.loc[edge_id,'srce']
#         sheet.vert_df.loc[srce,'is_active'] = 0
# 
# for i in [0,19,20,39]:
#     sheet.face_df.loc[i,'is_alive'] = 0
# =============================================================================

geom.update_all(sheet)

fig, ax = plot_forces(sheet, geom, smodel, ['x', 'y'], scaling=0.1)



""" Modelling the CT growth """

def division_stb(sheet, cell_id, division_threshold, division_rate, growth_rate,dt):
    """Defines a lateral division behavior.
    The function is composed of:
        1. check if the cell is CT cell and ready to split.
        2. generate a random number from (0,1), and compare with a threshold.
        3. two daughter cells starts growing until reach a threshold.
        
    
    Parameters
    ----------
    sheet: a :class:`Sheet` object
    cell_id: int
        the index of the dividing cell
    crit_area: float
        the area at which 
    growth_speed: float
        increase in the area per unit time
        A_0(t + dt) = A0(t) + growth_speed
    """

    # if the cell area is larger than the crit_area, we let the cell divide.
    if sheet.face_df.loc[cell_id, "cell_type"] == 'CT' and sheet.face_df.loc[cell_id, 'area'] > division_threshold:
        # A random float number is generated between (0,1)
        prob = np.random.uniform(0,1)
        if prob < division_rate:
            sheet.face_df.loc[face_id,'prefered_area'] = 1
            daughter = lateral_split(sheet, mother = cell_id)
            print(f"cell n°{daughter} is born")
            geom.update_all(sheet)

    elif sheet.face_df.loc[cell_id, "cell_type"] == 'CT' and sheet.face_df.loc[cell_id, 'area'] < division_threshold :
        sheet.face_df.loc[cell_id,'prefered_area'] = sheet.face_df.loc[cell_id,'area'] + dt*growth_rate
    
    else:
        pass


from tyssue import History# The History object records all the time steps 
history = History(sheet)


t1_threshold = sheet.edge_df.loc[:,'length'].min() / 10
t2_threshold = sheet.face_df.loc[:,'area'].min()/10
division_threshold = 2
max_movement = t1_threshold/2
d_min = t1_threshold
d_sep = d_min*1.5

t = 0

t_end = 1

while t <= t_end:
    dt = 0.01
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
    
    # Mannual split cell 9.
    sheet.face_df.loc[9,'prefered_area'] = 1
    daughter = lateral_split(sheet, 9)
    sheet.reset_index()
    geom.update_all(sheet)
    
# =============================================================================
#     # Cell division, use lateral_split.
#     for face_id in sheet.face_df.index:
#         if sheet.face_df.loc[face_id,'cell_type'] == 'ST':
#             continue
#         else:
#             if sheet.face_df.loc[face_id,'area'] > division_threshold:
#                 rand_num = rng.random()
#                 if rand_num < 0.1:
#                     sheet.face_df.loc[face_id,'prefered_area'] = 1
#                     daughter = lateral_split(sheet, face_id)
#                     sheet.reset_index()
#                     geom.update_all(sheet)
#                     print(f'cell {daughter} was born')
#                     
#             if sheet.face_df.loc[face_id, 'area'] < division_threshold:
#                 sheet.face_df.loc[face_id,'prefered_area'] += 0.1
#                 geom.update_all(sheet)
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
    
        
    t += dt

fig,ax = sheet_view(sheet)
ax.title.set_text(f't: {round(t,3)}')










"""
This is the end of the script.
"""
