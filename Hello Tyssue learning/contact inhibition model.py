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


# We need set the all the threshold value first.
t1_threshold = 0.01
t2_threshold = 0.1
division_threshold = 1
inhibition_threshold = 0.9
max_movement = t1_threshold/2

# Now assume we want to go from t = 0 to t= 0.2, dt = 0.1
t = 0
t_end = 0.005

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
    
    # T3 transition.
    while True:
        edge_to_process = None
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
                        pass
                    else:
                        vertex = sheet.vert_df.loc[v,['x','y']].values
                        dist, nearest = pnt2line(vertex, endpoint1 , endpoint2)
                        # Check the distance from the vertex to the edge.
                        if dist < 1.1:
                            edge_to_process = e
                            # store the associated edges, aka, rank
                            edge_associated = sheet.edge_df[(sheet.edge_df['srce'] == v) | (sheet.edge_df['trgt'] == v)]
                            rank = (len(edge_associated) - len(edge_associated[edge_associated['opposite'] == -1]))/2 + len(edge_associated[edge_associated['opposite'] == -1])
                            vert_associated = list(set(edge_associated['srce'].tolist() + edge_associated['trgt'].tolist()) - {v})
                            filtered_rows = sheet.vert_df[sheet.vert_df.index.isin(vert_associated)]
                            # extend the edge if needed.
                            if sheet.edge_df.loc[e,'length'] < 1.2*rank:
                                extension_needed = 1.2*rank - sheet.edge_df.loc[e,'length']
                                edge_extension(sheet, e, extension_needed)
                            
                            # Check adjacency.
                            v_adj = adjacent_vert(sheet, v, srce_id, trgt_id)
                            if v_adj is not None:
                                # First we move the common point.
                                sheet.vert_df.loc[v_adj,['x','y']] = list(nearest)
                                
                                # Then, we need to update via put-vert and update
                                # sequentially by d_sep.
                                # The sequence is determined by the sign of the difference
                                # between x-value of (nearest - end)
                                if nearest[0] - sheet.vert_df.loc[v_adj,'x'] < 0:
                                    # Then shall sort x-value from largest to lowest.
                                    sorted_rows = filtered_rows.sort_values(by='x', ascending = False)
                                    sorted_rows_id = list(sorted_rows.index)
                        # Then pop twice since an extra put vert is only needed for rank 2 adjacent.
                                    sorted_rows_id.pop(0)
                                    sorted_rows_id.pop(0)
                                    
                                else: # Then shall sort from lowest to largest.
                                    sorted_rows = filtered_rows.sort_values(by='x', ascending = True)
                                    sorted_rows_id = list(sorted_rows.index)
                                    sorted_rows_id.pop(0)
                                    sorted_rows_id.pop(0)
                                
                                # If rank is > 2, then we need to compute more.
                                if sorted_rows_id: 
                                    # Store the starting point as the nearest, then compute the unit vector.
                                    last_coord = nearest
                                    a = vector(nearest , sheet.vert_df.loc[v_adj, ['x', 'y']].values)
                                    a_hat = a / round(np.linalg.norm(a),4)
                                    print(sorted_rows_id)
                                    for i in sorted_rows_id:
                                        last_coord += a_hat*1.2
                                        new_vert_id = put_vert(sheet, e, last_coord)[0]
                                        sheet.edge_df.loc[sheet.edge_df['srce']==i,'srce'] = new_vert_id
                                        sheet.edge_df.loc[sheet.edge_df['trgt']==i,'trgt'] = new_vert_id
                                        
                            # Now, for the case of non adjacent.
                            elif v_adj is None:
                                # The number of points we need to put on the edge is same as the rank.
                                a = vector(nearest , sheet.vert_df.loc[srce_id , ['x', 'y']].values)
                                a_hat = a / round(np.linalg.norm(a),4)
                                if rank == 2:
                                    coord1 = nearest - 0.6*a_hat
                                    coord2 = nearest + 0.6*a_hat
                                    new_id_1 = put_vert(sheet, e, coord1)
                                    new_id_2 = put_vert(sheet, e, coord2)
                                    new_vert_id = [new_id_1, new_id_2]
                                    # Now, the x-value sorting is based on the distance 
                                    # between the point to the srce_id.
                                    if nearest[0] - sheet.vert_df.loc[srce_id ,'x'] < 0:
                                        sorted_rows = filtered_rows.sort_values(by='x', ascending = True)
                                        sorted_rows_id = list(sorted_rows.index)
                                    if nearest[0] - sheet.vert_df.loc[srce_id ,'x'] < 0:
                                        sorted_rows = filtered_rows.sort_values(by='x', ascending = False)
                                        sorted_rows_id = list(sorted_rows.index)
                                    for i in sorted_rows_id:
                                        for j in new_vert_id:
                                            sheet.edge_df.loc[sheet.edge_df['srce']==i,'srce'] = j
                                            sheet.edge_df.loc[sheet.edge_df['trgt']==i,'trgt'] = j
                                elif rank ==3 :
                                    coord1 = nearest - 0.6*a_hat
                                    coord2 = nearest
                                    coord3 = nearest + 0.6*a_hat
                                    new_id_1 = put_vert(sheet, e, coord1)
                                    new_id_2 = put_vert(sheet, e, coord2)
                                    new_id_3 = put_vert(sheet, e, coord3)
                                    new_vert_id = [new_id_1, new_id_2, new_id_3]
                                    # Now, the x-value sorting is based on the distance 
                                    # between the point to the srce_id.
                                    if nearest[0] - sheet.vert_df.loc[srce_id ,'x'] < 0:
                                        sorted_rows = filtered_rows.sort_values(by='x', ascending = True)
                                        sorted_rows_id = list(sorted_rows.index)
                                    if nearest[0] - sheet.vert_df.loc[srce_id ,'x'] < 0:
                                        sorted_rows = filtered_rows.sort_values(by='x', ascending = False)
                                        sorted_rows_id = list(sorted_rows.index)
                                    for i in sorted_rows_id:
                                        for j in new_vert_id:
                                            sheet.edge_df.loc[sheet.edge_df['srce']==i,'srce'] = j
                                            sheet.edge_df.loc[sheet.edge_df['trgt']==i,'trgt'] = j  
            
        if edge_to_process is None:
            break  # Exit loop if no edge was found to process
                
        geom.update_all(sheet)
        sheet.reset_index()
        
    
    # Cell division.
    # Store the centroid before iteration of cells.
    unique_edges_df = sheet.edge_df.drop_duplicates(subset='face')
    centre_data = unique_edges_df.loc[:,['face','fx','fy']]
    # Loop over all the faces.
    cells_can_divide = sheet.face_df[(sheet.face_df['area'] >= division_threshold) & (sheet.face_df['T_age'] == sheet.face_df['T_cycle'])]
    for index, series in cells_can_divide.iterrows():
        daughter_index = division_1(sheet,rng=rng, cent_data= centre_data, cell_id = index, dt = dt)
    # Update the T_age in mitosis.
    cells_are_mitosis = sheet.face_df[(sheet.face_df['T_age'] != sheet.face_df['T_cycle'])]
    for i in cells_are_mitosis.index:
        sheet.face_df.loc[i,'prefered_area'] = 1/2*(sheet.face_df.loc[i,'T_age']/sheet.face_df.loc[i,'T_cycle']+1)
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
    become_free = sheet.face_df[(sheet.face_df['area'] >= inhibition_threshold) & (sheet.face_df['T_age'] < sheet.face_df['T_cycle'])]
    for i in become_free.index:
        sheet.face_df.loc[i,'T_age'] = round(sheet.face_df.loc[i,'T_age']+ dt, 3)
        T_age = sheet.face_df.loc[i,'T_age']
        print(f'T_age for {i} is {T_age}')
    geom.update_all(sheet)
    

    mean_area = sheet.face_df.loc[:,'area'].mean()
    max_area = sheet.face_df.loc[:,'area'].max()
    min_area = sheet.face_df.loc[:,'area'].min()
    print(f'mean area: {mean_area}, max area: {max_area}, min area: {min_area}')
    
    # # Plot with title contain time.
    # fig, ax = sheet_view(sheet)
    # ax.title.set_text(f'time = {round(t, 5)}, mean area: {mean_area}')
    # update time_point:
    t += dt




""" This is the end of the script. """
