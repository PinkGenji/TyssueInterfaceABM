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

fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)

len(sheet.vert_df)

# =============================================================================
# The following code is how you move a single vertex position.
# move vert 27 lower.

# sheet.vert_df.loc[27, ['x','y']] = [9,2]
# geom.update_all(sheet)
# =============================================================================

# =============================================================================
# The following code is when you create a new vertex, then reconnect an edge
# to the newly created vertex.

# sheet.edge_df.loc[sheet.edge_df['srce']==27,'srce'] = 25
# sheet.edge_df.loc[sheet.edge_df['trgt']==27,'trgt'] = 25
# geom.update_all(sheet)
# sheet.reset_index()
# =============================================================================

# =============================================================================
# The following code is how to compute the rank.
# The logic is total rank = (2*double edges)/2 + single edges

# edge_associated = sheet.edge_df[(sheet.edge_df['srce'] == 27) | (sheet.edge_df['trgt'] == 27)]
# rank = (len(edge_associated) - len(edge_associated[edge_associated['opposite'] == -1]))/2 + len(edge_associated[edge_associated['opposite'] == -1])
# print(rank)
# =============================================================================

# =============================================================================
# The following code is how to compute get a sorted rows of sheet.vert that are conncected
# to the incoming vertex.

# vert_associated = list(set(edge_associated['srce'].tolist() + edge_associated['trgt'].tolist()) - {27})
# print(vert_associated)
# filtered_rows = sheet.vert_df[sheet.vert_df.index.isin(vert_associated)]
# sorted_rows = filtered_rows.sort_values(by='x', ascending = False)
# sorted_rows
# =============================================================================

    
# =============================================================================
sheet.reset_index()
sheet.reset_topo()
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)

sheet.vert_df.loc[26,'y'] -= 1
sheet.vert_df.loc[26,'x'] -= 0.5
geom.update_all(sheet)
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)



boundary_vert, boundary_edge = find_boundary(sheet)


"""Check which vertex will collide with which edge"""    

# Assume d_min = 1.1, d_sep = 1.2

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
                    print(f'end1 = {srce_id}, end2 = {trgt_id}, v = {v}')
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
                                
    geom.update_all(sheet)
    sheet.reset_index()
                        
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y, face)




""" This is the end of the script """
