# -*- coding: utf-8 -*-
"""
This script contains the most fundamental demo/code for my own cell division
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

# import my own functions
from my_headers import delete_face, xprod_2d, put_vert, lateral_split, divisibility_check



""" Here is the biased division. """
# Generate the cell sheet as three cells.
sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = 3, ny = 2, distx = 1, disty = 1)
geom.update_all(sheet)

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Plot the figure to see the index.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
    
delete_face(sheet, 4)
delete_face(sheet, 3)
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise

# Plot figures to check.
# Draw the cell mesh with face labelling and edge arrows.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)


# Energy minimization
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
sheet.update_specs(specs, reset = True)
geom.update_all(sheet)
solver = QSSolver()
res = solver.find_energy_min(sheet, geom, smodel)
sheet_view(sheet) 

# Draw with vertex labelling.
fig, ax= sheet_view(sheet, edge = {'head_width':0.1})
for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)

# Compute the basal boundary edge.
sheet.get_opposite()
condition = sheet.edge_df.loc[:,'face'] == 1
edge_in_cell = sheet.edge_df[condition]
basal_edges = edge_in_cell[ edge_in_cell.loc[:,'opposite']==-1 ]
basal_edge_index = basal_edges.index[np.random.randint(0,len(basal_edges))]
#get the vertex index of the newly added mid point.
basal_mid = add_vert(sheet, edge = basal_edge_index)[0]
print(basal_mid)
geom.update_all(sheet)

# Draw with vertex labelling.
fig, ax= sheet_view(sheet, edge = {'head_width':0.1})
for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)

condition = sheet.edge_df.loc[:,'face'] == 1
edge_in_cell = sheet.edge_df[condition]
# We use the notation: line = P0 + dt, where P0 is the offset point and d is
# the direction vector, t is the lambda variable.
condition = edge_in_cell.loc[:,'srce'] == basal_mid
# extract the x-coordiante from array, then convert to a float type.

c0x = float(edge_in_cell[condition].loc[:,'fx'].values[0])
c0y = float(edge_in_cell[condition].loc[:,'fy'].values[0])
c0 = [c0x, c0y]

# The append function adds the new row in the last row, we the use iloc to 
# get the index of the last row, hence the index of the centre point.
cent_index = sheet.vert_df.index[-1]

p0x = float(edge_in_cell[condition].loc[:,'sx'].values[0])
p0y = float(edge_in_cell[condition].loc[:,'sy'].values[0])
p0 = [p0x, p0y]

rx = float(edge_in_cell[condition].loc[:,'rx'].values[0])
ry = float(edge_in_cell[condition].loc[:,'ry'].values[0])
r  = [-rx, -ry]   # use the line in opposite direction.


# We need to use iterrows to iterate over rows in pandas df
# The iteration has the form of (index, series)
# The series can be sliced.
for index, row in edge_in_cell.iterrows():
    s0x = row['sx']
    s0y = row['sy']
    t0x = row['tx']
    t0y = row['ty']
    v1 = [s0x-p0x,s0y-p0y]
    v2 = [t0x-p0x,t0y-p0y]
    # if the xprod_2d returns negative, then line intersects the line segment.
    if xprod_2d(r, v1)*xprod_2d(r, v2) < 0:
        #print(f'The edge that is intersecting is: {index}')
        dx = row['dx']
        dy = row['dy']
        c1 = dx*ry-dy*rx
        c2 = s0y*rx-p0y*rx - s0x*ry + p0x*ry
        k=c2/c1
        intersection = [s0x+k*dx, s0y+k*dy]
        new_index = put_vert(sheet, index, intersection)[0]
    else:
        print('Error! No opposite intersection!')
print(f'The intersection has coordinates: {intersection} with edge: {index}. ')

first_half = face_division(sheet, mother = 1, vert_a = basal_mid, vert_b = new_index )
added = put_vert(sheet, 39, c0)
geom.update_all(sheet)


fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)


fig, ax= sheet_view(sheet)
for edge, data in edge_in_cell.iterrows():
    ax.text((data.sx+data.tx)/2, (data.sy+data.ty)/2, edge)
        
sheet.face_df.loc[1]
sheet.edge_df.loc[sheet.edge_df.loc[:,'face'] == 1]
sheet.update_num_sides()

""" Jump here for shorted. """
if divisibility_check(sheet, cell_id = 1):
    daughter = lateral_split(sheet, mother = 1)
    geom.update_all(sheet)
    
    fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
    for face, data in sheet.face_df.iterrows():
        ax.text(data.x, data.y, face)
else:
    print('Not appropriate cell to divide.')


    
""" Now we do for a non-orientated division """
# Generate the cell sheet as three cells.

sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = 3, ny = 2, distx = 1, disty = 1)
geom.update_all(sheet)

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Plot the figure to see the index.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
    
delete_face(sheet, 4)
delete_face(sheet, 3)
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise

# Plot figures to check.
# Draw the cell mesh with face labelling and edge arrows.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)

# Draw with vertex labelling.
fig, ax= sheet_view(sheet, edge = {'head_width':0.1})
for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)

# Store the centroid before iteration.
unique_edges_df = sheet.edge_df.drop_duplicates(subset='face')
centre_data = unique_edges_df.loc[:,['face','fx','fy']]

# Do division, pick number 2 cell for example.
condition = sheet.edge_df.loc[:,'face'] == 2
edge_in_cell = sheet.edge_df[condition]
# We need to randomly choose one of the edges in cell 2.
chosen_index = int(np.random.choice(list(edge_in_cell.index) , 1))
# Extract and store the centroid coordinate.
c0x = float(centre_data.loc[centre_data['face']==2, ['fx']].values[0])
c0y = float(centre_data.loc[centre_data['face']==2, ['fy']].values[0])
c0 = [c0x, c0y]
print(f'centre is: {c0}')

# Add a vertex in the middle of the chosen edge.
new_mid_index = add_vert(sheet, edge = chosen_index)[0]

# Extract for source vertex coordinates of the newly added vertex.
p0x = sheet.vert_df.loc[new_mid_index,'x']
p0y = sheet.vert_df.loc[new_mid_index,'y']
p0 = [p0x, p0y]

# Compute the directional vector from new_mid_point to centroid.
rx = c0x - p0x
ry = c0y - p0y
r  = [rx, ry]   # use the line in opposite direction.

# We need to use iterrows to iterate over rows in pandas df
# The iteration has the form of (index, series)
# The series can be sliced.
for index, row in edge_in_cell.iterrows():
    s0x = row['sx']
    s0y = row['sy']
    t0x = row['tx']
    t0y = row['ty']
    v1 = [s0x-p0x,s0y-p0y]
    v2 = [t0x-p0x,t0y-p0y]
    # if the xprod_2d returns negative, then line intersects the line segment.
    if xprod_2d(r, v1)*xprod_2d(r, v2) < 0 and index !=chosen_index :
        dx = row['dx']
        dy = row['dy']
        c1 = dx*ry-dy*rx
        c2 = s0y*rx-p0y*rx - s0x*ry + p0x*ry
        k=c2/c1
        intersection = [s0x+k*dx, s0y+k*dy]
        oppo_index = put_vert(sheet, index, intersection)[0]
    else:
        continue
# Split the cell with a line.
new_face_index = face_division(sheet, mother = 2, vert_a = new_mid_index , vert_b = oppo_index )
# Put a vertex at the centroid, on the newly formed edge (last row in df).
cent_index = put_vert(sheet, edge = sheet.edge_df.index[-1], coord_put = c0)[0]
# update geometry
geom.update_all(sheet)


# Draw with vertex labelling.
fig, ax= sheet_view(sheet, edge = {'head_width':0.1})
for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)


""" Implement an Euler simple forward solver. """
# Generate the cell sheet as three cells.
sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = 3, ny = 2, distx = 1, disty = 1)
geom.update_all(sheet)

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Plot the figure to see the index.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
    
delete_face(sheet, 4)
delete_face(sheet, 3)
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise

# Plot figures to check.
# Draw the cell mesh with face labelling and edge arrows.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)

# Draw with vertex labelling.
fig, ax= sheet_view(sheet, edge = {'head_width':0.1})
for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)

# First, we need a way to compute the energy, then use gradient descent.
model = model_factory([
    effectors.LineTension,
    effectors.FaceContractility,
    effectors.FaceAreaElasticity
    ])

sheet.vert_df['viscosity'] = 1.0
sheet.update_specs(model.specs, reset=True)
geom.update_all(sheet)

def my_ode(eptm):
    valid_verts = sheet.active_verts[sheet.active_verts.isin(sheet.vert_df.index)]
    grad_U = model.compute_gradient(eptm).loc[valid_verts]
    dr_dt = -grad_U.values/eptm.vert_df.loc[valid_verts, 'viscosity'].values[:,None]
    return dr_dt

def current_pos(eptm):
    valid_verts = sheet.active_verts[sheet.active_verts.isin(sheet.vert_df.index)]
    return eptm.vert_df.loc[valid_verts, eptm.coords].values

# Now assume we want to go from t = 0 to t= 1, dt = 0.1
t0 = 0
t_end = 0.1
dt = 0.01
time_points = np.linspace(t0, t_end, int((t_end - t0) / dt) + 1)
print(f'time points are: {time_points}.')

for t in time_points:
    print(f'start at t= {round(t, 5)}.')
    valid_active_verts = sheet.active_verts[sheet.active_verts.isin(sheet.vert_df.index)]
    pos = sheet.vert_df.loc[valid_active_verts, sheet.coords].values
    # Compute the moving direction.
    dot_r = my_ode(sheet)
    new_pos = pos + dot_r*dt
    # Save the new positions back to `vert_df`
    sheet.vert_df.loc[valid_active_verts , sheet.coords] = new_pos
    geom.update_all(sheet)
    # Plot with title contain time.
    fig, ax = sheet_view(sheet)
    ax.title.set_text(f'time = {round(t, 5)}')

""" Now implement mesh restructure at each time step. """

# T1 threshold is typically 1 magnitude smaller than a typical cell area.
T1_threshold = sheet.face_df.loc[:,'area'].mean()/2

sheet.get_extra_indices() # Computes extra indicies.
sheet.sgle_edges # Show all joint index over free and east edges.
for i in sheet.sgle_edges:
    if sheet.edge_df.loc[i,'length'] < T1_threshold:
        type1_transition(sheet, edge01 = i, multiplier=1.5)
    else:
        continue

sheet.reset_index()
geom.update_all(sheet)
sheet_view(sheet)

def T1_check(eptm, threshold, scale):
    for i in eptm.sgle_edges:
        if eptm.edge_df.loc[i,'length'] < threshold:
            type1_transition(eptm, edge01 = i, multiplier= scale)
            print(f'Type 1 transition applied to edge {i} \n')
        else:
            continue

# Now assume we want to go from t = 0 to t= 1, dt = 0.1
d_min = 0.2     # This is the value that will be used for T1 swap.
t0 = 0
t_end = 0.1
dt = 0.01
time_points = np.linspace(t0, t_end, int((t_end - t0) / dt) + 1)
print(f'time points are: {time_points}.')
sheet.get_extra_indices()
for t in time_points:
    print(f'start at t= {round(t, 5)}.')
    # Mesh restructure check
    for i in sheet.sgle_edges:
        if sheet.edge_df.loc[i,'length'] < T1_threshold:
            type1_transition(sheet, edge01 = i, multiplier=1.5)
        else:
            continue
    sheet.reset_index()
    geom.update_all(sheet)
    
    # Force computing and updating positions.
    valid_active_verts = sheet.active_verts[sheet.active_verts.isin(sheet.vert_df.index)]
    pos = sheet.vert_df.loc[valid_active_verts, sheet.coords].values
    # Compute the moving direction.
    dot_r = my_ode(sheet)
    new_pos = pos + max(-d_min/2, min(dot_r*dt, d_min/2))
    # Save the new positions back to `vert_df`
    sheet.vert_df.loc[valid_active_verts , sheet.coords] = new_pos
    geom.update_all(sheet)
    
    # Plot with title contain time.
    fig, ax = sheet_view(sheet)
    ax.title.set_text(f'time = {round(t, 5)}')












""" This is the end of the code """
