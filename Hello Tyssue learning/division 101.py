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
from my_headers import delete_face, xprod_2d, put_vert, lateral_split

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
cent_dict = {'y': c0y, 'is_active': 1, 'x': c0x}
sheet.vert_df = sheet.vert_df.append(cent_dict, ignore_index = True)
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
        c1 = (dx*ry/rx)-dy
        c2 = s0y-p0y - (s0x*ry/rx) + (p0x*ry/rx)
        k=c2/c1
        intersection = [s0x+k*dx, s0y+k*dy]
        new_index = put_vert(sheet, index, intersection)[0]
print(f'The intersection has coordinates: {intersection} with edge: {index}. ')

first_half = face_division(sheet, mother = 1, vert_a = basal_mid, vert_b = cent_index )
second_half = face_division(sheet, mother = 1, vert_a = new_index, vert_b = cent_index)
geom.update_all(sheet)

fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)


fig, ax= sheet_view(sheet)
for edge, data in edge_in_cell.iterrows():
    ax.text((data.sx+data.tx)/2, (data.sy+data.ty)/2, edge)
        



""" Jump here for shorted. """
daughter = lateral_split(sheet, mother = 1)
geom.update_all(sheet)


fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)

""" This is the end of the code """
