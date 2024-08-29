# -*- coding: utf-8 -*-
"""
This script investigate the angle direction within cell division function.
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
from tyssue.topology.base_topology import collapse_edge, remove_face
from tyssue.topology.sheet_topology import split_vert as sheet_split
from tyssue.topology.bulk_topology import split_vert as bulk_split
from tyssue.topology import condition_4i, condition_4ii

## model and solver
from tyssue.dynamics.planar_vertex_model import PlanarModel as smodel
from tyssue.solvers.quasistatic import QSSolver
from tyssue.generation import extrude
from tyssue.dynamics import model_factory, effectors
from tyssue.topology.sheet_topology import remove_face, cell_division, get_division_edges

# Event manager
from tyssue.behaviors import EventManager

# 2D plotting
from tyssue.draw import sheet_view, highlight_cells


# Generate the cell sheet as three cells.
sheet = Sheet.planar_sheet_2d('face', nx = 3, ny=4, distx=2, disty=2)
sheet.sanitize(trim_borders=True)
geom.update_all(sheet)
sheet.get_extra_indices()

fig, ax= sheet_view(sheet)
for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)


fig, ax= sheet_view(sheet)
for edge, data in sheet.edge_df.iterrows():
    # We only want the indexes that are in the east_edge list.
    if edge in sheet.east_edges:
        ax.text((data.sx+data.tx)/2, (data.sy+data.ty)/2, edge)
    else:
        continue
    
fig, ax= sheet_view(sheet)
for edge, data in sheet.edge_df.iterrows():
    # We only want the indexes that are in the west_edge list.
    if edge in sheet.west_edges:
        ax.text((data.sx+data.tx)/2, (data.sy+data.ty)/2, edge)
    else:
        continue

fig, ax= sheet_view(sheet)
for edge, data in sheet.edge_df.iterrows():
    # We only want the indexes that are in the west_edge list.
    if edge in sheet.free_edges:
        ax.text((data.sx+data.tx)/2, (data.sy+data.ty)/2, edge)
    else:
        continue

fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)

''' For a single face, we find the basal/free edges '''
# First we need to find all the edges associated with face x, for example x=1

def edges_within_a_face(sheet_obj, face_id):
    return sheet_obj.edge_df[(sheet_obj.edge_df['face'] == face_id)]

print(edges_within_a_face(sheet, 1))
face_1_edges = edges_within_a_face(sheet, 1)

def basal_edge_filter(given_edge_set, basal_edge_set):
    return [x for x in given_edge_set.index if x in basal_edge_set]

print(basal_edge_filter(face_1_edges, sheet.free_edges))



#now we obtain a list that contains the basal edges for face 1
face_1_basal_edges = basal_edge_filter(face_1_edges, sheet.free_edges)

''' How to find the opposite edge of the basal edge? '''
from tyssue.topology.base_topology import add_vert

# First we add a vertex in the middle of the basal edge.
basal_edge_index = face_1_basal_edges[0]
add_vert(sheet, basal_edge_index)
geom.update_all(sheet)
# Plot the diagram with vertex labelling
fig, ax= sheet_view(sheet)
for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)

# We only care about newedge starts from vert 6, since it's a basal edge.
new_edge = sheet.edge_df[(sheet.edge_df['srce'] == 6)]
new_edge_index = new_edge.index[0]
sheet.edge_df.loc[new_edge_index,]

















"""
This is the end of the script.
"""
