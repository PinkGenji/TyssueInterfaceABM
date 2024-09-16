# -*- coding: utf-8 -*-
"""
This script uses a four cell system as a demo.
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
from my_headers import delete_face

""" start the project """
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

# Draw with edge labelling.
fig, ax= sheet_view(sheet)
for edge, data in sheet.edge_df.iterrows():
    ax.text((data.sx+data.tx)/2, (data.sy+data.ty)/2, edge)


""" Assign the cell type to the edges in dataframe. """

# Create a list (as the entries) that we want to add into the dataframe.
default_entry = []
for i in list(range(len(sheet.edge_df))):
	default_entry.append("To be set")

# adding another attribtue called 'cell_type' in the df, and fill the new column.
sheet.edge_df = sheet.edge_df.assign(cell_type = default_entry)
print(sheet.edge_df.head())
print(sheet.edge_df.keys())

# If the edge associated with face 0 and 1,  cell_type = 'CT', OR, cell_type = 'ST'
for i in list(range(len(sheet.edge_df))):
    if sheet.edge_df.loc[i,'face'] <3:
        sheet.edge_df.loc[i,'cell_type'] = 'CT'
    else: 
        sheet.edge_df.loc[i,'cell_type'] = 'ST'
    
print(sheet.edge_df)

""" Develope a algorithm splits a cell approximately in half laterally. """

# First we check the edge faces within the cell
# pd.set_option('display.max_columns', 7)
face1_edges = sheet.edge_df[sheet.edge_df['face'] == 1]
print(face1_edges)
# Get the vertex index with minimum y-value.
vert_1 = face1_edges.loc[face1_edges['sy'].idxmin()]['srce']
# Since the df is clockwisely ordered, we get the opposite vertex index.
vert_2 = face1_edges.loc[face1_edges['sy'].idxmin()+3]['srce']
#divide the face with the two vert.
new = face_division(sheet = sheet, mother=1, vert_a = vert_1, vert_b = vert_2)
print(f'The newly added edge has number: {new}.')
geom.update_all(sheet)
# Draw the cell mesh with face labelling and edge arrows.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)

print(sheet.edge_df.loc[:,['face','cell_type']])

""" Add a vertex in the middle of split edge and another one in the bot half. """
print(sheet.edge_df.loc[sheet.edge_df.index[-1],])
add_vert(sheet, sheet.edge_df.index[-1])
geom.update_all(sheet)
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)

# add another vertex on the new edge.
add_vert(sheet, sheet.edge_df.index[-1])
geom.update_all(sheet)

# Plot the geometry
from tyssue.config.draw import sheet_spec
draw_specs = sheet_spec()
sheet.vert_df['rand'] = np.linspace(0.0, 1.0, num=sheet.vert_df.shape[0])
cmap = plt.cm.get_cmap('viridis')
color_cmap = cmap(sheet.vert_df.rand)
draw_specs['vert']['visible'] = True

draw_specs['vert']['color'] = color_cmap
draw_specs['vert']['alpha'] = 0.5
draw_specs['vert']['s'] = 50
fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)

# Perform energy minimization
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

fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)

""" Now, we tweak the position of the two vertices on the spliting edge. """






""" The daugther cells grow. Trial by 1-step adding area value. """
sheet.face_df.loc[6] *= 1.5
geom.update_all(sheet)
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
# Not working.






""" cell fusion, starts with a removal of a shared edge. """
sheet.get_opposite() #computes the 'twin' half edge.
sheet.edge_df
face2_edges = sheet.edge_df[sheet.edge_df['face'] == 2]
face2_edges.index[face2_edges['opposite' ]!=-1][0]
# Draw with edge labelling.
fig, ax= sheet_view(sheet, edge = {'head_width':0.1})
for edge, data in face2_edges.iterrows():
    ax.text((data.sx+data.tx)/2, (data.sy+data.ty)/2, edge)






""" Add mechanical properties for energy minimization """
# We are using the specs from 2007 Farhadifar model.
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



# Select all edges within cell 3.
sheet.edge_df[(sheet.edge_df['face'] == 3)]
# Deactivate the edges within cell 3.
sheet.edge_df.loc[sheet.edge_df[(sheet.edge_df['face'] == 3)].index,'is_active'] = 0
sheet.edge_df.loc[sheet.edge_df[(sheet.edge_df['face'] == 4)].index,'is_active'] = 0
sheet.edge_df.loc[sheet.edge_df[(sheet.edge_df['face'] == 5)].index,'is_active'] = 0
sheet.edge_df.loc[sheet.edge_df[(sheet.edge_df['face'] == 0)].index,'is_active'] = 0
sheet.edge_df.loc[sheet.edge_df[(sheet.edge_df['face'] == 2)].index,'is_active'] = 0


print(sheet.edge_df)

# energy minimisation.
solver = QSSolver()
res = solver.find_energy_min(sheet, geom, smodel)

sheet_view(sheet)   # Draw cell mesh.

print(sheet.face_df)



#plot forces
from tyssue.draw.plt_draw import plot_forces
fig, ax = plot_forces(sheet, geom, smodel, ['x', 'y'], scaling=0.1)
fig.set_size_inches(10, 12)

# Draw again with face labelling.
fig, ax= sheet_view(sheet)
for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)
#plot forces
from tyssue.draw.plt_draw import plot_forces
fig, ax = plot_forces(sheet, geom, smodel, ['x', 'y'], scaling=0.1)
fig.set_size_inches(10, 12)














"""
This is the end of the script.
"""
