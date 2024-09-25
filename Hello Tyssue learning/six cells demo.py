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
from my_headers import delete_face, xprod_2d

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


# Draw with vertex labelling.
fig, ax= sheet_view(sheet, edge = {'head_width':0.1})
for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)

# Split cell 1 again
new = face_division(sheet = sheet, mother=1, vert_a = 11, vert_b = 10)
print(f'The newly added edge has number: {new}.')
geom.update_all(sheet)
# Draw the cell mesh with face labelling and edge arrows.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
sheet.edge_df.loc[7]


""" Now, we tweak the position of the two vertices on the spliting edge. """
# Reset the cell sheet.
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

add_vert(sheet, sheet.edge_df.index[-1])
geom.update_all(sheet)

# add another vertex on the new edge.
add_vert(sheet, sheet.edge_df.index[-1])
geom.update_all(sheet)
fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)

# Draw with vertex labelling.
fig, ax= sheet_view(sheet, edge = {'head_width':0.1})
for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)

print(sheet.vert_df.loc[[22,23],:])
# I need to add 0.1 and minus 0.1 on the x-value of the two vertices respectively.
sheet.vert_df.loc[22,'x']=2.1
sheet.vert_df.loc[23,'x']=1.9
geom.update_all(sheet)
fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)

# Do energy minimization.
res = solver.find_energy_min(sheet, geom, smodel)

fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)



""" vertex-vertex division, connect vertex 8 and 11 """






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




""" Try edge-edge algorithm """
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

""" Draw from basal to centre. """
# Compute the basal boundary edge.
sheet.get_opposite()
condition = sheet.edge_df.loc[:,'face'] == 1
edge_in_cell = sheet.edge_df[condition]
basal_edge_index = edge_in_cell[ edge_in_cell.loc[:,'opposite']==-1 ].index[0]
#get the vertex index of the newly added mid point.
basal_mid = add_vert(sheet, edge = basal_edge_index)[0]
print(basal_mid)
geom.update_all(sheet)

# Draw with vertex labelling.
fig, ax= sheet_view(sheet, edge = {'head_width':0.1})
for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)


# Without need to set centroid attempt.

condition = sheet.edge_df.loc[:,'face'] == 1
edge_in_cell = sheet.edge_df[condition]
# We use the notation: line = P0 + dt, where P0 is the offset point and d is
# the direction vector, t is the lambda variable.
condition = edge_in_cell.loc[:,'srce'] == basal_mid
# extract the x-coordiante from array, then convert to a float type.
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
print(f'The intersection has coordinates: {intersection}')

# Add the intersection point.
new_index = len(sheet.vert_df)
# Note the order is y,x coordinate in the data frame.
sheet.vert_df.loc[new_index] = [intersection[1], 1, intersection[0]]

#draw the line without the coordiante of centroid.
face_division(sheet, mother = 1, vert_a = basal_mid, vert_b = new_index)
geom.update_all(sheet)
        
fig, ax= sheet_view(sheet)
for edge, data in edge_in_cell.iterrows():
    ax.text((data.sx+data.tx)/2, (data.sy+data.ty)/2, edge)
        
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)

# draw the line from mid of basal edge to centroid.

# Compute the centroid coordinates.
condition = sheet.edge_df.loc[:,'face'] == 1
edge_in_cell = sheet.edge_df[condition]
cx = edge_in_cell.iloc[1]['fx']
cy = edge_in_cell.iloc[1]['fy']
# add the cx and cy as a new row into vert_df
ct_index = len(sheet.vert_df)
sheet.vert_df.loc[ct_index] = [cy, 1, cx]
centroid = [cx,cy]
edge_in_cell.iloc[1,]

centroid
face_division(sheet, mother = 1 , vert_a = basal_mid , vert_b = ct_index)


geom.update_all(sheet)
sheet_view(sheet)
sheet.face_df

""" Draw from centre to the opposite """





"""
This is the end of the script.
"""
