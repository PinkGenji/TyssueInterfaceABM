"""
This script creates a trilayer geometry, then artificially detach an STB unit from the CTs.
The goal of this script is to investigate how we should remove an STB unit from its layer.
"""


# Load all required modules.
import numpy as np
import  re
import matplotlib.pyplot as plt
from tyssue import Sheet
from tyssue.topology.sheet_topology import remove_face
from tyssue import PlanarGeometry as geom #for simple 2d geometry
from tyssue.dynamics import effectors, model_factory
from tyssue.io import hdf5 # For saving the datasets

# 2D plotting
from tyssue.draw import sheet_view
from tyssue.topology.sheet_topology import cell_division
from tyssue.config.draw import sheet_spec


# import my own functions
from my_headers import *
from T3_function import *

import os


# Initialise the geometry first.

rng = np.random.default_rng(70)    # Seed the random number generator.

# Generate the initial cell sheet for tri-layer.
num_x = 16
num_y = 5

sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = num_x, ny = num_y, distx = 1, disty = 1)
geom.update_all(sheet)
#Updates the sheet geometry by updating: * the edge vector coordinates * the edge lengths * the face centroids
# * the normals to each edge associated face * the face areas.

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Delete the irregular polygons.
for i in sheet.face_df.index:
    if sheet.face_df.loc[i,'num_sides'] != 6:
        delete_face(sheet,i)
    else:
        continue

sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise
geom.update_all(sheet)
sheet.get_opposite()
# Add a new attribute to the face_df, called "cell class"
sheet.face_df['cell_class'] = 'default'
sheet.face_df['timer'] = 'NA'

for i in range(0,2*num_x-4):  # Looping over the bottom layer.
    sheet.face_df.loc[i,'cell_class'] = 'S'

for i in range(2*num_x-4,len(sheet.face_df)):     # These are the indices of the top layer.
    sheet.face_df.loc[i,'cell_class'] = 'STB'


# Add dynamics to the model, so we can draw based on edge status.
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
sheet.update_specs(specs, reset=True)
geom.update_all(sheet)

# Adjust for cell-boundary adhesion force.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i, 'opposite'] == -1:
        sheet.edge_df.loc[i, 'line_tension'] *= 2
    else:
        continue
geom.update_all(sheet)

# Commented out the following code to avoid fixed ends of the cells.
# Deactivate the cells on the leftmost and rightmost sides.
for cell_id in [0,13,14,27,28,41]:
    sheet.face_df.loc[cell_id,'is_alive'] = 0
    for i in sheet.edge_df[sheet.edge_df['face'] == cell_id]['srce'].tolist():
        sheet.vert_df.loc[i,'is_active'] = 0

# Deactivate the edges between STB units.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'opposite'] != -1:
        associated_cell = sheet.edge_df.loc[i,'face']
        opposite_edge = sheet.edge_df.loc[i,'opposite']
        opposite_cell = sheet.edge_df.loc[opposite_edge,'face']
        if sheet.face_df.loc[associated_cell,'cell_class'] == 'STB' and sheet.face_df.loc[opposite_cell,'cell_class'] == 'STB':
            sheet.edge_df.loc[i,'is_active'] = 0
            sheet.edge_df.loc[opposite_edge,'is_active'] = 0

draw_specs = sheet_spec()
# Enable face visibility.
draw_specs['face']['visible'] = True
for i in sheet.face_df.index:   # Assign face colour based on cell type.
    if sheet.face_df.loc[i,'cell_class'] == 'STB': sheet.face_df.loc[i,'color'] = 0.7
    else: sheet.face_df.loc[i,'color'] = 0.1
draw_specs['face']['color'] = sheet.face_df['color']
draw_specs['face']['alpha'] = 0.2   # Set transparency.

# Enable edge visibility
draw_specs['edge']['visible'] = True
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'is_active'] == 0: sheet.edge_df.loc[i,'width'] = 2
    else: sheet.edge_df.loc[i,'width'] = 0.5
draw_specs['edge']['width'] = sheet.edge_df['width']

fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
plt.show()
print('Initialised the geometry.')


# Detach the STB unit number 34 from the two cells beneath it.

sheet.get_extra_indices()
# find the edge number by checking the mutual edge between cell 19 and 34.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i, 'face'] == 19:
        opposite_edge = sheet.edge_df.loc[i, 'opposite']
        if opposite_edge != -1 and sheet.edge_df.loc[opposite_edge, 'face'] == 34:
            edge_to_process = i
            print(f'Edge {edge_to_process} is the edge we want to do T1 transition.')
            # Do a T1 transition on the edge we want, find the edge number first.
            type1_transition(sheet, edge_to_process, remove_tri_faces=False, multiplier=5)
            geom.update_all(sheet)
            sheet.reset_index(order=True)
    else:
        continue

# find the edge number by checking the mutual edge between cell 20 and 34.
sheet.get_extra_indices()
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i, 'face'] == 20:
        opposite_edge = sheet.edge_df.loc[i, 'opposite']
        if opposite_edge != -1 and sheet.edge_df.loc[opposite_edge, 'face'] == 34:
            edge_to_process = i
            print(f'Edge {edge_to_process} is the edge we want to do T1 transition.')
            # Do a T1 transition on the edge we want, find the edge number first.
            type1_transition(sheet, edge_to_process, remove_tri_faces=False, multiplier=5)
            geom.update_all(sheet)
            sheet.reset_index(order=True)
    else:
        continue


draw_specs = sheet_spec()
# Enable face visibility.
draw_specs['face']['visible'] = True
for i in sheet.face_df.index:   # Assign face colour based on cell type.
    if sheet.face_df.loc[i,'cell_class'] == 'STB': sheet.face_df.loc[i,'color'] = 0.7
    else: sheet.face_df.loc[i,'color'] = 0.1
draw_specs['face']['color'] = sheet.face_df['color']
draw_specs['face']['alpha'] = 0.2   # Set transparency.

# Enable edge visibility
draw_specs['edge']['visible'] = True
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'is_active'] == 0: sheet.edge_df.loc[i,'width'] = 2
    else: sheet.edge_df.loc[i,'width'] = 0.5
draw_specs['edge']['width'] = sheet.edge_df['width']

fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
plt.show()
print('STB unit 34 is detached.')


# Next, remove the STB unit number 34, then check if there are any unwanted vertices.



print('\n This is the end of this script. (＾• ω •＾) ')
