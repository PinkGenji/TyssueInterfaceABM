"""
This script produces a bilayer geometry first, then converts all edges between STB units into dummy edges.
A video of evolution is created in the end.
"""

# Load all required modules.
import numpy as np
import matplotlib.pyplot as plt
from tyssue import Sheet
from tyssue.topology.sheet_topology import remove_face
from tyssue import PlanarGeometry as geom #for simple 2d geometry
from tyssue.dynamics import effectors, model_factory

# 2D plotting
from tyssue.draw import sheet_view
from tyssue.topology.sheet_topology import cell_division

# import my own functions
from my_headers import *
from T3_function import *

import os
import imageio.v2 as imageio


rng = np.random.default_rng(70)    # Seed the random number generator.

# Generate the initial cell sheet for bilayer.
print('\n Now we change the initial geometry to bilayer.')
num_x = 16
num_y = 4

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

# Plot the figure to see the initial setup is what we want.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
plt.show()
ax.set_title("Initial Bilayer Setup")  # Adding title

print('Initial geometry plot generated. \n')


# Add dynamics to the model.
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
print('component of dynamics is added.')

# Add a new attribute to the face_df, called "cell class"
sheet.face_df['cell_class'] = 'default'
sheet.face_df['timer'] = 'NA'
total_cell_num = len(sheet.face_df)

print('New attributes: cell_class; timer created for all cells. \n ')
for i in range(0,num_x-2):  # These are the indices of bottom layer.
    sheet.face_df.loc[i,'cell_class'] = 'S'

for i in range(num_x-2,len(sheet.face_df)):     # These are the indices of top layer.
    sheet.face_df.loc[i,'cell_class'] = 'STB'
geom.update_all(sheet)
print(f'There are {total_cell_num} total cells; equally split into "S" and "STB" classes. ')

# Next I need to disable the edges between STB.
# If a half edge belongs to STB and its opposite half edge also belongs to STB, then we disable both half edges.


for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'opposite'] != -1:
        associated_cell = sheet.edge_df.loc[i,'face']
        opposite_edge = sheet.edge_df.loc[i,'opposite']
        opposite_cell = sheet.edge_df.loc[opposite_edge,'face']
        if sheet.face_df.loc[associated_cell,'cell_class'] == 'STB' and sheet.face_df.loc[opposite_cell,'cell_class'] == 'STB':
            sheet.edge_df.loc[i,'is_active'] = 0
            sheet.edge_df.loc[opposite_edge,'is_active'] = 0
            print(f'Deactivated the mutual edge between {associated_cell} and {opposite_cell}.')


# Now add the Euler solver to see how cells evolve.




print('\n This is the end of this script. (＾• ω •＾) ')
