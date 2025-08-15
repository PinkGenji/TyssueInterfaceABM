
# Load all required modules.
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from tyssue import Sheet
from tyssue.topology.sheet_topology import remove_face
from tyssue import PlanarGeometry as geom #for simple 2d geometry
from tyssue.dynamics import effectors, model_factory
from tyssue.io import hdf5 # For saving the datasets
from tyssue.collisions.intersection import self_intersections
# 2D plotting
from tyssue.draw import sheet_view
from tyssue.topology.sheet_topology import cell_division
from tyssue.config.draw import sheet_spec
from tyssue.draw.plt_draw import plot_forces

# Generate the initial cell sheet for bilayer
print('\n Now we change the initial geometry to bilayer.')
num_x = 16
num_y = 4

sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = num_x, ny = num_y, distx = 1, disty = 1)
geom.update_all(sheet)
#Updates the sheet geometry by updating: * the edge vector coordinates * the edge lengths * the face centroids
# * the normals to each edge associated face * the face areas.

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

def drop_face(sheet, face, **kwargs):
    """
    Removes the face indexed by "face" and all associated edges
    """
    edge = sheet.edge_df.loc[(sheet.edge_df['face'] == face)].index
    print(f"Removing face '{face}'")
    sheet.remove(edge, **kwargs)

# Repeatedly remove all non-hexagonal faces until none remain
while np.any(sheet.face_df['num_sides'].values != 6):
    bad_face = sheet.face_df[sheet.face_df['num_sides'] != 6].index[0]
    drop_face(sheet, bad_face)

geom.update_all(sheet)
sheet.get_opposite()

# Plot the figure to see the initial setup is what we want.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
plt.show()
ax.set_title("Initial Bilayer Setup")  # Adding title

print('Initial geometry plot generated. \n')

sheet.vert_df.loc[6,'x'] += 0.5
sheet.vert_df.loc[6,'y'] -= 0.5
geom.update_all(sheet)
# Plot the figure to see the initial setup is what we want.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
plt.show()





from tyssue.collisions.solvers import solve_sheet_collisions
position_buffer = sheet.vert_df[['x', 'y']].copy()

# Fix 2D collisions
changes_made = solve_sheet_collisions(sheet, position_buffer)

if changes_made:
    print("Collisions resolved.")
else:
    print("No correction needed.")




print('\n This is the end of this script. (＾• ω •＾) ')
