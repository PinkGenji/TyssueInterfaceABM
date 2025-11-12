"""
This script construct a model with following features:
1. A 2D bilayer sheet (no multiple CT layers).
2. A multi-class system assigned to cells.
3. Cells in CT class can proliferate.
"""

# Load all required modules.
import random
import numpy as np
import os
import sys
import re
import matplotlib.pyplot as plt
from tyssue import Sheet, PlanarGeometry
from tyssue.dynamics import effectors, model_factory
from tyssue.topology.base_topology import drop_face
from tyssue.behaviors import EventManager
from tyssue.behaviors.sheet.cell_class_events import cell_cycle_transition
from tyssue.solvers.viscous import EulerSolver
from tyssue.draw import sheet_view

# Set random seeds.
random.seed(42)  # Controls Python's random module (e.g. event shuffling)
np.random.seed(42)  # Controls NumPy's RNG (e.g. vertex positions, topology)

rng = np.random.default_rng(70)  # Seed the random number generator for my own division function.

# Generate the initial cell sheet for bilayer.
geom = PlanarGeometry
print('\n Now we change the initial geometry to bilayer.')
num_x = 16
num_y = 4

sheet = Sheet.planar_sheet_2d(identifier='bilayer', nx=num_x, ny=num_y, distx=1, disty=1)
geom.update_all(sheet)
# Updates the sheet geometry by updating: * the edge vector coordinates * the edge lengths * the face centroids
# * the normals to each edge associated face * the face areas.

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Repeatedly remove all non-hexagonal faces until none remain
while np.any(sheet.face_df['num_sides'].values != 6):
    bad_face = sheet.face_df[sheet.face_df['num_sides'] != 6].index[0]
    drop_face(sheet, bad_face)
# Update the geometry and computes the grabs all pairs of half-edges.
geom.update_all(sheet)
sheet.get_opposite()

# Plot the figure to see the initial setup is what we want.
fig, ax = sheet_view(sheet)
ax.set_title("Initial Bilayer Setup")  # Adding title
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face, fontsize=10, color="r")
plt.show()
print('Initial geometry plot generated. \n')

# Add a new attribute to the face_df, called "cell class"
sheet.face_df['cell_class'] = 'default'
sheet.face_df['timer'] = 'NA'
total_cell_num = len(sheet.face_df)
print('New attributes: cell_class; timer created for all cells. \n ')

for i in range(0,num_x-2):  # These are the indices of the bottom layer.
    sheet.face_df.loc[i,'cell_class'] = 'G1'
    # Add a timer for each cell enters "G1".
    sheet.face_df.loc[i, 'timer'] = 0.11

for i in range(num_x-2,len(sheet.face_df)):     # These are the indices of the top layer.
    sheet.face_df.loc[i,'cell_class'] = 'STB'
    # Add a timer for each cell enters "G1".
    sheet.face_df.loc[i, 'timer'] = np.nan

print(f'There are {total_cell_num} total cells; equally split into "G1" and "STB" classes. ')

# Load the effectors, then explicitly define what parameters are using in the simulation.
model = model_factory([
    effectors.LineTension,
    effectors.FaceContractility,
    effectors.FaceAreaElasticity
])
sheet.vert_df['viscosity'] = 1.0
specs = {
    'edge': {
        'is_active': 1,
        'line_tension': 0,
        'ux': 0.0,
        'uy': 0.0,
        'uz': 0.0
    },
   'face': {
       'area_elasticity': 1.0,
       'contractility': 0,
       'is_alive': 1,
       'prefered_area': 0.5},
   'settings': {
       'grad_norm_factor': 1.0,
       'nrj_norm_factor': 1.0
   },
   'vert': {
       'is_active': 1
   }
}
# Update the dynamic specs
sheet.update_specs(specs, reset=True)

# Initialise the event manager
time_step = 0.001
manager = EventManager('face')
manager.append(cell_cycle_transition, dt = time_step)   # using the default duration values
solver = EulerSolver(sheet, geom, model, manager=manager)
print('solver is set up. \n')
print('solver starts ...')
solver.solve(tf=3, dt=time_step)
geom.update_all(sheet)
fig, ax = sheet_view(sheet)
ax.set_title('Solver completed')
plt.show()
print('Solver completed, plot of what the current system looks like is generated.')


print('\n This is the end of this script. (＾• ω •＾) ')
