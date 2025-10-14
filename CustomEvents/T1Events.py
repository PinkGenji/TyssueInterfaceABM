"""
This script imports the behaviour function of T1 transition and how it runs with Euler solver
"""

# Load all required modules.
import random
import numpy as np
import os
import sys
import re
import matplotlib.pyplot as plt
from tyssue import Sheet, History, PlanarGeometry
from tyssue.topology.base_topology import drop_face
from    tyssue.dynamics import effectors, model_factory
from tyssue.behaviors import EventManager
from tyssue.solvers.viscous import EulerSolver
from tyssue.behaviors.sheet.basic_events import T1Swap
# Plotting related
from tyssue.draw import sheet_view

# Generate the initial cell sheet for bilayer.
geom = PlanarGeometry
print('\n Now we change the initial geometry to bilayer.')
num_x = 16
num_y = 4
sheet = Sheet.planar_sheet_2d(identifier='bilayer', nx=num_x, ny=num_y, distx=1, disty=1)
geom.update_all(sheet)
# remove non-enclosed faces
sheet.remove(sheet.get_invalid())
# Repeatedly remove all non-hexagonal faces until none remain
while np.any(sheet.face_df['num_sides'].values != 6):
    bad_face = sheet.face_df[sheet.face_df['num_sides'] != 6].index[0]
    drop_face(sheet, bad_face)
# Update the geometry and computes the grabs all pairs of half-edges.
geom.update_all(sheet)
sheet.get_opposite()

# Make edge 23 and edge 41 to be shorter.
srce = sheet.edge_df.loc[23,'srce']
ux = sheet.edge_df.loc[23,'ux']
uy = sheet.edge_df.loc[23,'uy']
sheet.vert_df.loc[srce, 'x'] += 0.5*ux
sheet.vert_df.loc[srce, 'y'] += 0.5*uy
geom.update_all(sheet)

srce = sheet.edge_df.loc[41,'srce']
ux = sheet.edge_df.loc[41,'ux']
uy = sheet.edge_df.loc[41,'uy']
sheet.vert_df.loc[srce, 'x'] += 0.5*ux
sheet.vert_df.loc[srce, 'y'] += 0.5*uy
geom.update_all(sheet)

# Plot the figure to see the initial setup is what we want.
fig, ax = sheet_view(sheet)
ax.set_title("Initial Bilayer Setup")  # Adding title
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face, fontsize=10, color="r")
plt.show()
print('Initial geometry plot generated. \n')

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
# Update the specs (adds / changes the values in the dataframes' columns)
sheet.update_specs(specs, reset=True)
sheet.get_opposite()
geom.update_all(sheet)
# Record as the initial history of the sheet
history = History(sheet)
print('The parameters for energy function is loaded, and history object is created. \n')

manager = EventManager('face')
solver = EulerSolver(sheet, geom, model, manager=manager)
print('Solver starts, please wait ...')
solver.solve(tf=1, dt=0.001)
geom.update_all(sheet)
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face, fontsize=10, color="r")
ax.set_title('System without T1')
plt.show()
min_edge = sheet.edge_df['length'].min()
print('Setting the T1 threshold arbitrarily to be 110% of the minimum length.')
my_t1 = min_edge *1.1

# Now, re-initialise the sheet
sheet = history.retrieve(0)
print('System reinitialized.')
manager = EventManager('face')

# Append the T1swap for all cells to the event manager.
manager.append(T1Swap, t1_threshold = my_t1, multiplier = 1.5)
manager.update()
solver = EulerSolver(sheet, geom, model, manager=manager)
print('Solver starts, please wait ...')
solver.solve(tf=1, dt=0.001)
geom.update_all(sheet)
fig, ax = sheet_view(sheet)
ax.set_title('After solver with T1Swap')
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face, fontsize=10, color="r")
plt.show()
print('Solver completed, plot of what the current system looks like is generated.')

print('\n This is the end of this script. (＾• ω •＾) ')
