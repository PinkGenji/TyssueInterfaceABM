"""
This script imports the T2Swap behaviour function from Tyssue, and runs with Euler solver
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
from tyssue.dynamics import effectors, model_factory
from tyssue.behaviors import EventManager
from tyssue.solvers.viscous import EulerSolver
from tyssue.behaviors.sheet.basic_events import T2Swap

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

history = History(sheet)
print('The parameters for energy function is loaded, and history object is created. \n')
manager = EventManager('face')
solver = EulerSolver(sheet, geom, model, manager=manager)
print('Solver starts, please wait ...')
solver.solve(tf=5, dt=0.001)
geom.update_all(sheet)
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face, fontsize=10, color="r")
plt.show()
min_area = sheet.face_df['area'].min()
max_area = sheet.face_df['area'].max()
mean_area = sheet.face_df['area'].mean()
print('Solver completed, from t=0 to t=5 with dt = 0.001: \n')
print(f'Minimum area is {min_area}; maximum area is {max_area}; mean area is {mean_area} \n')

# Now, re-initialise the sheet
sheet = history.retrieve(0)
print('System reinitialized.')
print('Setting the T2 threshold arbitraily to be 0.001 larger than minimum area evaluated before. \n')
t2_threshold = min_area+ 0.001
manager = EventManager('face')

# Append the T2swap for all cells to the event manager.
for i in sheet.face_df.index:
    stable_id = sheet.face_df.loc[i, 'id']
    manager.append(T2Swap, face_id = stable_id, crit_area=t2_threshold)
manager.update()
solver = EulerSolver(sheet, geom, model, manager=manager)
print('Solver starts, please wait ...')
solver.solve(tf=5, dt=0.001)
geom.update_all(sheet)
fig, ax = sheet_view(sheet)
ax.set_title('After solver with T2Swap')
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face, fontsize=10, color="r")
plt.show()
print('Solver completed, plot of what the current system looks like is generated.')


print('\n This is the end of this script. (＾• ω •＾) ')
