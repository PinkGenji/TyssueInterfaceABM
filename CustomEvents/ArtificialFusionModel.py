"""
This script articially select a CT cell to fuse with the STB layer.
This is used to demonstrate implementation of fusion mechanism, without details of biological triggers.
"""
# Load all required modules.

import numpy as np
import pandas as pd
import sys
import os
import json
import matplotlib as matplot
import matplotlib.pylab as plt
import ipyvolume as ipv

from tyssue import Sheet, config, History, PlanarGeometry
from tyssue import PlanarGeometry as geom #for simple 2d geometry

# For cell topology/configuration
from tyssue.topology.sheet_topology import type1_transition
from tyssue.topology.base_topology import drop_face

# model and solver
from tyssue.solvers.quasistatic import QSSolver
from tyssue.dynamics import model_factory, effectors
from tyssue.topology.sheet_topology import remove_face, cell_division, face_division
from tyssue.behaviors.sheet.basic_events import T1Swap, T2Swap
from tyssue.behaviors import EventManager
from tyssue.behaviors.sheet.cell_activity_events import fusion
from tyssue.solvers.viscous import EulerSolver

# 2D plotting
from tyssue.draw import sheet_view
from tyssue.config.draw import sheet_spec
from tyssue.draw.plt_draw import create_gif


# Set random seed for reproducibility
rng = np.random.default_rng(42)

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
sheet.get_extra_indices()

# Add a new attribute to the face_df, called "cell class"
sheet.face_df['cell_class'] = 'default'
sheet.face_df['timer'] = np.nan
print('New attributes: cell_class; timer created for all cells. \n ')

for i in range(0,num_x-2):  # These are the indices of the bottom layer.
    sheet.face_df.loc[i,'cell_class'] = 'G1'
    # Add a timer for each cell enters "G1".
    sheet.face_df.loc[i, 'timer'] = 0.11

for i in range(num_x-2,len(sheet.face_df)):     # These are the indices of the top layer.
    sheet.face_df.loc[i,'cell_class'] = 'STB'
    # Add a timer for each cell enters "G1".
    sheet.face_df.loc[i, 'timer'] = np.nan

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
        'line_tension': 10,
        'ux': 0.0,
        'uy': 0.0,
        'uz': 0.0
    },
   'face': {
       'area_elasticity': 110.0,
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
# Update the dynamic specs
sheet.update_specs(specs, reset=True)
# Adjust for cell-boundary adhesion force.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i, 'opposite'] == -1:
        sheet.edge_df.loc[i, 'line_tension'] *= 2
    else:
        continue
geom.update_all(sheet)
print('component of dynamics is added.')

# Plot the figure to see the initial setup is what we want.
draw_specs = sheet_spec()
# Enable face visibility.
draw_specs['face']['visible'] = True
for i in sheet.face_df.index:   # Assign face colour based on cell type.
    if sheet.face_df.loc[i,'cell_class'] == 'STB':
        sheet.face_df.loc[i,'color'] = 0.7
    else:
        sheet.face_df.loc[i,'color'] = 0.1
draw_specs['face']['color'] = sheet.face_df['color']
draw_specs['face']['alpha'] = 0.2   # Set transparency.

# Enable edge visibility
draw_specs['edge']['visible'] = True
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'is_active'] == 0:
        sheet.edge_df.loc[i,'width'] = 2
    else:
        sheet.edge_df.loc[i,'width'] = 0.5
draw_specs['edge']['width'] = sheet.edge_df['width']

fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
plt.show()
print('Initial geometry plot generated. \n')

# Let cells to grow to equilibrium first.
solver = QSSolver()
res = solver.find_energy_min(sheet, geom, model)
geom.update_all(sheet)

fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
ax.set_title('Detached a middle STB unit from the layer below')
plt.show()
history = History(sheet)

# Deactivate the edges between STB units.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'opposite'] != -1:
        associated_cell = sheet.edge_df.loc[i,'face']
        opposite_edge = sheet.edge_df.loc[i,'opposite']
        opposite_cell = sheet.edge_df.loc[opposite_edge,'face']
        if sheet.face_df.loc[associated_cell,'cell_class'] == 'STB' and sheet.face_df.loc[opposite_cell,'cell_class'] == 'STB':
            sheet.edge_df.loc[i,'is_active'] = 0
            sheet.edge_df.loc[opposite_edge,'is_active'] = 0

# Plot the figure to see the initial setup is what we want.
draw_specs = sheet_spec()
# Enable face visibility.
draw_specs['face']['visible'] = True
for i in sheet.face_df.index:   # Assign face colour based on cell type.
    if sheet.face_df.loc[i,'cell_class'] == 'STB':
        sheet.face_df.loc[i,'color'] = 0.7
    else:
        sheet.face_df.loc[i,'color'] = 0.1
draw_specs['face']['color'] = sheet.face_df['color']
draw_specs['face']['alpha'] = 0.2   # Set transparency.

# Enable edge visibility
draw_specs['edge']['visible'] = True
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i,'is_active'] == 0:
        sheet.edge_df.loc[i,'width'] = 2
    else:
        sheet.edge_df.loc[i,'width'] = 0.5
draw_specs['edge']['width'] = sheet.edge_df['width']

geom.update_all(sheet)
fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
ax.set_title('System at the steady state with disabled STB-STB edges')
plt.show()

# Set up the parameters for this simulation.
time_step = 0.001
my_t1 = sheet.edge_df['length'].mean()/10
t2_threshold = sheet.face_df['area'].mean()/10
uid = sheet.face_df.loc[6,'unique_id']

# Initialize the event manager with behaviour functions
manager = EventManager('face')
manager.append(fusion,geom=geom, drawing_spec = draw_specs, unique_id = uid)   # using the default duration values
manager.append(T1Swap, geom = geom, t1_threshold = my_t1, multiplier = 1.5)
manager.append(T2Swap, crit_area = t2_threshold)
manager.update()
solver = EulerSolver(sheet, geom, model, history=history, manager=manager)
print('solver is set up. \n')
print('solver starts ...')
solver.solve(tf=1, dt=time_step)
geom.update_all(sheet)
fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
ax.set_title('Solver completed')
plt.show()
print('Solver completed, plot of what the current system looks like is generated.')

create_gif(solver.history, "ArtificialFusionModel.gif", num_frames=150, draw_func=lambda s: sheet_view(s, ['x','y'], **draw_specs), margin= -1)

print('This is the end of this script. (＾• ω •＾) ')
