"""This script demonstrates how the event/behaviour function that controls cell class transition works"""

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
from tyssue.behaviors.sheet.bilayer_dummy_set import auto_dummy_edges, update_draw_specs, deactivate_cells
from tyssue.solvers.viscous import EulerSolver
from tyssue.solvers import QSSolver

# Plotting related
from tyssue.draw import sheet_view
from tyssue.config.draw import sheet_spec
from tyssue.draw.plt_draw import create_gif


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
print('Initial geometry plot generated. \n')

# Add a new attribute to the face_df, called "cell class"
sheet.face_df['cell_class'] = 'default'
sheet.face_df['timer'] = 'NA'
total_cell_num = len(sheet.face_df)
print('New attributes: cell_class; timer created for all cells. \n ')

for i in range(0,num_x-2):  # These are the indices of the bottom layer.
    # All CTs assigned with class ‘G1’, ‘S’, ‘M’, or ‘G2’ based on probabilities that reflect typical times in each stage of the cell cycle
    # Draw a random number between 0 and 1, it's G1 if  < 11/24, S if < 19/24, M if < 20/24, else, G2.
    random_num = rng.random()
    if random_num < 11/24:
        sheet.face_df.loc[i,'cell_class'] = 'G1'
        sheet.face_df.loc[i, 'timer'] = 8
    elif 11/24 <= random_num < 19/24:
        sheet.face_df.loc[i,'cell_class'] = 'S'
        sheet.face_df.loc[i, 'timer'] = 7
    elif 19/24 <= random_num < 20/24:
        sheet.face_df.loc[i,'cell_class'] = 'M'
        sheet.face_df.loc[i, 'timer'] = 0.5
    else:
        sheet.face_df.loc[i,'cell_class'] = 'G2'
        sheet.face_df.loc[i, 'timer'] = 3

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

# Use QS solver to start with the steady state of the system.
solver = QSSolver()
res = solver.find_energy_min(sheet, geom, model)
print("Successfull gradient descent? ", res['success'])

# Assign dummy edges
auto_dummy_edges(sheet)

# Deactivate the four cells at four corners, avoid them from energy minimisation.
cells_to_deactivate = [0, 13, 14, 27]
deactivate_cells(sheet, cells_to_deactivate)

# update the draw specs
draw_specs = update_draw_specs(sheet)
geom.update_all(sheet)

# Initialise the event manager
time_step = 0.001
manager = EventManager('face')
manager.append(cell_cycle_transition, dt = time_step)   # using the default duration values
solver = EulerSolver(sheet, geom, model, manager=manager)
print('solver is set up. \n')
print('solver starts ...')
solver.solve(tf=40, dt=time_step)
geom.update_all(sheet)
fig, ax = sheet_view(sheet)
ax.set_title('Solver completed \n')
plt.show()
create_gif(
    solver.history,
    "simulation.gif",
    num_frames=500,
    draw_func=lambda s: sheet_view(s, ['x','y'], **draw_specs),
    margin=-1
)

print('Solver completed, plot and gif of what the current system looks like is generated.')


print('\n This is the end of this script. (＾• ω •＾) ')
