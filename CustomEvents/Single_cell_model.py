"""
This script setups a single cell model using the Tyssue framework for all kinds of test.
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

from tyssue import Sheet, config, History
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
from tyssue.behaviors import EventManager
from tyssue.behaviors.sheet.cell_activity_events import proliferation
from tyssue.solvers.viscous import EulerSolver


# 2D plotting
from tyssue.draw import sheet_view
from tyssue.config.draw import sheet_spec
from tyssue.draw.plt_draw import create_gif


rng = np.random.default_rng(42)

def drop_face(sheet, face, **kwargs):
    """
    Removes the face indexed by "face" and all associated edges
    """
    edge = sheet.edge_df.loc[(sheet.edge_df['face'] == face)].index
    print(f"Dropping face '{face}'")
    sheet.remove(edge, **kwargs)

# Generate the cell sheet as three cells.
num_x = 3
num_y = 3
sheet = Sheet.planar_sheet_2d('face', nx = num_x, ny=num_y, distx=0.5, disty=0.5)
geom.update_all(sheet)
# remove non-enclosed faces
sheet.remove(sheet.get_invalid())
drop_face(sheet, 1)
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise
fig, ax = sheet_view(sheet)
plt.show()
sheet.get_extra_indices()

# Setup the solver.
sheet.vert_df['viscosity'] = 1.0

solver = QSSolver()

# Specify the specs, just want to expand the cell to size of 1.
model = model_factory([
    effectors.LineTension,
    effectors.FaceContractility,
    effectors.FaceAreaElasticity
])
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
       'prefered_area': 1.0},
   'settings': {
       'grad_norm_factor': 1.0,
       'nrj_norm_factor': 1.0
   },
   'vert': {
       'is_active': 1
   }
}


# Update the specs (adds / changes the values in the dataframes' columns)
sheet.update_specs(specs)

res = solver.find_energy_min(sheet, geom, smodel)
geom.update_all(sheet)
print(sheet.face_df.loc[0,'area'])

fig, ax = sheet_view(sheet)
ax.set_title('Start at equilibrium')
plt.show()
history = History(sheet)

# Initialise the event manager
time_step = 0.001
manager = EventManager('face')
uid = sheet.face_df.loc[0,'unique_id']
manager.append(proliferation,geom=geom,unique_id = uid, crit_area = 2, growth_rate = 1, dt = time_step )   # using the default duration values
solver = EulerSolver(sheet, geom, model, history=history, manager=manager)
print('solver is set up. \n')
print('solver starts ...')
solver.solve(tf=3, dt=time_step)
geom.update_all(sheet)
fig, ax = sheet_view(sheet)
ax.set_title('Solver completed')
plt.show()
print('Solver completed, plot of what the current system looks like is generated.')

create_gif(solver.history, "isolated_cell_division.gif", num_frames=150, margin= -1)





print('This is the end of this script. (＾• ω •＾) ')
