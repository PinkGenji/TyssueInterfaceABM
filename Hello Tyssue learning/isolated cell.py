# -*- coding: utf-8 -*-
"""
Create an isolated cell the see if the area stops a stable state.
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
from tyssue.draw.plt_draw import plot_forces, plot_forces2
from tyssue.config.draw import sheet_spec
# import my own functions
from my_headers import *

rng = np.random.default_rng(70)

# Generate the cell sheet as three cells.
num_x = 1
num_y = 1
sheet = Sheet.planar_sheet_2d('face', nx = num_x, ny=num_y, distx=0.5, disty=0.5)
geom.update_all(sheet)
# remove non-enclosed faces
sheet.remove(sheet.get_invalid())  
delete_face(sheet, 1)
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise
sheet_view(sheet)
sheet.get_extra_indices()
# We need to creata a new colum to store the cell cycle time, default a 0, then minus.
sheet.face_df['T_cycle'] = 0
# Visualize the sheet.
fig, ax = sheet_view(sheet,  mode = '2D')
# First, we need a way to compute the energy, then use gradient descent.
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
       'perimeter': 3.0,
       'contractility': 0,
       'is_alive': 1,
       'prefered_area': 1},
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
sheet.update_specs(specs, reset = True)
geom.update_all(sheet)

# Adjust for cell-boundary adhesion force.
for i in sheet.edge_df.index:
    if sheet.edge_df.loc[i, 'opposite'] == -1:
        sheet.edge_df.loc[i, 'line_tension'] *=2
    else:
        continue
geom.update_all(sheet)

# We need set the all the threshold value first.
t1_threshold = 0.1
t2_threshold = 0.1
max_movement = t1_threshold/2

move = time_step_bot(sheet, dt=0.01, max_dist_allowed = max_movement )[1]

fig, ax = plot_forces2(sheet, geom, smodel, movement=move, coords=['x', 'y'], dt=0.01, scaling=0.1)

sheet_view(sheet)
sheet.face_df.loc[:,'area']


# Now assume we want to go from t = 0 to t= 0.2, dt = 0.1
t0 = 0
t_end = 4
dt = 0.01
time_points = np.linspace(t0, t_end, int((t_end - t0) / dt) + 1)



from tyssue.dynamics.base_gradients import length_grad
from tyssue.dynamics.planar_gradients import area_grad
area_grad(sheet)

for t in time_points:
    #print(f'start at t= {round(t, 5)}.')

     # Force computing and updating positions.
    valid_active_verts = sheet.active_verts[sheet.active_verts.isin(sheet.vert_df.index)]
    pos = sheet.vert_df.loc[valid_active_verts, sheet.coords].values
    # get the movement of position based on dynamical dt.
    dt, movement = time_step_bot(sheet, dt, max_dist_allowed = max_movement )
    new_pos = pos + movement
     # Save the new positions back to `vert_df`
    sheet.vert_df.loc[valid_active_verts , sheet.coords] = new_pos
    geom.update_all(sheet)
plot_forces2(sheet, geom, smodel, movement=movement, coords=['x', 'y'], dt=0.01, scaling=0.1)

sheet_view(sheet)
area = sheet.face_df.loc[:,'area'][0]
print(f'After evolve {t} time, area is: {area}')




# =============================================================================
# def to_nd(df, ndim):
#     """
#     Give a new shape to an input data by duplicating its column.
# 
#     Parameters
#     ----------
# 
#     df : input data that will be reshape
#     ndim : dimension of the new reshape data.
# 
#     Returns
#     -------
# 
#     df_nd : return array reshaped in ndim.
# 
#     """
#     df_nd = np.asarray(df).reshape((df.size, 1))
#     return df_nd
# 
# gamma_ = sheet.face_df.eval("contractility * perimeter * is_alive")
# gamma = sheet.upcast_face(gamma_)
# 
# grad_srce = -sheet.edge_df[sheet.ucoords] * to_nd(gamma, len(sheet.coords))
# grad_srce.columns = ["g" + u for u in sheet.coords]
# grad_trgt = -grad_srce
# print(grad_srce, grad_trgt)
# =============================================================================

"""
This is the end of the script.
"""
