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
import sys
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
from tyssue.draw import sheet_view
from tyssue.draw.plt_draw import plot_forces
from tyssue.config.draw import sheet_spec

# Set relative path, then import my own functions.
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model with multiple cell class'))
sys.path.append(model_path)
print("Model path:", model_path)
print("Files in directory:", os.listdir(model_path))
from my_headers import *

rng = np.random.default_rng(70)

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
# We need to creata a new colum to store the cell cycle time, default a 0, then minus.
sheet.face_df['T_cycle'] = 0
# print the cell area and visualize the cell.
CA_initial = sheet.face_df.loc[0,'area']
print(f'Initial Cell Area is: {CA_initial}')
fig, ax = sheet_view(sheet,  mode = '2D')
plt.show()


# Setup the solver.
solver = QSSolver()

# Specify the specs, just want to expand the cell to size of 1.
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
ax.set_title('After energy minimisation')
plt.show()

for i in sheet.face_df.index:
    sheet.face_df.loc[i,'prefered_area'] =1
    sheet.face_df.loc[i,'area_elasticity'] =500
    sheet.face_df.loc[i,'contractility'] = 0.0
    sheet.face_df.loc[i,'line_tension'] =0.0
    sheet.face_df.loc[i,'sub_area'] =0.0

# Check the tissue is at its equilibrium
res = solver.find_energy_min(sheet, geom, smodel)

# Print cell area after relaxation and plot the cell.
CA_relaxed = sheet.face_df.loc[0,'area']
print(f'Relaxed Cell Area is: {CA_relaxed}')
fig, ax = sheet_view(sheet, mode="2D")



""" The following part draw the plot of area_elasticity against cell area """

# Define specific area elasticity values to test
area_elasticities = [1, 5, 10, 100, 200, 300, 500]
resulting_areas = []

for elasticity in area_elasticities:
    # Generate the cell sheet with a single cell
    num_x, num_y = 1, 1
    sheet = Sheet.planar_sheet_2d('face', nx=num_x, ny=num_y, distx=0.5, disty=0.5)
    geom.update_all(sheet)
    
    # Remove non-enclosed faces
    sheet.remove(sheet.get_invalid())  
    delete_face(sheet, 1)
    sheet.reset_index(order=True)

    # Setup solver and update specs **before** setting elasticity
    solver = QSSolver()
    nondim_specs = config.dynamics.quasistatic_plane_spec()
    dim_model_specs = model.dimensionalize(nondim_specs)
    sheet.update_specs(dim_model_specs, reset=True)

    # Manually override area elasticity after specs update
    sheet.face_df['prefered_area'] = 1
    sheet.face_df['area_elasticity'] = elasticity  # Must be set after update_specs
    sheet.face_df['contractility'] = 0.0
    sheet.face_df['line_tension'] = 0.0
    sheet.face_df['sub_area'] = 0.0

    # Solve for equilibrium
    res = solver.find_energy_min(sheet, geom, model)
    
    # Store the resulting area
    final_area = sheet.face_df.loc[0, 'area']
    resulting_areas.append(final_area)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(area_elasticities, resulting_areas, marker='o', linestyle='-')
plt.xlabel("Area elasticity value ")
plt.ylabel("Cell area")
plt.title("Effect of Area Elasticity on Cell Area (with no contractility)")
plt.grid(True)
plt.show()






"""
This is the end of the script.
"""
