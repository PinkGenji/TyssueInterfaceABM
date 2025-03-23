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
from tyssue.draw.plt_draw import plot_forces
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
# print the cell area and visualize the cell.
CA_initial = sheet.face_df.loc[0,'area']
print(f'Initial Cell Area is: {CA_initial}')
fig, ax = sheet_view(sheet,  mode = '2D')


# Setup the solver.
solver = QSSolver()
nondim_specs = config.dynamics.quasistatic_plane_spec()
dim_model_specs = model.dimensionalize(nondim_specs)
sheet.update_specs(dim_model_specs, reset=True)

res = solver.find_energy_min(sheet, geom, model)

print(sheet.face_df.loc[0,'area'])

fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)


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
