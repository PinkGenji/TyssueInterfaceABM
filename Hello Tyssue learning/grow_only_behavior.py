# -*- coding: utf-8 -*-
"""
This code contains grow-only behaviour function.
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
from tyssue.topology.base_topology import collapse_edge, remove_face
from tyssue.topology.sheet_topology import split_vert as sheet_split
from tyssue.topology.bulk_topology import split_vert as bulk_split
from tyssue.topology import condition_4i, condition_4ii

## model and solver
from tyssue.dynamics.planar_vertex_model import PlanarModel as smodel
from tyssue.solvers.quasistatic import QSSolver
from tyssue.generation import extrude
from tyssue.dynamics import model_factory, effectors
from tyssue.topology.sheet_topology import remove_face, cell_division

# Event manager
from tyssue.behaviors import EventManager

# 2D plotting
from tyssue.draw import sheet_view, highlight_cells

from my_headers import delete_face, xprod_2d, put_vert


""" start the project """
# Generate the cell sheet as three cells.
num_x = 9
num_y = 8

sheet = Sheet.planar_sheet_2d('face', nx = num_x, ny=num_y, distx=2, disty=2)

geom.update_all(sheet)

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Plot the figure to see the index.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
    
delete_face(sheet, num_x)
delete_face(sheet, num_x+1)
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise

# Plot figures to check.
# Draw the cell mesh with face labelling and edge arrows.
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
    
""" Assign the cell type to face_df and edge_df """
num_ct = num_x
# First we assign cell type in face data frame.
sheet.face_df.loc[0:num_ct-1, "cell_type"] = 'CT'
sheet.face_df.loc[num_ct:, "cell_type"] = 'ST'

# Then update the edge data frame via 'face' column.

sheet.edge_df['cell_type'] = 'to be set'

for i in list(range(len(sheet.edge_df))):
    sheet.edge_df.loc[i, 'cell_type'] = sheet.face_df.loc[sheet.edge_df.loc[i,'face'],'cell_type']


            
# add mechanical properties.
specs = {
    'edge': {
        'is_active': 1,
        'line_tension': 0.12,
        'ux': 0.0,
        'uy': 0.0,
        'uz': 0.0
    },
   'face': {
       'area_elasticity': 1.0,
       'contractility': 0.04,
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
sheet.update_specs(specs, reset = True)
geom.update_all(sheet)

# Minimize the potential engery
solver = QSSolver()
res = solver.find_energy_min(sheet, geom, smodel)

# Visualize the sheet.
fig, ax = sheet_view(sheet,  mode = '2D')


def grow_only(sheet, manager, cell_id, growth_speed):
    """
        
    
    Parameters
    ----------
    sheet: a :class:`Sheet` object
    cell_id: int
        the index of the dividing cell 
    growth_speed: float
        increase in the area per unit time
        A_0(t + dt) = A0(t) + growth_speed
    """

    # if the cell area is larger than the crit_area, we let the cell divide.
    if sheet.face_df.loc[cell_id, "cell_type"] == 'CT':
        sheet.face_df.loc[cell_id, "prefered_area"] += growth_speed
    else:
        pass

# Initialisation of manager 
manager = EventManager('face')

from tyssue import History

t= 0
stop = 1

# initialise the History object.
sim_recorder = History(sheet)

while manager.current and t < stop:
	# Execute the event in the current list.
    for i in sheet.face_df.index:
        print(f'we are at time step {t}, cell {i} is being checked.')
        manager.append(grow_only, cell_id = i, growth_speed = 0.5)
        manager.execute(sheet)
        # Find energy min.
        res = solver.find_energy_min(sheet, geom, smodel)
    # Record the step.
        sim_recorder.record()
	# Switch event list from the next list to the current list.
        manager.update()

    t += 1

# Visualisation of the tissue
fig, ax = sheet_view(sheet, mode="2D")

from IPython import display
from tyssue.draw import (
    sheet_view,
    highlight_faces,
    create_gif,
    browse_history
)






""" 
This is the end of the script.
"""
