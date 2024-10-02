# -*- coding: utf-8 -*-
"""
This script draws a cell evolution of 10 cells with lateral split.
"""


""" load modules """
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

# import my own functions
from my_headers import delete_face, xprod_2d, put_vert, lateral_split, lateral_division

""" Start programming. """
# Generate the cell sheet as three cells.
num_x = 10
num_y = 2
sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = num_x, ny = num_y, distx = 1, disty = 1)
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


""" Add another column for division status """
sheet.face_df['division_status'] = 'ready'

# Change the division_status for ST cells.
for i in list(range(len(sheet.face_df))):
    if sheet.face_df.loc[i, 'cell_type'] == 'ST':
        sheet.face_df.loc[i, 'division_status'] = 'N/A'
    else:
        continue

""" Add another column to store the growth speed. """
sheet.face_df['growth_speed'] = 'N/A'

''' Energy minimization '''
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
solver = QSSolver()
res = solver.find_energy_min(sheet, geom, smodel)
sheet_view(sheet) 

fig,ax = sheet_view(sheet)
ax.title.set_text('test')


sheet.face_df.loc[1,'growth_speed']


""" Modelling the tissue evolution """

from tyssue.behaviors import EventManager

# Initialisation of manager 
manager = EventManager("face")

from tyssue import History# The History object records all the time steps 
history = History(sheet)

# We assume one time step is 6 hours.
# At time t = 0, the cells are created.
# Then at each time step, 12/4 = 3% of CT cells laterally split.
# After spliting, the cells grow to preferred area within 5 time steps.

t = 0
stop = 8

while manager.current and t <= stop:
    for i in sheet.face_df.index:
        print(f'we are at time step {t}, cell {i} is being checked.')
        manager.append(lateral_division, cell_id = i, division_rate = 0.03)
        manager.execute(sheet)
        # Find energy min state and record.
        res = solver.find_energy_min(sheet, geom, smodel)
        history.record()
        # Switch event list from the next list to the current list
        manager.update()
    fig, ax = sheet_view(sheet)
    ax.title.set_text(f'Snapshot at t = {t}')
    t += 1












time_window = list(range(0,48,6))
for t in time_window:
    # Draw the plot at the starting of each time step.
    fig, ax = sheet_view(sheet)
    ax.title.set_text(f'Snapshot at t = {t}')
    
    # Do iteration for division check for all CTs.
    for i in list(range(len(sheet.face_df))):
        if sheet.face_df[sheet.face_df.loc[i,'cell_type']] == 'CT' and sheet.face_df.loc[i, 'division_status'] == 'ready':
            # A random float number is generated between (0,1)
            prob = np.random.uniform(0,1)
            # If the random number is less than 3%, then this cell divides.
            # Make sure change the division_status.
            if prob < 0.03:
                daughter = lateral_split(sheet, mother = i)
                sheet.face_df.loc[i,'growth_speed'] = (sheet.face_df.loc[i,'prefered_area'] - sheet.face_df.loc[i, 'area'])/5
                sheet.face_df.loc[i, 'division_status'] = 'growing'
                sheet.face_df.loc[daughter,'growth_speed'] = (sheet.face_df.loc[daughter,'prefered_area'] - sheet.face_df.loc[i, 'area'])/5
                sheet.face_df.loc[daughter, 'division_status'] = 'growing'
        
        elif sheet.face_df[sheet.face_df.loc[i,'cell_type']] == 'CT' and sheet.face_df.loc[i, 'division_status'] == 'growing':
            sheet.face_df.loc[i,'area'] = sheet.face_df.loc[i,'area'] + sheet.face_df.loc[i,'growth_speed']
            if sheet.face_df.loc[i,'area'] <= sheet.face_df.loc[i,'prefered_area']:
                sheet.face_df.loc[i, 'division_status'] = 'ready'
        
        else:
            continue







"""
This is the end of the script. 
"""
