#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusion 101
"""


# =============================================================================
# First we need to surpress the version warnings from Pandas.
import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning) 
# =============================================================================

# Load all required modules.
import sys
import os
import numpy as np
import pandas as pd

import os

import matplotlib.pylab as plt


from tyssue import Sheet, config #import core object
from tyssue import PlanarGeometry as geom #for simple 2d geometry

# 2D plotting
from tyssue.draw import sheet_view

from tyssue.config.draw import sheet_spec
# import my own functions

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Model with multiple cell class'))
sys.path.append(model_path)
print("Model path:", model_path)
print("Files in directory:", os.listdir(model_path))

import my_headers as mh

rng = np.random.default_rng(70)

# Generate the cell sheet as three cells.
num_x = 4
num_y = 4
sheet = Sheet.planar_sheet_2d('face', nx = num_x, ny=num_y, distx=0.5, disty=0.5)
geom.update_all(sheet)
# remove non-enclosed faces
sheet.remove(sheet.get_invalid())  
for i in [2,3,4,5]:
    mh.delete_face(sheet, i)
geom.update_all(sheet)
sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise
fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
plt.show()
sheet.get_extra_indices()
# We need to creata a new colum to store the cell cycle time, default a 0, then minus.
# First we assign cell class in face data frame.
sheet.face_df.loc[0, "cell_class"] = 'CT'
sheet.face_df.loc[1, "cell_class"] = 'STB'

# Add dynamics to the model.
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

# Merge the STB units into one whole STB.
# Repeatedly merge STB–STB neighbors until none remain
new_cell = mh.cell_merge(sheet, 0,1,'STB')

# Finally, update geometry
geom.update_all(sheet)


fig, ax = sheet_view(sheet, edge = {'head_width':0.1})
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
plt.show()










""" This is the end of the script. """
