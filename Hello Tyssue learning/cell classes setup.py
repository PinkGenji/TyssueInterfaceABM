# -*- coding: utf-8 -*-
"""
This script aims to do two major tasks:
    (1) set up the cell classes we need in the model.
    (2) Set up a model where there is an initial layer of STB and mature CT,
    demonstrate that we can swap CT from the mature CT group to the G2 (growing for mitosis)
    with some probability p.
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

# 2D plotting
from tyssue.draw import sheet_view, highlight_cells
from tyssue.draw.plt_draw import plot_forces
from tyssue.config.draw import sheet_spec
# import my own functions
from my_headers import *

rng = np.random.default_rng(70)    # Seed the random number generator.

# Generate the cell sheet as three cells.
num_x = 5
num_y = 2
sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = num_x, ny = num_y, distx = 1, disty = 1)
geom.update_all(sheet)

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())
delete_face(sheet, 5)
delete_face(sheet, 6)

sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise
geom.update_all(sheet)

# Plot the figure to see the index.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)

# Add a new attribute to the face_df, called "cell class"
sheet.face_df['cell_class'] = 'default'
for i in sheet.face_df.index:
    if i in [0,1,2,3,4]:
        sheet.face_df.loc[i,'cell_class'] = "S" # Set them to be mature CT at start.
    else:
        sheet.face_df.loc[i,'cell_class'] = "STB"











"""
This is the end of the script.
"""
