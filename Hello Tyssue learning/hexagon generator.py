# -*- coding: utf-8 -*-
"""
This script does the boundary shape change.
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
from tyssue.topology.base_topology import collapse_edge, remove_face, merge_border_edges
from tyssue.topology.sheet_topology import split_vert as sheet_split
from tyssue.topology.bulk_topology import split_vert as bulk_split
from tyssue.topology import condition_4i, condition_4ii




# 2D plotting
from tyssue.draw import sheet_view, highlight_cells

#I/O
from tyssue.io import hdf5


# Generate a bilayer structure.
bilayer = Sheet.planar_sheet_2d(identifier='bilayer', nx = 10, ny = 4, distx = 1, disty = 1)

geom.update_all(bilayer)

# Have a look of the generated bilayer.
sheet_view(bilayer)
fig, ax = sheet_view(bilayer)
for face, data in bilayer.face_df.iterrows():
    ax.text(data.x, data.y, face)

from scipy.spatial import Voronoi, voronoi_plot_2d
from tyssue.generation import hexa_grid2d, from_2d_voronoi

nx = 10
ny=4
distx=5
disty = 2
noise = 0

grid = hexa_grid2d(nx, ny, distx, disty, noise)
datasets = from_2d_voronoi(Voronoi(grid))

vor = Voronoi(grid)
import matplotlib.pyplot as plt
fig = voronoi_plot_2d(vor)
plt.show()

vor.ridge_point


