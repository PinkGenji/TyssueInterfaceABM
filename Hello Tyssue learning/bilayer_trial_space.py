# -*- coding: utf-8 -*-
"""
This script where I implement any trial actions onto the bilayer structure.
"""


# =============================================================================
# First we need to surpress the version warnings from Pandas.
import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning) 
# =============================================================================

# Load all required modules.

import pandas as pd
import numpy as np
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
from tyssue.dynamics.planar_vertex_model import PlanarModel as pmodel
from tyssue.solvers.quasistatic import QSSolver
from tyssue.generation import extrude
from tyssue.dynamics import model_factory, effectors
from tyssue.topology.sheet_topology import remove_face, cell_division


# 2D plotting
from tyssue.draw import sheet_view, highlight_cells

#I/O
from tyssue.io import hdf5

'''
Action attempted: energy minimisation after setup the bilayer.
'''
# Generate a bilayer structure.
bilayer = Sheet.planar_sheet_2d(identifier = 'basic2D', nx = 30, ny = 4, distx = 2, disty = 2)

bilayer.sanitize(trim_borders=True, order_edges=True)
geom.update_all(bilayer)

# Have a look of the generated bilayer.
fig, ax = sheet_view(bilayer, mode = '2D')
fig.set_size_inches(10,10)

# Update bilayer specs to add attributes for energy minimization.
nondim_specs = config.dynamics.quasistatic_plane_spec()
bilayer.update_specs(nondim_specs, reset = True)

# Perform energy minimization.
solver = QSSolver()
res = solver.find_energy_min(bilayer, geom, pmodel)
fig, ax = sheet_view(bilayer)
fig.set_size_inches(12, 5)




"""
This is the end of the script.
"""
