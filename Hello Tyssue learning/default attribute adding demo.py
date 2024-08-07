# -*- coding: utf-8 -*-
"""
This script investigates how to add default attribute in Tyssue cells.
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


# Generate the cell sheet as three cells.
sheet = Sheet.planar_sheet_2d('face', nx = 3, ny=4, distx=2, disty=2)
sheet.sanitize(trim_borders=True)
geom.update_all(sheet)


# Add mechanical properties.
nondim_specs = nondim_specs = config.dynamics.quasistatic_plane_spec()
sheet.update_specs(nondim_specs, reset = True)
geom.update_all(sheet)

# Try add an attribute
print(sheet.face_df.keys())

cell_id_copied = list(sheet.face_df.index)
print(cell_id_copied)
sheet.face_df = sheet.face_df.assign(track_id = cell_id_copied)
print(sheet.face_df)

# perform a division
daughter = cell_division(sheet, 2, geom)
geom.update_all(sheet)

''' We can see track_id is duplicated. '''
print(sheet.face_df)











"""
This is the end of the script.
"""
