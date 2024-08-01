# -*- coding: utf-8 -*-
"""
This script is to run a more automated simulation of cell division.
This simulation aims to provide better understanding on the data frame changes
during the process.
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

# 2D plotting
from tyssue.draw import sheet_view, highlight_cells


""" start the project """
# Generate the cell sheet as three cells.
sheet = Sheet.planar_sheet_2d('face', nx = 3, ny=4, distx=2, disty=2)
sheet.sanitize(trim_borders=True)
geom.update_all(sheet)
sheet_view(sheet, mode = '2D')


# Add more mechanical properties, take four factors
# line tensions; edge length elasticity; face contractility and face area elasticity
new_specs = model_factory([effectors.LineTension, effectors.LengthElasticity, effectors.FaceContractility, effectors.FaceAreaElasticity])

sheet.update_specs(new_specs.specs, reset = True)
geom.update_all(sheet)

fig, ax = sheet_view(sheet, mode = '2D')


# Now we are going to 









""" 
This is the end of the script.
"""
