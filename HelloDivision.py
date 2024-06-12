# -*- coding: utf-8 -*-
"""
This is for first drawing of proliferation and make sure I have parameter control of different layers.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Supress pandas warning

import matplotlib as matplot

from tyssue import Sheet #import core object
from tyssue import PlanarGeometry as geom #for simple 2d geometry
from tyssue.draw import sheet_view #for sheet view

import pandas as pd

# What we're here for
from tyssue.topology.sheet_topology import type1_transition
from tyssue.topology.base_topology import collapse_edge, remove_face
from tyssue.topology.sheet_topology import split_vert as sheet_split
from tyssue.topology.bulk_topology import split_vert as bulk_split
from tyssue.topology import condition_4i, condition_4ii

## model and solver
from tyssue.dynamics.sheet_vertex_model import SheetModel as model
from tyssue.solvers.quasistatic import QSSolver
from tyssue.generation import extrude
from tyssue.dynamics import model_factory, effectors

# 2D plotting
from tyssue.draw import sheet_view, highlight_cells

#I/O
from tyssue.io import hdf5
#plt.style.use('bmh')

import logging


'''
The following code draw a bilayer structure, with each layer only have a single
layer of cells. With more cells than the preivous bilayer.

'''
#start with specifying the properties of the sheet:
#the sheet is named 'basic2D', cell number on x-axis =6, y-axis=7 and distance between 2 cells along x and y are both 1
bilayer = Sheet.planar_sheet_2d(identifier = 'basic2D', nx = 30, ny = 4, distx = 2, disty = 2)
geom.update_all(bilayer) #generate the sheet

# =============================================================================
# #sheet_view() function displays the created object in a matplotlib figure
# fig,ax = sheet_view(sheet) 
# fig.set_size_inches(10,10)
# =============================================================================

bilayer.sanitize(trim_borders=True, order_edges=True)
geom.update_all(bilayer)
# We pass an option to display the edge directions:
fig, ax = sheet_view(bilayer, mode = '2D')
fig.set_size_inches(10,10)


####

'''
Try to combine two sheets
'''

#Generate the second bilayer sheet, try combine them.
bilayer2 = Sheet.planar_sheet_2d(identifier = 'basic2D', nx = 30, ny = 4, distx = 1, disty = 1)
geom.update_all(bilayer2) #generate the sheet

# =============================================================================
# #sheet_view() function displays the created object in a matplotlib figure
# fig,ax = sheet_view(sheet) 
# fig.set_size_inches(10,10)
# =============================================================================

bilayer2.sanitize(trim_borders=True, order_edges=True)
geom.update_all(bilayer2)
# We pass an option to display the edge directions:
fig, ax = sheet_view(bilayer2, mode = '2D')
fig.set_size_inches(10,10)


bilayer.vert_df
type(bilayer.vert_df)
vert_frames = [bilayer.vert_df, bilayer2.vert_df]
vert_resulted = pd.concat(vert_frames)

edge_frames = [bilayer.edge_df, bilayer2.edge_df]
edge_resulted = pd.concat(edge_frames)

face_frames = [bilayer.face_df, bilayer2.face_df]
face_resulted = pd.concat(face_frames)

results = [vert_resulted, edge_resulted, face_resulted]
geom.update_all(results)  # gives error, cannot combine them in list concat way.



'''
End of the sheets-combining block via dataframes.
'''


'''
Try sheets-combining via hdf5 files.
'''
# Generate the second sheet.
bilayer2 = Sheet.planar_sheet_2d(identifier = 'basic2D', nx = 30, ny = 4, distx = 1, disty = 1)
geom.update_all(bilayer2) #generate the sheet

# =============================================================================
# #sheet_view() function displays the created object in a matplotlib figure
# fig,ax = sheet_view(sheet) 
# fig.set_size_inches(10,10)
# =============================================================================

bilayer2.sanitize(trim_borders=True, order_edges=True)
geom.update_all(bilayer2)
# We pass an option to display the edge directions:
fig, ax = sheet_view(bilayer2, mode = '2D')
fig.set_size_inches(10,10)

# export bilayer and bilayer2 into two hdf 5 files, then combine them into one.
from tyssue.io import hdf5

#Writing into files.
hdf5.save_datasets('bilayer_data.hdf5', bilayer) 
hdf5.save_datasets('bilayer2_data.hdf5', bilayer2)

#Combine the two hdf5 files.



'''
End of the sheet-combining block via hdf5 files.
'''

solver = QSSolver()
datasets = bilayer.datasets
specs = config.geometry.cylindrical_sheet()
sheet = Sheet('emin', datasets, specs)
geom.update_all(sheet)

nondim_specs = config.dynamics.quasistatic_sheet_spec()
dim_model_specs = model.dimensionalize(nondim_specs)
sheet.update_specs(dim_model_specs, reset=True)

solver_settings = {'options': {'gtol':1e-4}}

sheet.get_opposite()
sheet.vert_df.is_active = 0

active_edges = (sheet.edge_df['opposite'] > -1)
active_verts = np.unique(sheet.edge_df[active_edges]['srce'])

sheet.vert_df.loc[active_verts, 'is_active'] = 1

fig, ax = sheet_view(sheet, ['z', 'x'],
                     edge={'head_width': 0.5},
                     vert={'visible': False})
fig.set_size_inches(10, 6)

type1_transition(sheet, 82)
geom.update_all(sheet)

res = solver.find_energy_min(bilayer, geom, model, **solver_settings)
fig, ax = sheet_view(sheet, mode="quick", coords=['z', 'x'])




'''
This is the end of the script.
'''