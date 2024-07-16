# -*- coding: utf-8 -*-
"""
This file is for learning 08DebuggingExample of the tyssue package.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pylab as plt

from tyssue import Sheet, Monolayer, config
from tyssue import SheetGeometry, PlanarGeometry


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
from tyssue.draw import sheet_view, highlight_cells, highlight_faces

#I/O
from tyssue.io import hdf5
plt.style.use('bmh')

import logging
geom = SheetGeometry

'''
Get log messages and set log level
'''

log = logging.getLogger("tyssue")
log.setLevel("DEBUG")
hand = logging.StreamHandler()
hand.setLevel("DEBUG")
log.addHandler(hand)

'''
Display indices, highlight faces
'''

from tyssue.generation import three_faces_sheet

sheet = Sheet('3f', *three_faces_sheet())
geom.update_all(sheet)
fig, ax = sheet_view(sheet, edge={'head_width': 0.05})
fig.set_size_inches(8, 8)

# This can be long for a big (Nf > 100) eptm
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
    
for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)
    
highlight_faces(sheet.face_df, [0,], reset_visible=True)

fig, ax = sheet_view(sheet, edge={'head_width': 0.05}, face={'visible': True})
fig.set_size_inches(8, 8)

for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)

for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)

'''
Use "validate()" to check the topology is correct.
'''

sheet.validate()

'''
Create a small patch of cells in 2D and a simple mechanical model
'''

print("dangling trgt:", set(sheet.vert_df.index).difference( set(sheet.edge_df.trgt)))
print("dangling srce:", set(sheet.vert_df.index).difference( set(sheet.edge_df.srce)))

sheet.edge_df.query("face == 5")

sheet = Sheet.planar_sheet_2d('flat', 30, 30, 1, 1, noise=0.001)
geom = PlanarGeometry

to_cut = sheet.cut_out([(0.1, 6), (0.1, 6)])
sheet.remove(to_cut, trim_borders=True)
sheet.sanitize(trim_borders=True)
geom.center(sheet)
geom.update_all(sheet)
sheet.update_rank()

model = model_factory(
    [
        effectors.LineTension,
        effectors.FaceContractility,
        effectors.FaceAreaElasticity
    ]
)

specs = {
    "face": {
        "contractility": 5e-2,
        "prefered_area": sheet.face_df.area.mean(),
        "area_elasticity": 1.
    },
    "edge": {
        "line_tension": 1e-2,
        "is_active": 1
    },
    "vert": {
        "is_active": 1
    },
}

sheet.update_specs(specs, reset=True)

print("dangling trgt:", set(sheet.vert_df.index).difference( set(sheet.edge_df.trgt)))
print("dangling srce:", set(sheet.vert_df.index).difference( set(sheet.edge_df.srce)))

'''
Gradient descent
'''

solver = QSSolver()

res = solver.find_energy_min(sheet, geom, model)

fig, ax = sheet_view(sheet)
for f, (x, y) in sheet.face_df[["x", "y"]].iterrows():
    ax.text(x, y, f)

'''
Remove a vertex from vert_df to create an invalid sheet
'''

v = np.random.choice(sheet.vert_df.index)

sheet.vert_df.drop([v,], axis=0, inplace=True)

set(sheet.edge_df.trgt) == set(sheet.vert_df.index)

sheet.validate()

sheet.reset_index()

sheet.validate()

set(sheet.edge_df.trgt) == set(sheet.vert_df.index)

geom.update_all(sheet)

fig, ax = sheet_view(sheet)

bad_edges = sheet.edge_df[sheet.get_invalid()]

bad_edges



'''
This is the end of the file.
'''
