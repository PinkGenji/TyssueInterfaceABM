# -*- coding: utf-8 -*-
"""
This file is for learning 05Rearrangements of the tyssue package.
"""

import os
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
from tyssue.draw import sheet_view, highlight_cells

#I/O
from tyssue.io import hdf5
plt.style.use('bmh')

import logging

'''
Type 1 Transition:

First, we generate the initial cells.

'''

geom = SheetGeometry


solver = QSSolver()

wd = r"C:\Users\lyu195\Documents\GitHub\tyssueHello"
os.chdir(wd)

os.path.isfile('small_hexagonal.hf5')  #Check working directory set correctly.

h5store = 'small_hexagonal.hf5'

datasets = hdf5.load_datasets(h5store, data_names=['face', 'vert', 'edge'])
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

'''
Type 1 transition starts:
'''

type1_transition(sheet, 82)
geom.update_all(sheet)

res = solver.find_energy_min(sheet, geom, model, **solver_settings)
fig, ax = sheet_view(sheet, mode="quick", coords=['z', 'x'])

# Closer look using sheet_view:

fig, ax = sheet_view(sheet, ['z', 'x'], mode="quick")

ax.set_xlim(3, 10)
ax.set_ylim(-7.5, -2.5)

ax.set_aspect('equal')
fig.set_size_inches(8, 5)


fig, mesh = sheet_view(sheet, mode='3D')
fig

res = solver.find_energy_min(sheet, geom, model)
print(res['success'])
fig, ax = sheet_view(sheet, ['z', 'x'], mode="quick")

sheet.validate()

'''
Type 1 transitions can also be performed on border faces.
'''

from tyssue.generation import three_faces_sheet

# First plot

sheet = Sheet('3f', *three_faces_sheet())
geom.update_all(sheet)
fig, ax = sheet_view(sheet, edge={'head_width': 0.05})
fig.set_size_inches(8, 8)

for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)

# Second plot
type1_transition(sheet, 0, multiplier=2)
sheet.reset_index()

geom.update_all(sheet)

fig, ax = sheet_view(sheet, edge={'head_width': 0.05})
fig.set_size_inches(8, 8)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)

for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)

# Third plot
type1_transition(sheet, 16, multiplier=5)

geom.update_all(sheet)

fig, ax = sheet_view(sheet, edge={'head_width': 0.05})
fig.set_size_inches(8, 8)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)

for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)

# Fourth plot
type1_transition(sheet, 17,  multiplier=5)

geom.update_all(sheet)
print(sheet.validate())

fig, ax = sheet_view(sheet, edge={'head_width': 0.05})
fig.set_size_inches(8, 8)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)

for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.1, vert)


'''
Rosette:

Create a small patch of cells in 2D and a simple mechjanical model:

'''

sheet = Sheet.planar_sheet_2d('flat', 30, 30, 1, 1, noise=0.2)
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

# Gradient descent

solver = QSSolver()

res = solver.find_energy_min(sheet, geom, model)

fig, ax = sheet_view(sheet, mode="quick")
for f, (x, y) in sheet.face_df[["x", "y"]].iterrows():
    ax.text(x, y, f)

# Rearrange:

sheet = Sheet.planar_sheet_2d('flat', 30, 30, 1, 1, noise=0.001)
geom = PlanarGeometry

to_cut = sheet.cut_out([(0.1, 6), (0.1, 6.)])
sheet.remove(to_cut, trim_borders=True)
sheet.sanitize(trim_borders=True)
geom.center(sheet)
geom.update_all(sheet)

remove_face(sheet, 13)
geom.update_all(sheet)

fig, ax = sheet_view(sheet, mode="quick")
for f, (x, y) in sheet.face_df[["x", "y"]].iterrows():
    ax.text(x, y, f)

'''
Formation of rosettes (Finegan et al. 2019):

Formation of a 4-way vertex: a 4-way vertex is formed whenever two vertices: i 
and j, of rank 3 (number of cells sharing the vertex) are located less
than a minimum threshold distance apart (much smaller than a typical cell diameter).

In this case, we merge the two vertices into a single vertex lcoated at their midpoint
and all cells previously connected to vertcies i and j now share a common vertex
of rank 4.

"Rosette vertex rank increase": Extending the principle of 4-way vertices,
we allow for a vertex of rank m to merge with an existing vertex of rank n to 
form a hole of rank (n+m-2). This occurs whenever two vertices i and j are
located less than a minimum threshold distance apart. In this case, the vertex
with higher degree remains in position, while the other vertex is merged
into it. Vertices with rank > 4 are termed rosette vertices.

'''

# Merge vertices, or, said otherwise, collapse an edge:

fig, ax = sheet_view(sheet, mode="quick", edge={"alpha": 0.5})
center_edge = sheet.edge_df.eval("sx**2 + sy**2").idxmin()
ax.scatter(sheet.edge_df.loc[center_edge, ["sx", "tx"]],
           sheet.edge_df.loc[center_edge, ["sy", "ty"]])

collapse_edge(sheet, center_edge)
sheet.update_rank()

geom.update_all(sheet)
fig, ax = sheet_view(sheet, mode="quick", ax=ax, edge={"alpha": 0.5})

print("Maximum vertex rank: ", sheet.vert_df['rank'].max())

# Rearrange:
sheet.update_specs(specs, reset=False)
res = solver.find_energy_min(sheet, geom, model)

fig, ax = sheet_view(sheet, mode="quick", edge={"alpha": 0.5})


# Do it again to increase rank:

# Plot to be overlayed:
    
fig, ax = sheet_view(sheet, mode="quick", edge={"alpha": 0.5})

# overlay:

for i in range(4):
    center_edge = sheet.edge_df.eval("sx**2 + sy**2").idxmin()
    collapse_edge(sheet, center_edge)
    geom.update_all(sheet)
    res = solver.find_energy_min(sheet, geom, model)
    sheet.update_rank()

fig, ax = sheet_view(sheet, mode="quick", ax=ax, edge={"alpha": 0.5})

print("Maximum vertex rank: ", sheet.vert_df['rank'].max())


'''
Rosettes resolution:

'''












'''
This is the end of the file.
'''
