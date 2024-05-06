# -*- coding: utf-8 -*-
"""
This file is to learn the code related to geometry of the tyssue package.
"""

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from tyssue import Sheet
from tyssue import PlanarGeometry
from tyssue.generation import generate_ring
from tyssue import config
from tyssue.draw import sheet_view

'''
A Geometry class is a stateless class holding functions to compute geometrical
aspects of an epithelium, such as the edge lengths or the cells volume. 
As they are classes, inheritance can be used to define more and more specialized geometries.

For the user, a geometry class is expected to have an update_all method that
takes an Epithelium instance as sole argument.

Calling this method will compute the relevant geometry on the epithelium.
'''

sheet_2d = Sheet.planar_sheet_2d('planar', nx = 6, ny = 6,
                                 distx = 1, disty = 1)
sheet_2d.sanitize(trim_borders=True, order_edges=True)
fig, ax = sheet_view(sheet_2d)

'''
Displacing vertices

Most of the geometry is purely defined by the vertex positions.
It is possible to change those by modifying directly the vertex dataframe.
For example, we can centre the vertices around 0 like so:
'''
com = sheet_2d.vert_df[sheet_2d.coords].mean(axis=0)
print("Sheet centre of mass: ")
print(com)

# Translate vertices by -
sheet_2d.vert_df[sheet_2d.coords] -= com
print("New centre of mass: ")
print(sheet_2d.vert_df[sheet_2d.coords].mean(axis=0))

# The view is not changed tho:
fig, ax = sheet_view(sheet_2d)

# To propagate the change in vertex positions, we need to update the geometry:
PlanarGeometry.update_all(sheet_2d)
fig, ax = sheet_view(sheet_2d)


'''
Closed 2D geometry
We can also use the generate_ring function to create a 2D ring of 4-sided cells.

'''
ring = generate_ring(Nf=24, R_in = 12, R_out=14)
PlanarGeometry.update_all(ring)
fig, ax = sheet_view(ring)

# There is an AnnularGeometry class that does all PlanarGeometry does,
# plus computing the "lumen" volume, here the area inside the ring.
from tyssue.geometry.planar_geometry import AnnularGeometry

AnnularGeometry.update_all(ring)
print(ring.settings["lumen_volume"])

'''
Sheet geometry in 2.5D

The SheetGeometry class computes the geoemtry of a 2D surface mesh embeded in 3D.
The positions of the vertices and edges are thus defined in 3D.

'''
from tyssue import SheetGeometry

sheet_3d = Sheet.planar_sheet_3d('sheet', nx=6, ny=6, distx=1, disty=1)
sheet_3d.sanitize(trim_borders=True)
SheetGeometry.update_all(sheet_3d)

# The height columns in the following df can be used to compute pseudo-volume
# for each face, computed as the face area times its height.
sheet_3d.vert_df.head()
sheet_3d.face_df.head()

'''
Relative coordinates in edge_df

The edge df stores a copy of the face and source and target vertices positions,
and other relative values.

'''

sheet_3d.edge_df.head()

'''
Closed sheet in 2.5D

For closed surfaces, a ClosedSheetGeometry is available. Calling update_all computes
the enclosed volume of the sheet, and sotres it in the settings attribute as 'lumen_vol'

Note: The following codes need to be fixed.
'''

from tyssue.geometry.sheet_geometry import ClosedSheetGeometry
from tyssue.generation import ellipsoid_sheet

ellipso = ellipsoid_sheet(a = 12, b = 12, c = 21, n_zs = 12)
ClosedSheetGeometry().update_all(ellipso)

lumen_vol = ellipso.seetings['lumen_vol']
print(f"Lumen volume: {lumen_vol: .0f}")

fig, (ax0, ax1) = plt.subplot(1, 2)

fig, ax0 = sheet_view(ellipso, coords=['z','y'], ax=ax0)
fig, ax1 = sheet_view(ellipso, coords = ['x','y'], ax=ax1)

import ipyvolume as ipv
ipv.clear()
fig, mesh = sheet_view(ellipso, mode='3D')
fig


'''
Monolayer (working codes)

To represent monolayers, we add a cell element and dataframe to the datasets.

'''

# One way to create a monolayer is to extrude it from a sheet.
# That means duplicating the 2D mesh to represent the basal surface, and
# linking both meshes together to form lateral faces.

from tyssue import Monolayer, MonolayerGeometry, ClosedMonolayerGeometry
from tyssue.generation import extrude

extruded = extrude(sheet_3d.datasets, method = 'translation')
specs = config.geometry.bulk_spec()
monolayer = Monolayer('mono', extruded, specs)
MonolayerGeometry.update_all(monolayer)
MonolayerGeometry.center(monolayer)

monolayer.cell_df.head()

# In cell_df, the num_face is self-explanatory, the num_ridges means the number of half edges on the face.
# Similar to edges, in 3D each face is a 'half-face'.
# The interface between two cells consists of two faces, one per cell.

import ipyvolume as ipv
ipv.clear()
fig, mesh = sheet_view(monolayer, mode = '3D')
fig

monolayer.face_df['segment'].unique()

apical_faces = monolayer.face_df[
    monolayer.face_df['segment'] == 'apical'
    ]

basal = monolayer.get_sub_sheet('basal')
fix, ax = sheet_view(basal, coords = ['x','y'], edge = {'head_width': 0.1})


'''
Closed Monolayer

Similarly to sheet, monolayers can be closed with a defined lumen.

'''
datasets = extrude(ellipso.dataset, method = 'homotecy', scale = 0.9)

mono_ellipso = Monolayer('mono_ell',  datasets)
mono_ellipso.vert_df['z'] +=5

ClosedMonolayerGeometry.update_all(mono_ellipso)

ipv.clear()

fig, mesh = sheet_view(mono_ellipso, mode = '3D')
fig

mono_ellipso.settings

'''
Bulk tissue

Eventually, we can define arbitrary assemblies of cells.
A way to generate such a tissue is through 3D Voroni tessellation.

This part of the code seems not working.

'''

from tyssue import Epithelium, BulkGeometry
from tyssue.generation import from_3d_voronoi, hexa_grid3d
from tyssue.draw import highlight_cells
from scipy.spatial import Voronoi

cells = hexa_grid3d(4, 4, 6)
datasets = from_3d_voronoi(Voronoi(cells))
bulk = Epithelium('bulk', datasets)
bulk.reset_topo()
bulk.reset_index(order = True)
bulk.sanitize()

BulkGeometry.update_all(bulk)

# We will see next how to configure visualization.
bulk.face_df['visible'] = False

highlight_cells(bulk, [12,4])
ipv.clear()
fig2, mesh = sheet_view(bulk, mode = "3D", face = {"visible":True})
fig2


'''
Advanced example: better initial cells

Due to the artfacts in the Voronoi tessellation at the boundaries of the above epithelium,
the cells are ugly, with verticies protruding away from the tissue.

Here we show how to bring the vertices too far from the cell at a closer distance to the cells.

This will be the occasion to apply 'upcasting' and 'downcasting'.

The algorithm is simple: for each vertex belonging to only one cell, bring the vertex at
a distance equal to the median cell-vertex distance towards the cell.
'''

# step 1: get the vertices that belong to a single cell. 
bulk.vert_df['num_cells'] = bulk.edge_df.groupby('srce').apply(lambda df:df['cell'].unique().size)
bulk.vert_df['num_cells'].head()

# step 2: create a binary mask over vertices with only one neighbour cell.
lonely_vs = bulk.vert_df['num_cells']<2

# step 3: for each source vertex, compute the vector from cell centre to the vertex, its length, and the median distance.
rel_pos = bulk.edge_df[['sx','sy','sz']].to_numpy() - bulk.edge_df[['cx', 'cy', 'cz']].to_numpy()

rel_dist = np.linalg.norm(rel_pos, axis=1)
med_dist = np.median(rel_dist)
# the displacement we need to apply is parallel to the cell-to-vertex vector, and can be expressed as:
displacement = rel_pos*(med_dist/rel_dist-1)[:,np.newaxis]

'''
We use np.newaxis to multiply 2D arrays of shape(Ne,3) with 1D arrays of shape(Ne,).
This is still defined for each edge source.
We can come back to vertices by taking the mean over all outgoing edges:
'''
# create a df so we can groupby
displacement = pd.DataFrame(displacement, index = bulk.edge_df.index)
displacement['srce'] = bulk.edge_df['srce']
vert_displacement = displacement.groupby('srce').mean()

vert_displacement[~lonely_vs]=0

#apply the displacement and update the geometry.
bulk.vert_df[['x','y','z']] += vert_displacement.to_numpy()

BulkGeometry.update_all(bulk)
ipv.clear()
fig2, mesh = sheet_view(bulk, mode = '3D', face = {'visible':True})
fig2



'''
This is the end of the file.
'''
