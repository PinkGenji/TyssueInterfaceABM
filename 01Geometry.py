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
Monolayer

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

















