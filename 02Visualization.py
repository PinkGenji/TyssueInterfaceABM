# -*- coding: utf-8 -*-
"""
This file is for learning the 02Visualization.py of tyssue package.
"""

'''
As seen before, parameters are passed around in tyssue through specifications,
nested dictionaries of parameters. 
We use a similar mechanism to specify visulization functions.

'''
# Most visualization will be done with sheet_view function.

from pprint import pprint
import numpy as np
import pandas as pd

import matplotlib.pylab as plt
%matplotlib qt5
import ipyvolume as ipv

import tyssue

from tyssue import Sheet, SheetGeometry as geom
from tyssue.generation import three_faces_sheet
from tyssue.draw import sheet_view
from tyssue import config
from tyssue import Monolayer, config, MonolayerGeometry
from tyssue.generation import extrude

datasets, _ = three_faces_sheet()
sheet = Sheet('3cells_2D',datasets)

geom.update_all(sheet)

'''
Epithelial sheet representation

Usually in tyssue, parameters and settings are stored in a spec nested dictionary.
This dictionary has 4 (2D) or 5(3D) keys corresponding to the vertices, edges,
faces and cells plus a "setting" key for parameters affecting the whole obejct.
Defaults are defined in the tyssue.config module.

'''

# the default:
draw_specs = tyssue.config.draw.sheet_spec()
pprint(draw_specs)

# most parameters of the dictionaries should be self explanatory. 
# The draw_spec dictionary can be passed as **draw_specs to sheet_view.

fig, ax = sheet_view(sheet, ['x','y'], **draw_specs)

ax.set_xlim(-3,2.5)
ax.set_ylim(-2.75, 2.75)
fig.set_size_inches((8,8))

'''
Showing the edges half-edges:

'''
draw_specs['edge']['head_width'] = 0.1

fig, ax = sheet_view(sheet,['x','y'], **draw_specs)
ax.set_xlim(-3, 2.5)
ax.set_ylim(-2.75, 2.75)
fig.set_size_inches((8,8))


'''
Colouring:

For vertices, we can pass colour as a matplotlib colour map:

'''

# Let's add a column to sheet.vert_df
sheet.vert_df['rand'] = np.linspace(0.0, 1.0, num=sheet.vert_df.shape[0])

cmap = plt.cm.get_cmap('viridis')
color_cmap = cmap(sheet.vert_df.rand)
draw_specs['vert']['visible'] = True

draw_specs['vert']['color'] = color_cmap
draw_specs['vert']['alpha'] = 0.5
draw_specs['vert']['s'] = 500

coords = ['x','y']
fig, ax = sheet_view(sheet, coords, **draw_specs)
ax.set_xlim(-3, 2.5)
ax.set_ylim(-2.75, 2.75)
fig.set_size_inches((8,8))

'''
Filling the cells

For faces and edges, we can pass directly an array or a pd.Series:

'''
sheet.face_df['color'] = np.linspace(0.0, 1.0, num = sheet.face_df.shape[0])

draw_specs['edge']['visible'] = False

draw_specs['face']['visible'] = True
draw_specs['face']['color'] = sheet.face_df['color']
draw_specs['face']['alpha'] = 0.5

fig, ax = sheet_view(sheet, coords, **draw_specs)

# To generate thickened edges:
draw_specs['edge']['visible'] = True

draw_specs['face']['color'] = sheet.face_df['color']
draw_specs['face']['alpha'] = 0.2

edge_color = np.linspace(0.0, 1.0, num = sheet.edge_df.shape[0])

draw_specs['edge']['color'] = edge_color

# Edge width can be passed as a parameter also, but only in 2D
draw_specs['edge']['width'] = 8 * np.linspace(0.0, 1.0, num = sheet.edge_df.shape[0])

fig, ax = sheet_view(sheet, coords, **draw_specs)



'''
Numbering the faces or vertices:

In tough to debug situations, it is useful to print on the graph the face and vertex indices.

'''
fig, ax = sheet_view(sheet)
fig.set_size_inches(8, 8)

for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face, fontsize=14, color='r')

for vert, data in sheet.vert_df.iterrows():
    ax.text(data.x, data.y+0.02, vert, weight = 'bold', color = 'blue')


'''
ipyvolume based drawing for 3D:

'''

extruded = extrude(sheet.datasets, method = 'translation')
monolayer = Monolayer('mono', extruded)

MonolayerGeometry.update_all(monolayer)

ipv.clear()
fig2, mesh = sheet_view(monolayer, mode = '3D')
fig2

'''
Vertex based color:

With ipyvolume, verteces are not represented. Edge color can be specified
on a vertex basis or on an edge basis.    

'''
color = (monolayer.vert_df.x**2 + monolayer.vert_df.y**2 + monolayer.vert_df.z**2)

ipv.clear()
fig2, mesh = sheet_view(monolayer, edge = {'color': color}, mode = '3D')
fig2




'''
This is the end of the file.
'''
