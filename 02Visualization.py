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
This is the end of the file.
'''
