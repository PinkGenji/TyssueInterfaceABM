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

bilayer.data_names
type(bilayer.edge_df)

display(bilayer.edge_df.head().to_string())




'''
This is the end of the script.
'''