# -*- coding: utf-8 -*-
"""
This script is about drawing different initial configuration of the vertex 
network.

"""

import matplotlib as matplot
from tyssue import Sheet #import core object
from tyssue import PlanarGeometry as geom #for simple 2d geometry
from tyssue.draw import sheet_view #for sheet view

'''
The following code draw a bilayer structure, with each layer only have a single
layer of cells.

'''
#start with specifying the properties of the sheet:
#the sheet is named 'basic2D', cell number on x-axis =6, y-axis=7 and distance between 2 cells along x and y are both 1
sheet = Sheet.planar_sheet_2d(identifier = 'basic2D', nx = 10, ny = 4, distx = 1, disty = 1)
geom.update_all(sheet) #generate the sheet

# =============================================================================
# #sheet_view() function displays the created object in a matplotlib figure
# fig,ax = sheet_view(sheet) 
# fig.set_size_inches(10,10)
# =============================================================================

sheet.sanitize(trim_borders=True, order_edges=True)
geom.update_all(sheet)
# We pass an option to display the edge directions:
fig, ax = sheet_view(sheet, mode = '2D')
fig.set_size_inches(10,10)


'''
The Following code draw 4 layers of cells in total. Two layers of smooth cells,
and 2 layers of classic vertex cells.

'''

#start with specifying the properties of the sheet:
#the sheet is named 'basic2D', cell number on x-axis =6, y-axis=7 and distance between 2 cells along x and y are both 1
sheet = Sheet.planar_sheet_2d(identifier = 'basic2D', nx = 10, ny = 6, distx = 1, disty = 1)
geom.update_all(sheet) #generate the sheet


sheet.sanitize(trim_borders=True, order_edges=True)
geom.update_all(sheet)
# We pass an option to display the edge directions:
fig, ax = sheet_view(sheet, mode = '2D')
fig.set_size_inches(10,10)












'''
This is the end of the script.
'''
