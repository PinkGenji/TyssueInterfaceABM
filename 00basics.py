#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This document is to learn the basics of the tyssue package.
"""

import matplotlib as matplot
from tyssue import Sheet #import core object
from tyssue import PlanarGeometry as geom #for simple 2d geometry
from tyssue.draw import sheet_view #for sheet view

#start with specifying the properties of the sheet:
#the sheet is named 'basic2D', cell number on x-axis =6, y-axis=7 and distance between 2 cells along x and y are both 1
sheet = Sheet.planar_sheet_2d(identifier = 'basic2D', nx = 6, ny = 7, distx = 1, disty = 1)
geom.update_all(sheet) #generate the sheet

#sheet_view() function displays the created object in a matplotlib figure
fig,ax = sheet_view(sheet) 
fig.set_size_inches(8,8)

# A cleaner and better ordered sheet can be generated with sanitize method:
    #Give the ttisue a haircut:
sheet.sanitize(trim_borders=True, order_edges=True)
geom.update_all(sheet)
# We pass an option to display the edge directions:
fig, ax = sheet_view(sheet, mode = '2D', edge = {'head_width':0.1})
fig.set_size_inches(8,8)
'''
From the code above, we got a plot with directed edges and better odered sheet.
Note that, each edge between two cells is composed of two half-edges 
(only one half-edge is present in the border ones). 
This makes it easier to compute many cell-specific quantities, 
as well as keeping a well oriented mesh.
'''

'''
The data associated with the mesh we generated above is stored in pandas DataFrame objects,
stored in the datasets directory: datasets["edge"]; datasets["vert"]; datasets["face"]; datasets["cell"]

'''
#Let's have a look of these data:
for element, data in sheet.datasets.items():
    print(f"{element} table has shape {data.shape}")


'''
The edge_df dataframe contains most of the information, in particular, each time the geometry is 
updated with the geom.update_all() function, the position of the source and target vertices 
of each edge are copied to 'sx', 'sy' and 'tx','ty' columns repectively.
The datatype of sheet.datasets is dictionary.
See the following code:
'''
sheet.datasets['edge'].head()

sheet.face_df.head() # Similar method can be applied to facet.

sheet.edge_df.keys() # This shows the column headers of the table.

sheet.edge_df.groupby('face')['length'].mean().head() # We can compute the average.

'''
Specifications are defined as a nested dictionary, sheet.spcs
For each element ('vert', 'edge', 'facet' 'cell'), the specification defines
the columns of the corresponding DataFrame and their default values. 
An extra key at the root of the specification is called 'settings', and can hold
specific parameters, for example the arguments for an energy minimization procedure.
For example, consider the following spec dictionary:
'''
spec = {
        "vert":{
            "x":0.0,
            "y":0.0,
            "active":True
            },
        "edge":{
            "tension":0.0,
            "length":1,
            },
        "face":{
            "area":0.0,
            "alive":True,
            },
        }     # This defines an 'area' column for sheet.face_df
 
sheet.update_specs(spec)    #those columns will be added to the dataframes.
sheet.face_df.head()

sheet.update_specs({"edge":{"tension":0.0}})
sheet.edge_df['tension'].head()












'''
This is the end of the file
'''
