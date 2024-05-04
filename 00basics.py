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
We can also using reset = True to update the column.
'''
sheet.update_specs(
    {"edge":{"tension":1.0}},
    reset = True
    )
sheet.edge_df['tension'].head()

'''
Input and Output:
The 'native' format is to save the datasets to hdf5 via pandas.HDFStore.
The io.obj also provides functions to export the junction mesh or triangulations
to the wavefront OBJ format (requires vispy), for easy import in 3D software,
such as Blender.
Here is the code to save the data in wavefront OBJ:
    obj.save_junction_mesh('junctions.obj', sheet)
The standard data format for the datasets is HDF:
'''

# The io may depends on the linux/unix system. skip it on windows.
from tyssue.io import hdf5

hdf5.save_datasets('temp_data.hdf5', sheet) # Writing a file.

# Reading a file:
dsets = hdf5.load_datasets('temp_data.hdf5')
sheet2 = Sheet('reloaded', dsets)

# Remove the file in the directory:
!rm temp_data.hdf5

'''
Specs can also be saved as json files:
'''
import json
with open ('tmp_specs.json','w') as jh:
    json.dump(sheet.specs, jh)

# reading a file:
with open ('tmp_specs.json','r') as jh:
    specs = json.load(jh)

sheet2.update_specs(specs, reset=False)

# Remove the file:
!rm tmp_specs.json


'''
upcastign and downcasting data:
It is often necessary to use a vertex-associated data on a computation that
involves faces, and other combincations of elements. Tyssue offers the upcast
and downcast mechanisms to do that.
We will see these in the following codes.
'''

# Upcasting
# We often need to access the cell related data on each of the cell's edges.
# The epithelium class and its derivatives defines utilities to make this. 
# i.e. copying the area of each face to each of its edges.

print('Faces associated with the first edges:')
print(sheet.edge_df['face'].head())
print('\n')

# First edge associated face
face = sheet.edge_df.loc[0, 'face'] # Details of using the loc[] method is in supplementary file

print('Area of cell # {}:'.format(int(face)))
print(sheet.face_df.loc[face, 'area'])

print('\n')
print('upcasted areas over the edges:')
print(sheet.upcast_face(sheet.face_df['area']).head())





'''
This is the end of the file
'''
