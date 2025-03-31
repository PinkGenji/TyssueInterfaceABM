# -*- coding: utf-8 -*-
"""
This script aims to do two major tasks:
    (1) Generate a geometry of regular polygons to represent cell sheet.
    (2) Set up and assign the cell classes we need in the model.
"""

# Load all required modules.
import numpy as np
import matplotlib.pyplot as plt
from tyssue import Sheet
from tyssue import PlanarGeometry as geom #for simple 2d geometry
# 2D plotting
from tyssue.draw import sheet_view
# import my own functions
import my_headers as mh

rng = np.random.default_rng(70)    # Seed the random number generator.

# Generate the initial cell sheet. Note: 6 horizontal and
num_x = rng.integers(10,20)
num_y = rng.integers(10,20)
sheet =Sheet.planar_sheet_2d(identifier='bilayer', nx = num_x, ny = num_y, distx = 1, disty = 1)
geom.update_all(sheet)

# remove non-enclosed faces
sheet.remove(sheet.get_invalid())

# Delete the irregular polygons.
for i in sheet.face_df.index:
    if sheet.face_df.loc[i,'num_sides'] != 6:
        mh.delete_face(sheet,i)
    else:
        continue

sheet.reset_index(order=True)   #continuous indices in all df, vertices clockwise
geom.update_all(sheet)

# Plot the figure to see the initial setup is what we want.
fig, ax = sheet_view(sheet)
for face, data in sheet.face_df.iterrows():
    ax.text(data.x, data.y, face)
plt.show()
print('Initial geometry plot generated. \n')

# Add a new attribute to the face_df, called "cell class"
sheet.face_df['cell_class'] = 'default'
total_cell_num = len(sheet.face_df)
print(f'Attribute "cell class" with value "default" created, there are {total_cell_num} total cells.\n')

"""
We have a list of all cell classes:
  STB for resting STB; STB-Ex for extruding stb; S for mature CT, G1 for growing for division CT; 
M for CT undergoing division; G2 for CT growing for maturity and F for fusing CT.
"""

print('Now start assigning cell types. \n')
cell_class_list = ['STB', 'STB-Ex', 'S', 'G1', 'M', 'G2','F']
for i in sheet.face_df.index:
    sheet.face_df.loc[i,'cell_class'] = rng.choice(cell_class_list)     # randomly select one cell class for the cell.
cell_class_table = sheet.face_df['cell_class'].value_counts()
print('The table summaries the number of different class of cells we have: ')
print(cell_class_table)


print('\n This is the end of this script. (＾• ω •＾)')
