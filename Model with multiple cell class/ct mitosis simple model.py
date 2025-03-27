
"""
The use of this script:
Set up a model where there is an initial layer of STB and mature CT, again we need to see this model in action.
Demonstrate that you can swap CT from the mature CT group to the G2 (growing for mitosis) with some probability p.
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
num_x = 15
num_y = 4

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
print('Cell class attribute created for all cells and set value as "default". ')
for i in range(0,num_x-2):  # These are the indices of bottom layer.
    sheet.face_df.loc[i,'cell_class'] = 'S'

for i in range(num_x-2,len(sheet.face_df)):     # These are the indices of top layer.
    sheet.face_df.loc[i,'cell_class'] = 'STB'

print(f'There are {total_cell_num} total cells; equally split into "S" and "STB" classes: ')
cell_class_table = sheet.face_df['cell_class'].value_counts()
print(cell_class_table)


# Now, loop over all "S" cells, that we can change the cell property to "G2" with a probability of 0.7
S_cells = sheet.face_df.index[sheet.face_df['cell_class'] == 'S'].tolist()
print('\n Start changing "S" cell into "G2" by probability 0.7 now. \n')
for cell in S_cells:
    # Use rng to randomly generate a number between 1 and 10, this will determine the fate of the mature CT.
    cell_fate_roulette = rng.integers(1,11)
    if cell_fate_roulette <=7 :
        sheet.face_df.loc[cell,'cell_class'] = 'G2'
    else:
        continue

# Check the distribution of cell classes.
cell_class_table = sheet.face_df['cell_class'].value_counts()
print('The table summaries the number of different class of cells we have: ')
print(cell_class_table)

print('\n This is the end of this script. (＾• ω •＾)')
